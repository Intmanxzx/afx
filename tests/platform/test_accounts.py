import http.client
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import pytest
from autofusex.server import accounts
from autofusex.server import auth
from autofusex.server.store import Store
from autofusex.server.server import PlatformServer

@pytest.fixture
def world(tmp_path):
    store = Store(tmp_path)
    auth.add_user(store, 'root', 'admin')
    token = auth.issue_token(store, 'root')
    return store, auth.authenticate(store, 'Bearer ' + token), token


def application(store, admin, name='alice'):
    receipt = accounts.apply(store, dict(user_id=name))
    req = accounts.listing(store, admin)['requests'][0]
    accounts.review(store, admin, dict(id=req['id'], action='approve'))
    return receipt


def test_existing_token_and_one_time_claim(world):
    store, admin, old = world
    receipt = application(store, admin)
    assert accounts.status(store, receipt)['status'] == 'approved'
    token = accounts.claim(store, receipt)['token']
    assert auth.authenticate(store, 'Bearer ' + token).role == 'developer'
    with pytest.raises(accounts.AccountError):
        accounts.claim(store, receipt)
    assert auth.authenticate(store, 'Bearer ' + old).role == 'admin'
    listing = json.dumps(accounts.listing(store, admin))
    assert token not in listing and receipt['receipt'] not in listing
    assert token.encode() not in store.core_path.read_bytes()


def test_controls_and_reset(world):
    store, admin, old = world
    receipt = application(store, admin)
    token = accounts.claim(store, receipt)['token']
    accounts.manage(store, admin, dict(user_id='alice', action='disable'))
    with pytest.raises(auth.AuthError): auth.authenticate(store, 'Bearer ' + token)
    accounts.manage(store, admin, dict(user_id='alice', action='enable'))
    assert auth.authenticate(store, 'Bearer ' + token)
    accounts.manage(store, admin, dict(user_id='alice', action='role', role='admin'))
    assert auth.authenticate(store, 'Bearer ' + token).role == 'admin'
    fresh = accounts.manage(store, admin, dict(user_id='alice', action='reset-token'))['token']
    with pytest.raises(auth.AuthError): auth.authenticate(store, 'Bearer ' + token)
    assert auth.authenticate(store, 'Bearer ' + fresh)
    accounts.manage(store, admin, dict(user_id='alice', action='role', role='developer'))
    for body in [dict(user_id='root', action='disable'), dict(user_id='root', action='role', role='developer')]:
        with pytest.raises(accounts.AccountError): accounts.manage(store, admin, body)
    with pytest.raises(accounts.AccountError): accounts.manage(store, admin, dict(user_id='alice', action='role', role='viewer'))


def test_expiry_rejection_and_unauthorized(world):
    store, admin, _ = world
    receipt = accounts.apply(store, dict(user_id='alice'))
    with pytest.raises(accounts.AccountError): accounts.apply(store, dict(user_id='alice'))
    with pytest.raises(auth.AuthError): accounts.listing(store, auth.ANONYMOUS)
    with pytest.raises(auth.AuthError): accounts.review(store, auth.Principal('dev', 'developer', 99), dict(id=1, action='approve'))
    accounts.review(store, admin, dict(id=1, action='reject'))
    with pytest.raises(accounts.AccountError): accounts.claim(store, receipt)
    receipt = application(store, admin, 'bob')
    with store.write() as db: db.execute("UPDATE registration_requests SET expires_at=0 WHERE user_id='bob'")
    assert accounts.status(store, receipt)['status'] == 'expired'
    with pytest.raises(accounts.AccountError): accounts.claim(store, receipt)
    with pytest.raises(accounts.AccountError): accounts.status(store, dict(receipt='x' * 43))


def test_concurrent_claim_and_admin_reset(world):
    store, admin, _ = world
    receipt = application(store, admin)
    def claim():
        try: return accounts.claim(store, receipt)['token']
        except accounts.AccountError: return None
    with ThreadPoolExecutor(2) as pool: values = list(pool.map(lambda _: claim(), range(2)))
    assert len([v for v in values if v]) == 1
    receipt = application(store, admin, 'bob')
    token = accounts.manage(store, admin, dict(user_id='bob', action='reset-token'))['token']
    with pytest.raises(accounts.AccountError): accounts.claim(store, receipt)
    assert auth.authenticate(store, 'Bearer ' + token)


def test_http_public_and_admin_boundaries(world):
    store, _, token = world
    server = PlatformServer(store, ('127.0.0.1', 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    def request(path, body=None, bearer=None, method='POST'):
        conn = http.client.HTTPConnection(*server.server_address)
        headers = {'Content-Type': 'application/json'}
        if bearer: headers['Authorization'] = 'Bearer ' + bearer
        conn.request(method, path, json.dumps(body) if body is not None else None, headers)
        res = conn.getresponse(); data = json.loads(res.read()); status = res.status
        assert res.getheader('Cache-Control') == 'no-store'
        conn.close(); return status, data
    try:
        code, data = request('/api/registration/apply', dict(user_id='newuser'))
        assert code == 200
        receipt = data['data']
        assert request('/api/registration/status', receipt)[1]['data']['status'] == 'pending'
        assert request('/api/admin/users', method='GET')[0] == 401
        code, data = request('/api/admin/users', bearer=token, method='GET'); assert code == 200
        rid = data['data']['requests'][0]['id']
        assert request('/api/admin/users/review', dict(id=rid, action='approve'), token)[0] == 200
        code, data = request('/api/registration/claim', receipt); assert code == 200
        usertoken = data['data']['token']
        assert request('/api/admin/users', bearer=usertoken, method='GET')[0] == 403
        assert request('/api/registration/claim', receipt)[0] == 409
        for i in range(3): code, _ = request('/api/registration/apply', dict(user_id='spam'+str(i)))
        assert code == 429
    finally:
        server.shutdown(); server.server_close(); thread.join()


def test_upgrade_preserves_existing_credentials(tmp_path):
    from autofusex.server import schema
    store = Store(tmp_path)
    auth.add_user(store, 'existing', 'admin')
    token = auth.issue_token(store, 'existing')
    with store.write() as db:
        db.execute('DROP TABLE registration_requests')
        db.execute('DELETE FROM schema_migrations WHERE version=13')
    upgraded = Store(tmp_path)
    assert auth.authenticate(upgraded, 'Bearer ' + token).user_id == 'existing'
    assert upgraded.reader().execute('SELECT count(*) FROM user_tokens').fetchone()[0] == 1
    assert upgraded.reader().execute('SELECT count(*) FROM registration_requests').fetchone()[0] == 0
    with pytest.raises(ValueError): auth.add_user(upgraded, 'readonly', 'viewer')
