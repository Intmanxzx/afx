import json
from fastapi.testclient import TestClient
from autofusex.server import auth, client_access
from autofusex.server.server import create_app
from autofusex.server.store import Store


def test_gate_is_live_user_scoped_and_preserves_existing_routes(tmp_path):
    store = Store(tmp_path)
    for user in ('alice', 'bob'):
        auth.add_user(store, user, 'developer')
    headers = {u: {'Authorization': 'Bearer '+auth.issue_token(store, u)} for u in ('alice','bob')}
    with TestClient(create_app(store)) as client:
        body = {'machine_id': 'pilot'}
        assert client.post('/api/client/heartbeat', json=body, headers=headers['alice']).status_code == 200
        client_access.write(tmp_path, {'enabled': True, 'clients': []})
        response = client.post('/api/client/heartbeat', json=body, headers=headers['alice'])
        assert response.status_code == 503
        assert response.headers['Retry-After'] == '60'
        assert response.json()['error']['code'] == 'client_paused'
        assert client.get('/api/me', headers=headers['alice']).status_code == 200
        assert client.post('/api/client/heartbeat', json=body).status_code == 401
        client_access.write(tmp_path, {'enabled': True, 'clients': [{'user_id': 'alice', 'machine_id': 'pilot'}]})
        assert client.post('/api/client/heartbeat', json=body, headers=headers['alice']).status_code == 200
        assert client.post('/api/client/heartbeat', json=body, headers=headers['bob']).status_code == 503
        assert client.post('/api/client/executions', json={'client': {'machine_id':'other'}}, headers=headers['alice']).status_code == 503
        # Admitted requests still go through the normal schema validation.
        assert client.post('/api/client/executions', json={'client': body}, headers=headers['alice']).status_code == 400
        (tmp_path/client_access.FILENAME).write_text('broken')
        assert client.post('/api/client/heartbeat', json=body, headers=headers['alice']).status_code == 503
        assert client.get('/api/me', headers=headers['alice']).status_code == 200
        client_access.write(tmp_path, {'enabled': False, 'clients': []})
        assert client.post('/api/client/heartbeat', json=body, headers=headers['bob']).status_code == 200


def test_cli_can_enable_before_database_initialization(tmp_path):
    from autofusex.server.cli import main
    policy=tmp_path/'policy.json'
    policy.write_text(json.dumps({'enabled':True,'clients':[]}))
    home=tmp_path/'data'
    assert main(['--data',str(home),'client-access','apply',str(policy)]) == 0
    assert not (home/'core.db').exists()
    assert client_access.read(home)['enabled']


def test_legacy_upload_can_retry_after_gate_reopens_without_duplicates(tmp_path):
    from test_platform import body, item
    store=Store(tmp_path)
    auth.add_user(store,'alice','developer')
    headers={'Authorization':'Bearer '+auth.issue_token(store,'alice')}
    payload=body([item('gate-upload-1')])
    with TestClient(create_app(store)) as client:
        client_access.write(tmp_path, {'enabled':True,'clients':[]})
        response=client.post('/api/client/executions',json=payload,headers=headers)
        assert response.status_code==503
        assert store.reader().execute('SELECT count(*) FROM executions').fetchone()[0]==0
        client_access.write(tmp_path, {'enabled':True,'clients':[{'user_id':'alice','machine_id':'m1'}]})
        response=client.post('/api/client/executions',json=payload,headers=headers)
        assert response.status_code==200,response.text
        assert response.json()['data']['accepted']==1
        assert client.post('/api/client/executions',json=payload,headers=headers).json()==response.json()
        assert store.reader().execute('SELECT count(*) FROM executions').fetchone()[0]==1
