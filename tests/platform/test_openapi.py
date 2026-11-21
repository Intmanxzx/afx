import http.client
import json
import threading

from openapi_spec_validator import validate
from autofusex.platform.server import PlatformServer, create_app
from autofusex.platform.store import Store


def test_discovery_contract(tmp_path):
    app = create_app(Store(tmp_path))
    spec = app.openapi()
    validate(spec)
    live = {(r.path_format, method.lower()) for r in app.routes if getattr(r, 'include_in_schema', False)
            for method in r.methods}
    documented = {(path, method) for path, methods in spec['paths'].items() for method in methods}
    assert live == documented
    body = spec['paths']['/api/issues/{issue_id}/verify']['post']['requestBody']
    assert body['content']['application/json']['schema']['$ref'].endswith('/Verify')
    assert 'fix_sha256' in spec['components']['schemas']['Verify']['required']
    assert any(p['name'] == 'backend' and p['required']
               for p in spec['paths']['/api/overview/backend']['get']['parameters'])
    assert 'security' not in spec['paths']['/api/registration/apply']['post']
    assert spec['paths']['/api/admin/users']['get']['security']


def test_public_openapi_http(tmp_path):
    server = PlatformServer(Store(tmp_path), ('127.0.0.1', 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    try:
        connection.request('GET', '/openapi.json')
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader('Content-Type').startswith('application/json')
        spec = json.loads(response.read())
        assert spec == server.app.openapi()
        assert spec['openapi'] == '3.1.0'
        assert 'data' not in spec
        connection.request('GET', '/api/me')
        response = connection.getresponse()
        response.read()
        assert response.status == 401
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join()


def test_runtime_and_schema_share_validation(tmp_path):
    from fastapi.testclient import TestClient
    from autofusex.platform import auth
    store = Store(tmp_path)
    auth.add_user(store, 'dev', 'developer')
    token = auth.issue_token(store, 'dev')
    app = create_app(store)
    headers = {'Authorization': 'Bearer ' + token}
    with TestClient(app) as client:
        response = client.post('/api/issues/F1/verify', json={'request_key': 'x'}, headers=headers)
        assert response.status_code == 400
        assert 'fix_sha256' in response.json()['error']['message']
        assert client.post('/api/issues/claim', json={}).status_code == 401
        assert client.get('/api/config/default').status_code == 401
        assert client.get('/api/admin/users', headers=headers).status_code == 403
        assert client.get('/api/overview/backend').status_code == 400
        spec = client.get('/openapi.json').json()
        assert {} in spec['paths']['/api/cases']['get']['security']
        assert '422' not in spec['paths']['/api/issues/claim']['post']['responses']
        # Newly registered real routes appear without updating any documentation inventory.
        @app.get('/api/test-contract')
        def probe(limit: int):
            return {'limit': limit}
        app.openapi_schema = None
        assert '/api/test-contract' in client.get('/openapi.json').json()['paths']
        assert client.get('/api/test-contract?limit=bad').status_code == 400


def test_transport_limits_and_raw_export(tmp_path):
    from fastapi.testclient import TestClient
    from autofusex.platform.server import MAX_BODY
    app = create_app(Store(tmp_path))
    with TestClient(app) as client:
        response = client.post('/api/registration/apply', content=b'x' * (MAX_BODY + 1))
        assert response.status_code == 413
        response = client.post('/api/registration/apply', content=b'{}', headers={'Content-Type': 'text/plain'})
        assert response.status_code == 415
        response = client.post('/api/registration/apply', content=b'{bad', headers={'Content-Type': 'application/json'})
        assert response.status_code == 400 and response.json()['error']['code'] == 'invalid_json'
        response = client.get('/api/knowledge/export')
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('application/x-ndjson')
