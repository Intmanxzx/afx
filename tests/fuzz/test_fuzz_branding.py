"""Central branding, icon delivery and accessible login markup."""
import threading
from urllib.request import urlopen
from autofusex.fuzz.server.central.server import create_server


def test_central_branding_and_navigation_order(tmp_path):
    server=create_server(tmp_path/'db',0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        base=f'http://127.0.0.1:{server.server_port}'
        for path in ('/','/report'):
            with urlopen(base+path) as response:html=response.read().decode()
            assert '<title>Autofuse一站式工作平台</title>' in html
            assert html.index('<nav ')<html.index('id="refresh-time"')<html.index('id="session-controls"')
            assert 'href="favicon.svg"' in html and 'src="favicon.svg"' in html
        with urlopen(base+'/favicon.svg') as response:
            assert response.headers['Content-Type']=='image/svg+xml'
            assert b'<svg' in response.read()
        with urlopen(base+'/session.js') as response:script=response.read().decode()
        assert '自己的 Token' not in script and '仅用于验证身份' not in script
        assert 'actions.append(cancel, submit)' in script
        assert "setAttribute('aria-label', 'Token')" in script
    finally:server.shutdown();server.server_close();thread.join()
