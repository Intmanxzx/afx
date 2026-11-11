"""Run with an installed wheel in an empty venv, not with the source on PYTHONPATH."""
import importlib.util
from pathlib import Path
import tempfile
import threading
from urllib.request import urlopen


def main():
    for name in ('torch', 'numpy', 'torch_npu', 'nnsmith'):
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError('wheel smoke requires an empty no-deps environment')
    from autofusedebugger.fuzz.storage import ResultRepository
    from autofusedebugger.fuzz.view.server import create_server as view
    from autofusedebugger.fuzz.central.server import create_server as central
    with tempfile.TemporaryDirectory(prefix='fuzz-wheel-smoke-') as root:
        database = Path(root) / 'results.sqlite'
        ResultRepository(database).save({
            'seed': str(2*2**128+42), 'workload_id': 'a' * 64, 'oracle_id': 'ref/1',
            'backend': 'cpu', 'status': 'ok', 'shape_results': [],
        })
        for factory in (view, central):
            with factory(database, port=0) as server:
                thread = threading.Thread(target=server.serve_forever)
                thread.start()
                try:
                    url = getattr(server, 'viewer_url', f'http://127.0.0.1:{server.server_port}/')
                    with urlopen(url, timeout=10) as response:
                        assert b'charts.js' in response.read()
                    for asset in ('app.js', 'charts.js', 'style.css'):
                        with urlopen(url + asset, timeout=10) as response:
                            assert response.status == 200 and len(response.read()) > 100
                finally:
                    server.shutdown()
                    thread.join()
    print('Installed wheel: offline viewer and central HTTP/assets passed without model dependencies')


if __name__ == '__main__':
    main()
