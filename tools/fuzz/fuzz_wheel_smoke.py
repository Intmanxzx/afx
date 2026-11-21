"""Run with an installed wheel in an empty venv, not with the source on PYTHONPATH."""
from autofusex.fuzz.client.generation.sampling import SamplingConfig
import importlib.util
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from urllib.request import urlopen


def main():
    import afx
    import autofusex
    assert importlib.util.find_spec('autofusedebugger') is None
    distribution = importlib.metadata.distribution('autofusex')
    # PEP 440 normalizes release spellings such as 0.2.2.001 to 0.2.2.1.
    assert distribution.version == '.'.join(str(int(part)) for part in autofusex.__version__.split('.'))
    assert afx.__version__ == autofusex.__version__
    assert not any(str(path).startswith('autofusedebugger/') for path in distribution.files)
    entries = {entry.name: entry.value for entry in distribution.entry_points
               if entry.group == 'console_scripts'}
    assert entries == {'afx': 'autofusex.cli:main'}, entries
    executable = Path(sys.executable).parent / ('afx.exe' if sys.platform == 'win32' else 'afx')
    with tempfile.TemporaryDirectory(prefix='afx-cli-smoke-') as cwd:
        for command in ([str(executable)], [sys.executable, '-I', '-m', 'afx'],
                        [sys.executable, '-I', '-m', 'autofusex']):
            for args in (['--help'], ['fuzz', '--help'], ['fuzz', 'replay', '--help'], ['daemon', '--help']):
                result = subprocess.run([*command, *args], cwd=cwd, capture_output=True, text=True)
                assert result.returncode == 0, result.stderr
                assert 'usage: afx' in result.stdout, result.stdout
    print('Installed wheel: autofusex metadata, afx console command and python -m afx passed')
    for name in ('torch', 'numpy', 'torch_npu', 'nnsmith'):
        if importlib.util.find_spec(name) is not None:
            raise RuntimeError('wheel smoke requires an environment without model dependencies')
    from autofusex.fuzz.client.storage import ResultRepository
    from autofusex.fuzz.server.view.server import create_server as view
    with tempfile.TemporaryDirectory(prefix='fuzz-wheel-smoke-') as root:
        database = Path(root) / 'results.sqlite'
        ResultRepository(database).save({
            'seed': str(SamplingConfig.profile('small',max_nodes=8,dtype_mode='float32').seed(42).value), 'workload_id': 'a' * 64, 'oracle_id': 'ref/1',
            'backend': 'cpu', 'status': 'ok', 'shape_results': [],
        })
        with view(database, port=0) as server:
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
    print('Installed wheel: offline viewer HTTP/assets passed without model dependencies')


if __name__ == '__main__':
    main()
