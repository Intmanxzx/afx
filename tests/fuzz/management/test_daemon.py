import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from autofusex.client.daemon import request


def test_daemon_cli_durable_operation_without_heavy_imports(tmp_path):
    home=tmp_path/'home';home.mkdir()
    child=subprocess.Popen([sys.executable,'-m','autofusex','daemon','run','--data-dir',str(home)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            if child.poll() is not None:raise AssertionError(child.communicate())
            try:
                status=request(home,dict(action='status'));break
            except OSError:time.sleep(.05)
        else:raise AssertionError('daemon did not start')
        assert not status['paired']
        op=request(home,dict(action='submit',kind='environment.inspect',payload={},request_key='once'))
        for _ in range(100):
            result=request(home,dict(action='operation',id=op['id']))
            if result['state']=='succeeded':break
            time.sleep(.05)
        assert result['result']['environments']==[]
        second=request(home,dict(action='submit',kind='environment.inspect',payload={},request_key='once'))
        assert second['id']==op['id']
        with pytest.raises(ValueError):request(home,dict(action='submit',kind='shell',payload={'command':'true'},request_key='bad'))
        output=subprocess.check_output([sys.executable,'-m','autofusex','daemon','status','--data-dir',str(home),'--json'])
        assert json.loads(output)['ok']
        request(home,dict(action='shutdown'));assert child.wait(timeout=10)==0
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=10)


def test_management_import_is_lightweight():
    script='import sys; import autofusex.client.cli; assert not any(x in sys.modules for x in ("torch","numpy","nnsmith","torch_npu"))'
    subprocess.run([sys.executable,'-c',script],check=True)


def test_task_worker_persists_failure_message_and_traceback(tmp_path):
    run_dir = tmp_path / 'run'; run_dir.mkdir()
    spec = run_dir / 'spec.json'
    # An incomplete manifest fails before any execution dependency is loaded.
    spec.write_text(json.dumps({'directory': str(run_dir), 'spec': {}}))
    process = subprocess.run([sys.executable, '-m',
                              'autofusex.client.task_worker',
                              str(spec), '0'], capture_output=True, text=True)
    assert process.returncode == 1
    state = json.loads((run_dir / 'state.json').read_text())
    assert state['state'] == 'failed'
    assert state['error'] == 'KeyError'
    assert 'spec' in state['message']
    log = (run_dir / 'execution.log')
    # The worker itself has no redirection in this unit test; task start()
    # owns that pipe. The traceback must still be emitted to stderr for the
    # parent-created execution log to capture.
    assert 'KeyError' in process.stderr
