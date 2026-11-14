import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from autofusex.management.daemon import request


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
    script='import sys; import autofusex.management.cli; assert not any(x in sys.modules for x in ("torch","numpy","nnsmith","torch_npu"))'
    subprocess.run([sys.executable,'-c',script],check=True)
