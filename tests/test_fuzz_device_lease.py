import os
import subprocess
import sys
import pytest
from autofusedebugger.fuzz.backends.device_lease import DeviceLease

pytestmark=pytest.mark.skipif(os.name!='posix',reason='POSIX lock')


def test_uuid_lock_excludes_other_process_and_releases(tmp_path):
    code='''
import sys
from autofusedebugger.fuzz.backends.device_lease import DeviceLease
try:
 lease=DeviceLease('physical-uuid',sys.argv[1]).acquire()
except ValueError:
 sys.exit(7)
lease.close()
'''
    lease=DeviceLease('physical-uuid',tmp_path).acquire()
    try:assert subprocess.run([sys.executable,'-c',code,str(tmp_path)]).returncode==7
    finally:lease.close()
    assert subprocess.run([sys.executable,'-c',code,str(tmp_path)]).returncode==0


def test_distinct_physical_devices_and_crash_release(tmp_path):
    first=DeviceLease('uuid-0',tmp_path).acquire()
    second=DeviceLease('uuid-1',tmp_path).acquire()
    first.close();second.close()
    code="import os,sys;from autofusedebugger.fuzz.backends.device_lease import DeviceLease;l=DeviceLease('uuid-0',sys.argv[1]).acquire();os._exit(9)"
    assert subprocess.run([sys.executable,'-c',code,str(tmp_path)]).returncode==9
    DeviceLease('uuid-0',tmp_path).acquire().close()


def test_execution_always_closes_adapter_on_failure():
    from autofusedebugger.fuzz.execution import execute
    from autofusedebugger.fuzz.generation import materialize
    from autofusedebugger.fuzz.contracts import EvaluationConfig
    class Adapter:
        name='fake'
        closed=False
        def check(self,*args):raise RuntimeError('prepare failure')
        def close(self):self.closed=True
    adapter=Adapter();result=execute(materialize('42'),adapter,EvaluationConfig())
    assert result['status']=='error' and adapter.closed
