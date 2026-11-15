"""Native extensions can write while retaining the GIL; Python os.write cannot model it."""
import os
import subprocess
import sys

import pytest

from autofusex.fuzz.core_logs import ProcessTail, capture_tail


@pytest.mark.parametrize('forward', [False, True])
def test_native_writer_holding_gil_completes(forward):
    code = f'''
import ctypes
from autofusex.fuzz.core_logs import capture_tail
lib = ctypes.PyDLL(None)
lib.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
data = ctypes.create_string_buffer(b'x' * 1048576 + b'\\nerror: native token=private-value\\n')
with capture_tail(forward={forward!r}) as tail:
    assert lib.write(2, data, len(data)-1) == len(data)-1
assert 'error: native' in tail()
assert 'private-value' not in tail()
assert len(tail()) <= 1600
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr[-2000:]


def test_restores_output_on_exception_and_repeated_capture():
    before = os.fstat(1)
    with pytest.raises(ValueError):
        with capture_tail() as tail:
            os.write(2, b'error: before exception\n')
            raise ValueError('test')
    assert 'before exception' in tail()
    assert os.fstat(1) == before
    with capture_tail() as second:
        os.write(1, b'next capture\n')
    assert 'next capture' in second()


def test_parent_tail_preserves_verdict_across_noisy_crash():
    code = "import os; os.write(2,b'MLIR Codegen Result: kFailed\\n'+b'noise\\n'*20000+b'error: native token=secret\\n'); os._exit(3)"
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail = ProcessTail(child.stdout)
    assert child.wait(timeout=10) == 3
    text = tail.finish()
    assert 'MLIR Codegen Result: kFailed' in text
    assert 'error: native' in text and 'secret' not in text
    assert len(text) <= 1600


def test_pool_timeout_keeps_native_log(tmp_path, monkeypatch):
    import socket
    from autofusex.fuzz.scheduling.pool import Pool
    from autofusex.fuzz.scheduling.budgets import Budget
    pool = Pool(Budget(cpu_workers=1), tmp_path, 0, [])
    left, right = socket.socketpair()
    class Scratch:
        def cleanup(self):
            pass
    child = subprocess.Popen([sys.executable, '-c',
        "import os,time; os.write(2,b'MLIR Codegen Result: kFailed\\nerror: native token=secret\\n'); time.sleep(30)"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    worker = dict(child=child, socket=left, tail=ProcessTail(child.stdout), scratch=Scratch())
    pool.workers.append(worker)
    monkeypatch.setattr(pool, '_worker', lambda key: worker)
    try:
        with pytest.raises(RuntimeError) as error:
            pool._run('stage', ('af-mlir', 'npu:0'), {'kind': 'first_call_combined'}, 0.5)
        assert 'TimeoutError' in str(error.value)
        assert 'MLIR Codegen Result: kFailed' in str(error.value)
        assert 'secret' not in str(error.value)
        assert child.poll() is not None and not pool.workers
    finally:
        right.close()
        pool.close()
