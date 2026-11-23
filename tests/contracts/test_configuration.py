"""The shared configuration layer must work without any workload runtime."""
import subprocess
import sys


def test_shared_configuration_without_workload_or_execution_imports():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.startswith(('autofusex.workloads.fuzz', 'autofusex.workloads', 'autofusex.server')) or name.split('.')[0] in {'torch', 'numpy', 'nnsmith'}:
            raise AssertionError(name)
sys.meta_path.insert(0, Block())
from autofusex.contracts.configuration import parse, canonical, fingerprint, quantity, merge
assert parse('enabled: true')['enabled'] is True
assert quantity('2GiB') == 2147483648
original = {'a': {'b': 1}}
assert merge(original, {'a': {'c': 2}}) == {'a': {'b': 1, 'c': 2}}
assert original == {'a': {'b': 1}}
assert fingerprint({'a': 1, 'b': 2}) == fingerprint({'b': 2, 'a': 1})
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_supervisor_transport_does_not_import_client_runtime():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.startswith(('autofusex.client.daemon', 'autofusex.execution', 'autofusex.workloads')):
            raise AssertionError(name)
sys.meta_path.insert(0, Block())
from autofusex.contracts.ipc import _socket_path, request
assert str(_socket_path('/tmp/afx-check')) == '/tmp/afx-check/manager.sock'
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
