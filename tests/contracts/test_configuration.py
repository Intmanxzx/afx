"""The shared configuration layer must work without any workload runtime."""
import subprocess
import sys


def test_shared_configuration_without_workload_or_execution_imports():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.startswith(('autofusex.fuzz', 'autofusex.workloads', 'autofusex.platform')) or name.split('.')[0] in {'torch', 'numpy', 'nnsmith'}:
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
