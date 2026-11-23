"""Dependency boundaries after separating workloads, execution and control."""
import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parents[2] / 'autofusex'


def test_nnsmith_imports_are_owned_by_fuzz():
    for path in ROOT.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module or ''] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if any(name.split('.')[0] == 'nnsmith' for name in names):
                assert path.is_relative_to(ROOT / 'workloads/fuzz'), path


def test_generation_and_contracts_do_not_depend_on_execution_layers():
    for path in [ROOT / 'workloads/fuzz/contracts.py', ROOT / 'workloads/fuzz/tensors.py', *(ROOT / 'workloads/fuzz/generation').glob('*.py')]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not set((node.module or '').split('.')) & {'backends', 'execution', 'storage', 'cli', 'torch', 'numpy'}, path


def test_independent_module_entry_without_heavy_imports():
    code = '''
import importlib.abc,sys,runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','torch_npu','nnsmith'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
sys.argv=['autofusex','doctor']
runpy.run_module('autofusex',run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Single NNSmith ATen E/F protocol' in result.stdout


def test_view_and_storage_do_not_import_execution_backends():
    forbidden = {'generation', 'backends', 'torch', 'numpy', 'nnsmith'}
    for folder in ('execution/storage', 'viewer'):
        for path in (ROOT / folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [node.module or ''] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
                for name in names:
                    assert not set(name.split('.')) & forbidden, path
