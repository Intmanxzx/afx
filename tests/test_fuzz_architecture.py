"""Enforce independent fuzz package and dependency direction."""
import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1] / "autofusedebugger/fuzz"


def test_fuzz_does_not_import_legacy_debugger_or_nnsmith():
    for path in ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 2, path
                module = node.module or ""
                assert not module.startswith("nnsmith"), path
                if module.startswith("autofusedebugger"):
                    assert module.startswith("autofusedebugger.fuzz"), path
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("nnsmith"), path


def test_generation_and_contracts_do_not_depend_on_outer_layers():
    for path in [ROOT / "contracts.py", ROOT / "tensors.py", *(ROOT / "generation").glob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not set((node.module or "").split(".")) & {"backends", "execution", "storage", "cli", "torch", "numpy"}, path


def test_independent_module_entry_without_heavy_or_debugger_imports():
    code = '''
import importlib.abc,sys,runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','torch_npu','nnsmith'}: raise AssertionError(name)
  if name.startswith('autofusedebugger.') and not name.startswith('autofusedebugger.fuzz'): raise AssertionError(name)
sys.meta_path.insert(0,Block())
sys.argv=['autofusedebugger.fuzz','seed','42']
runpy.run_module('autofusedebugger.fuzz',run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '"seed": "A000000016"' in result.stdout


def test_view_and_storage_do_not_depend_on_execution_layers():
    forbidden = {'generation', 'execution', 'backends', 'torch', 'numpy', 'nnsmith'}
    for folder in ('storage', 'view'):
        for path in (ROOT / folder).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert not set((node.module or '').split('.')) & forbidden, path
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not set(alias.name.split('.')) & forbidden, path
