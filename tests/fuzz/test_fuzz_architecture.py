"""Enforce independent fuzz package and dependency direction."""
import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[2] / "autofusex/fuzz"


def test_fuzz_dependencies_stay_within_module_boundaries():
    nnsmith_adapters = {
        ROOT / 'generation' / name for name in
        ('__init__.py', 'runtime.py', 'nnsmith_sampled.py', 'nnsmith_dynamic.py', 'torch_dialect.py')
    }
    # The retained AscendC integration shares the debugger's backend hooks.
    shared_imports = {
        ROOT / 'backends/torch.py': {'autofusex.compare'},
        ROOT / 'nnsmith/fuzz_loop.py': {'autofusex.inductor_autofuse', 'autofusex.autofuse_graph'},
        ROOT / 'nnsmith/ascendc_backend.py': {'autofusex.inductor_autofuse'},
        ROOT / 'nnsmith/ascendc_opset.py': {'autofusex.inductor_autofuse'},
    }
    for path in ROOT.rglob("*.py"):
        allows_nnsmith = path in nnsmith_adapters or ROOT / 'nnsmith' in path.parents
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 2, path
                module = node.module or ""
                assert not module.startswith("nnsmith") or allows_nnsmith, path
                if module.startswith("autofusex"):
                    assert module.startswith("autofusex.fuzz") or module in shared_imports.get(path, set()), path
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("nnsmith") or allows_nnsmith, path


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
  if name.startswith('autofusex.') and not name.startswith('autofusex.fuzz'): raise AssertionError(name)
sys.meta_path.insert(0,Block())
sys.argv=['autofusex.fuzz','maintenance','doctor']
runpy.run_module('autofusex.fuzz',run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Single NNSmith ATen C2/D2 protocol' in result.stdout


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
