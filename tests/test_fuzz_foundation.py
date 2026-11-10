"""D0/D1 tests: stdlib-only paths must work without execution dependencies."""
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from autofusedebugger.fuzz.config import setup
from autofusedebugger.fuzz.contracts import (
    CaseBundle, CaseSpec, EvaluationConfig, ExecutionConfig, Seed,
    ShapeResult, TensorSpec, canonical_json,
)
from autofusedebugger.fuzz.diagnostics import GenerationTrace, first_divergence


def test_import_and_light_cli_do_not_load_dependencies_or_mutate_environment(tmp_path):
    code = '''
import importlib.abc, os, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'nnsmith', 'numpy'}:
            raise AssertionError('heavy import: '+fullname)
sys.meta_path.insert(0, Block())
before = dict(os.environ)
import autofusedebugger
import autofusedebugger.cli
from autofusedebugger.fuzz.cli import main
assert main(['maintenance', 'doctor']) == 0
from autofusedebugger.fuzz.config import setup
setup(sys.argv[1])
from autofusedebugger.fuzz.cli import replay_parser
replay_parser()
assert before == dict(os.environ)
'''
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)


def test_setup_preserves_identity_and_unknown_config(tmp_path):
    key = tmp_path / "admin.key"
    key.write_text("do-not-print-me")
    first = setup(tmp_path, user_id="user-7", display_name="unique-name", key_file=key, cpu_workers=2)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config["future_extension"] = {"keep": True}
    path.write_text(json.dumps(config))
    second = setup(tmp_path)
    assert first == second
    assert "do-not-print-me" not in json.dumps(first) + path.read_text()
    assert json.loads(path.read_text())["future_extension"] == {"keep": True}
    assert (tmp_path / "results").is_dir()


def test_setup_damaged_config_not_replaced(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"version":99}')
    with pytest.raises(ValueError):
        setup(tmp_path)
    assert path.read_text() == '{"version":99}'
    assert not (tmp_path / ".setup.lock").exists()


def test_setup_lock_does_not_replace_source(tmp_path):
    (tmp_path / ".setup.lock").touch()
    with pytest.raises(FileExistsError):
        setup(tmp_path)
    assert not (tmp_path / "config.json").exists()


@pytest.mark.parametrize("token", ["042", "v2:small-v1:42", "v1:small:42", "v1:small-v1:042", "-1"])
def test_seed_rejects_ambiguous_or_unknown_protocol(token):
    with pytest.raises(ValueError):
        Seed.parse(token)


def test_contracts_are_immutable_and_do_not_pretend_generator_exists():
    seed = Seed.parse("42")
    assert str(seed) == "42"
    spec = TensorSpec("x", "float32", (2, 3), (3, 1))
    bundle = CaseBundle(CaseSpec(seed, canonical_json({"ops": ["add"]}), (spec,), ()), (bytes(24),))
    with pytest.raises(FrozenInstanceError):
        bundle.storages = ()
    with pytest.raises(TypeError):
        CaseBundle(bundle.spec, (bytearray(24),))
    with pytest.raises(ValueError):
        ExecutionConfig("ascendc", "npu:0", scheduling="split")
    with pytest.raises(ValueError):
        EvaluationConfig(measured_steps=0)
    with pytest.raises(ValueError):
        ShapeResult("ascendc", 0, "error", error_summary="x" * 501)


def test_first_divergence_reports_stage_axis_and_input_order():
    a = GenerationTrace()
    a.record("graph", {"broadcast_axes": [0, 2]})
    a.record("inputs", [{"name": "a", "shape": [1, 8]}, {"name": "b", "shape": [4, 8]}])
    b = json.loads(json.dumps(a.stages))
    b[0]["payload"]["broadcast_axes"][1] = 1
    result = first_divergence(a.stages, b)
    assert result["stage"] == "graph"
    assert result["path"] == "$.payload.broadcast_axes[1]"
    b = json.loads(json.dumps(a.stages))
    b[1]["payload"].reverse()
    assert first_divergence(a.stages, b)["stage"] == "inputs"
    assert first_divergence(a.stages, a.stages) is None


def test_tensor_diagnostics_include_values_layout_and_alias():
    np = pytest.importorskip("numpy")
    from autofusedebugger.fuzz.diagnostics import tensor_signature
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    sig = tensor_signature([("x", x), ("view", x.T), ("copy", x.copy())])
    assert sig[0]["alias_group"] == sig[1]["alias_group"] != sig[2]["alias_group"]
    assert sig[1]["stride"] == [1, 3]
    assert sig[0]["sha256"] == sig[2]["sha256"]
    changed = x.copy()
    changed[0, 0] = 5
    assert tensor_signature([("x", changed)])[0]["sha256"] != sig[0]["sha256"]


def test_dependency_check_never_applies_patches_implicitly(monkeypatch, tmp_path):
    from autofusedebugger.nnsmith_integration import _nnsmith_patch as patch
    entry = patch._AscendcPatch("missing.patch", lambda: False)
    monkeypatch.setattr(patch, "_ASCENDC_PATCHES", (entry,))
    monkeypatch.setattr(patch, "_find_nnsmith_package_parent", lambda: tmp_path)
    monkeypatch.setattr(patch, "_apply_one_patch", lambda *args: pytest.fail("shared dependency mutation"))
    with pytest.raises(patch.NNSmithPatchError, match="never modified"):
        patch.ensure_ascendc_patch_applied()


def test_replay_dispatch_and_removed_legacy_entry(monkeypatch):
    import autofusedebugger.cli as cli
    from autofusedebugger.fuzz import cli as fuzz_cli
    seen = []
    monkeypatch.setattr(fuzz_cli, "execute_args", lambda args: seen.append(args.seed) or 0)
    assert cli.main(["fuzz", "replay", "42"]) == 0
    assert seen == ["42"]
    with pytest.raises(SystemExit):
        fuzz_cli.main(["run", "--seed-token", "v1:preview-dynamic-v1:42"])


def test_input_size_and_order_report_first_binding_divergence():
    np = pytest.importorskip("numpy")
    from autofusedebugger.fuzz.diagnostics import InputSignatureError, validate_input_signature
    expected = [{"name": "x", "dtype": "float32", "shape": [2, 3]}]
    with pytest.raises(InputSignatureError) as error:
        validate_input_signature(expected, {"x": np.zeros((3, 2), dtype="float32")}, stage="before_backend")
    report = json.loads(str(error.value))
    assert report["stage"] == "before_backend"
    assert report["path"] == "$.payload[0].shape[0]"
    assert report["expected"] == 2 and report["actual"] == 3
    validate_input_signature(expected, {"x": np.zeros((2, 3), dtype="float32")}, stage="materialized")


def test_empty_or_malformed_trace_is_not_reported_equal():
    with pytest.raises(ValueError):
        first_divergence([], [])
    with pytest.raises(ValueError):
        first_divergence([{"stage": "graph"}], [{"stage": "graph"}])


def test_graph_trace_captures_axes_omitted_from_ir_text():
    from types import SimpleNamespace
    class IR:
        vars = {}
        def __init__(self, axis):
            op = SimpleNamespace(extra_attrs={"dim0": 0, "dim1": axis})
            self.insts = [SimpleNamespace(identifier=1, iexpr=SimpleNamespace(op=op, args=("x",)))]
        def __str__(self):
            return "y = core.Transpose(x)"
    first, second = GenerationTrace(), GenerationTrace()
    first.graph(IR(1))
    second.graph(IR(2))
    difference = first_divergence(first.stages, second.stages)
    assert difference["stage"] == "graph_extra_attrs"
    assert difference["path"] == "$.payload[0].extra_attrs.dim1"


def test_setup_short_user_flag(tmp_path, capsys):
    from autofusedebugger.fuzz.cli import main
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        main(['setup', '--help'])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert '--user ' in help_text
    assert '--user-name' not in help_text and '--display-name' not in help_text
