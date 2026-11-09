"""v1 preview: protocol vectors, byte/storage validation and unified execution."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from autofusedebugger.fuzz.contracts import CaseBundle, EvaluationConfig, ExecutionConfig
from autofusedebugger.fuzz.protocol_v1 import materialize, plan, reference, workload_id
from autofusedebugger.fuzz.execution import execute, TorchAdapter, UnsupportedBackend


@pytest.mark.parametrize("entropy", [0, 1, 42, 389609433, 2**128-1])
def test_seed_does_not_depend_on_previous_random_history(entropy):
    import random
    token = str(entropy)
    first = materialize(token)
    random.seed(123)
    for _ in range(500):
        random.random()
    materialize("77")
    assert materialize(token) == first
    assert all(type(v) is bytes for v in first.storages)
    assert len(first.spec.shape_samples) == 3
    for index in range(len(first.spec.shape_samples)):
        assert len(reference(first, index)) == 1


def test_unknown_profile_and_legacy_integer_rejected():
    for token in ("042", "v1:preview-dynamic-v1:42", str(2**128)):
        with pytest.raises(ValueError):
            materialize(token)


def test_storage_view_bounds_checked():
    bundle = materialize("42")
    with pytest.raises(ValueError, match="storage"):
        CaseBundle(bundle.spec, (b"", *bundle.storages[1:]))
    bad = replace(bundle.spec.inputs[0], storage_offset=10000)
    with pytest.raises(ValueError, match="bounds"):
        CaseBundle(replace(bundle.spec, inputs=(bad,)), bundle.storages)


def test_stdlib_only_fresh_process_no_cache_or_rng_dependency(tmp_path):
    code = '''
import importlib.abc,sys,os,random
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','numpy','nnsmith','torch_npu'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
from autofusedebugger.fuzz.protocol_v1 import materialize,workload_id
random.seed(int(os.environ['PYTHONHASHSEED']))
print(workload_id(materialize('42')))
'''
    values = []
    for hash_seed in ("0", "123"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed,
                   PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        values.append(subprocess.check_output([sys.executable, "-c", code], cwd=tmp_path, env=env, text=True))
    assert values[0] == values[1]
    assert not list(tmp_path.iterdir())


class FakeAdapter:
    name = "test"
    def __init__(self, *, mismatch=False, fail_measurement=False):
        self.mismatch, self.fail_measurement, self.calls = mismatch, fail_measurement, 0
    def check(self, bundle, evaluation):
        pass
    def prepare(self, bundle):
        return object()
    def inputs(self, bundle, index):
        return (index,)
    def invoke(self, prepared, inputs):
        self.calls += 1
        if self.fail_measurement and self.calls > 1:
            raise RuntimeError("measurement failed")
        return (inputs[0],)
    def synchronize(self):
        pass
    def compare(self, output, expected):
        return not self.mismatch


def test_correctness_failure_still_measures_each_dynamic_shape():
    bundle = materialize("42")
    adapter = FakeAdapter(mismatch=True)
    result = execute(bundle, adapter, EvaluationConfig(performance=True, warmup_steps=1, measured_steps=2))
    assert result["status"] == "fail"
    assert adapter.calls == 12
    assert len(result["shape_results"]) == 3
    assert all(r["correctness"] == "fail" and r["performance"] == "measured" for r in result["shape_results"])
    assert result["prepare_is_compile_complete"] is False


def test_measurement_error_does_not_erase_correctness():
    result = execute(materialize("0"), FakeAdapter(fail_measurement=True),
                     EvaluationConfig(performance=True))
    assert result["shape_results"][0]["correctness"] == "pass"
    assert result["shape_results"][0]["performance"] == "error"


def test_strict_npu_performance_refused_without_lease():
    bundle = materialize("0")
    result = execute(bundle, TorchAdapter(ExecutionConfig("ascendc", "npu:0")), EvaluationConfig(performance=True))
    assert result["status"] == "unsupported"
    assert "lease" in result["error_summary"]


def test_cpu_eager_matches_independent_exact_reference():
    pytest.importorskip("torch")
    coverage = set()
    for seed in range(32):
        bundle = materialize(str(seed))
        graph = json.loads(bundle.spec.graph_json)
        coverage.add((bundle.spec.inputs[0].dtype, graph["nodes"][1]["op"]))
        result = execute(bundle, TorchAdapter(ExecutionConfig("cpu-eager", "cpu")), EvaluationConfig())
        assert result["status"] == "ok", result
    assert len(coverage) == 6


def test_output_count_tree_shape_dtype_are_checked():
    torch = pytest.importorskip("torch")
    bundle = materialize("42")
    adapter = TorchAdapter(ExecutionConfig("cpu-eager", "cpu"))
    adapter.check(bundle, EvaluationConfig())
    expected = reference(bundle, 0)
    actual = adapter.invoke(adapter.prepare(bundle), adapter.inputs(bundle, 0))
    assert adapter.compare(actual, expected)
    assert not adapter.compare(actual + actual, expected)
    assert not adapter.compare(list(actual), expected)
    assert not adapter.compare((actual[0].to(torch.float64),), expected)
    assert not adapter.compare((actual[0].reshape(-1, 1),), expected)


def test_frozen_protocol_vectors():
    import hashlib
    path = Path(__file__).parent / "fixtures/fuzz/numeric_v1_vectors.json"
    for vector in json.loads(path.read_text()):
        bundle = materialize(vector["seed"])
        assert workload_id(bundle) == vector["workload_id"]
        assert [hashlib.sha256(v).hexdigest() for v in bundle.storages] == vector["storage_sha256"]
        assert [hashlib.sha256(reference(bundle, i)[0][1]).hexdigest()
                for i in range(len(bundle.spec.shape_samples))] == vector["oracle_sha256"]


def test_all_frozen_profile_choices_are_reachable():
    choices = {key: set() for key in ("dtype", "broadcast", "layout", "operation", "reduce")}
    for seed in range(128):
        spec = plan(str(seed))
        graph = json.loads(spec.graph_json)
        choices["dtype"].add(spec.inputs[0].dtype)
        choices["broadcast"].add(graph["nodes"][0]["weight_broadcast_axes"][0])
        choices["layout"].add(spec.inputs[0].stride[0] == 1)
        choices["operation"].add(graph["nodes"][1]["op"])
        choices["reduce"].add(graph["reduce_axis"])
    assert {k: len(v) for k, v in choices.items()} == {"dtype": 2, "broadcast": 2, "layout": 2, "operation": 3, "reduce": 3}


def test_adapter_rejects_graph_outside_its_declared_profile():
    from autofusedebugger.fuzz.contracts import canonical_json
    bundle = materialize("0")
    graph = json.loads(bundle.spec.graph_json)
    graph["nodes"][1]["op"] = "unknown"
    tampered = CaseBundle(replace(bundle.spec, graph_json=canonical_json(graph)), bundle.storages)
    result = execute(tampered, TorchAdapter(ExecutionConfig("cpu-eager", "cpu")), EvaluationConfig())
    assert result["status"] == "unsupported"
    assert "arbitrary graphs" in result["error_summary"]


def test_cpu_backend_cannot_silently_run_on_another_device():
    bundle = materialize("0")
    result = execute(bundle, TorchAdapter(ExecutionConfig("cpu-eager", "cuda:0")), EvaluationConfig())
    assert result["status"] == "unsupported"
    assert "device mismatch" in result["error_summary"]


def test_numeric_version_encoding_and_rejection():
    from autofusedebugger.fuzz.contracts import Seed
    assert Seed.parse(42) == Seed.parse("42")
    assert Seed.parse("42").version == 1 and Seed.parse("42").entropy == 42
    assert Seed.parse(str(2**128-1)).version == 1
    for value in (True, -1, "-1", "+42", " 42", "042", "٤٢", 2**128, 2**129):
        with pytest.raises(ValueError):
            Seed.parse(value)


def test_static_selection_preserves_entire_case_and_base_input():
    pytest.importorskip("torch")
    bundle = materialize("42")
    original = (bundle.spec, bundle.storages, workload_id(bundle))
    full_adapter = TorchAdapter(ExecutionConfig("cpu-eager", "cpu"))
    base_adapter = TorchAdapter(ExecutionConfig("cpu-eager", "cpu", static_only=True))
    full = execute(bundle, full_adapter, EvaluationConfig())
    base = execute(bundle, base_adapter, EvaluationConfig(performance=True, warmup_steps=0, measured_steps=1))
    assert full["status"] == base["status"] == "ok"
    assert full["workload_id"] == base["workload_id"]
    assert len(full["shape_results"]) == 3 and len(base["shape_results"]) == 1
    assert (bundle.spec, bundle.storages, workload_id(bundle)) == original
    assert full_adapter.torch.equal(full_adapter.inputs(bundle, 0)[0], base_adapter.inputs(bundle, 0)[0])


def test_replay_requires_only_one_seed_and_rejects_generation_overrides(capsys):
    from autofusedebugger.fuzz.cli import main
    pytest.importorskip("torch")
    assert main(["replay", "42"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["seed"] == "42" and len(result["shape_results"]) == 3
    for flag in ("--dynamic", "--max-nodes", "--dtype", "--seed-token"):
        with pytest.raises(SystemExit):
            main(["replay", "42", flag, "1"])
    assert main(["replay", "v1:preview-dynamic-v1:42"]) == 2


def test_numeric_seed_cli_is_stdlib_only(tmp_path):
    code = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'torch','torch_npu','numpy','nnsmith'}: raise AssertionError(name)
sys.meta_path.insert(0,Block())
from autofusedebugger.fuzz.cli import main
assert main(['seed','42','--show-spec'])==0
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["shape_count"] == 3
    assert not list(tmp_path.iterdir())
