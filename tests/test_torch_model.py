from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import autofusedebugger.torch_model as torch_model
from autofusedebugger.cli import build_parser
from autofusedebugger.artifact_io import tensor_to_numpy

from autofusedebugger.torch_model import (
    TorchCase,
    _execution_context,
    _model_device_hints,
    _stage_case,
    _to_device,
    cpu_candidates_for_npu,
    export_cpu_graph_and_dump,
    load_torch_case,
    match_npu_to_cpu,
)


def test_to_device_converts_numpy_in_nested_pytree() -> None:
    from collections import namedtuple

    import numpy as np

    torch = pytest.importorskip("torch")
    Batch = namedtuple("Batch", ("tokens",))
    value = {"batch": Batch(np.array([[1, 2]], dtype=np.int64))}

    converted = _to_device(value, "cpu")

    assert isinstance(converted["batch"], Batch)
    assert isinstance(converted["batch"].tokens, torch.Tensor)
    assert converted["batch"].tokens.device.type == "cpu"
    assert converted["batch"].tokens.tolist() == [[1, 2]]


def test_model_device_hints_are_aligned_and_restored() -> None:
    torch = pytest.importorskip("torch")

    class Child(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("meta")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dev = "npu:0"
            self.child = Child()

    model = Model()

    with _model_device_hints(model, "cpu", enabled=True):
        assert model.dev == "cpu"
        assert model.child.device == torch.device("cpu")

    assert model.dev == "npu:0"
    assert model.child.device == torch.device("meta")


def test_model_device_hints_are_preserved_by_default() -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = "cpu"

    model = Model()

    with _model_device_hints(model, "npu:0"):
        assert model.device == "cpu"


def test_stage_case_uses_independent_model_and_inputs() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2).train()
    inputs = (torch.ones(1, 2, requires_grad=True),)
    case = TorchCase(
        model=model,
        inputs=inputs,
        name="isolation",
        training=True,
        grad_enabled=False,
    )

    stage = _stage_case(case, "test")
    with torch.no_grad():
        stage.model.weight.zero_()
        stage.inputs[0].zero_()

    assert stage.model is not model
    assert stage.model.training is True
    assert stage.inputs[0] is not inputs[0]
    assert inputs[0].tolist() == [[1.0, 1.0]]
    assert not torch.equal(stage.model.weight, model.weight)
    with _execution_context(stage, "cpu"):
        assert torch.is_grad_enabled() is False


def test_tensor_to_numpy_uses_float32_storage_for_bfloat16() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    array, dtype = tensor_to_numpy(tensor)

    assert str(array.dtype) == "float32"
    assert dtype == "bfloat16"


def test_load_torch_case_from_tuple(tmp_path: Path) -> None:
    script = tmp_path / "case.py"
    script.write_text(
        """
class Model:
    pass

def create_case():
    return Model(), (1, 2)
""",
        encoding="utf-8",
    )

    case = load_torch_case(script)

    assert case.name == "case"
    assert case.inputs == (1, 2)
    assert case.fullgraph is False


def test_load_torch_case_from_dict(tmp_path: Path) -> None:
    script = tmp_path / "case.py"
    script.write_text(
        """
class Model:
    pass

def make():
    return {"name": "custom", "model": Model(), "inputs": [1], "fullgraph": False}
""",
        encoding="utf-8",
    )

    case = load_torch_case(script, "make")

    assert case.name == "custom"
    assert case.inputs == (1,)
    assert case.fullgraph is False


def test_cpu_candidates_for_npu_common_ops() -> None:
    assert cpu_candidates_for_npu("Add.Add")[0] == "add"
    assert "relu" in cpu_candidates_for_npu("Relu.Relu")
    assert "matmul" in cpu_candidates_for_npu("MatMul.MatMul")
    assert "sum" in cpu_candidates_for_npu("ReduceSumD.ReduceSum")
    assert cpu_candidates_for_npu("AutomaticBufferFusionOp.AddReluMul_1")[0] == "mul_1"


def test_match_npu_to_cpu_uses_shape_size(tmp_path: Path) -> None:
    import numpy as np

    npu = tmp_path / "Add.Add.0.1.2.output.0.npy"
    np.save(npu, np.ones((2, 2), dtype=np.float32))

    matches = match_npu_to_cpu([npu], {"add": {"size": 4}, "add_1": {"size": 8}})

    assert matches == {"Add.Add": "add"}


def test_match_npu_to_cpu_chooses_lowest_error_candidate(tmp_path: Path) -> None:
    import numpy as np

    npu = tmp_path / "Sigmoid.SigmoidMulTanhAdd_1.0.1.2.output.0.npy"
    sigmoid = tmp_path / "sigmoid.npy"
    add_1 = tmp_path / "add_1.npy"
    np.save(npu, np.array([2.0, 3.0], dtype=np.float32))
    np.save(sigmoid, np.array([0.5, 0.6], dtype=np.float32))
    np.save(add_1, np.array([2.0, 3.0], dtype=np.float32))

    matches = match_npu_to_cpu(
        [npu],
        {
            "sigmoid": {"size": 2, "path": sigmoid},
            "add_1": {"size": 2, "path": add_1},
        },
    )

    assert matches == {"Sigmoid.SigmoidMulTanhAdd_1": "add_1"}


def test_cli_has_run_torch_model() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-torch-model", "--script", "case.py", "--out-dir", "out"])
    assert args.entry == "create_case"
    assert args.device == "npu:0"
    assert args.rtol is None
    assert args.atol is None
    assert args.fullgraph is None

    explicit = parser.parse_args(
        ["run-torch-model", "--script", "case.py", "--out-dir", "out", "--fullgraph"]
    )
    assert explicit.fullgraph is True


def test_cli_has_run_script_with_script_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-script",
            "--script",
            "model.py",
            "--out-dir",
            "out",
            "--framework",
            "pytorch",
            "--",
            "--batch-size",
            "2",
        ]
    )
    assert args.backend == "inductor_autofuse"
    assert args.script_args == ["--", "--batch-size", "2"]
    assert args.fullgraph is False
    assert args.rewrite_device_hints is False

    explicit = parser.parse_args(
        [
            "run-script",
            "--script",
            "model.py",
            "--out-dir",
            "out",
            "--fullgraph",
            "--rewrite-device-hints",
        ]
    )
    assert explicit.fullgraph is True
    assert explicit.rewrite_device_hints is True


def test_export_cpu_graph_supports_keyword_tensor_inputs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def forward(self, x, *, bias):
            return torch.relu(x + bias)

    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    case = TorchCase(
        model=Model().eval(),
        inputs=(torch.ones(2, 4),),
        kwargs={"bias": torch.ones(4)},
        name="kwargs",
    )

    graph, outputs = export_cpu_graph_and_dump(case, dump_dir)

    assert [op["type"] for op in graph["graph"][0]["op"]][-2:] == ["add", "relu"]
    assert outputs["relu"]["shape"] == (2, 4)


def test_export_cpu_graph_includes_parameter_edges_for_convolution(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, bias=True)

        def forward(self, x):
            return self.conv(x)

    graph, _ = export_cpu_graph_and_dump(
        TorchCase(model=Model().eval(), inputs=(torch.ones(1, 3, 8, 8),), name="convolution"), tmp_path
    )
    ops = graph["graph"][0]["op"]
    convolution = next(op for op in ops if op["type"] == "convolution")

    assert len(convolution["input"]) == 3
    assert all(next(op for op in ops if op["name"] == value.split(":", 1)[0])["type"] == "Data" for value in convolution["input"])


def test_export_cpu_graph_runs_lifted_constant_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Model(torch.nn.Module):
        def forward(self, x):
            return x * torch.tensor(2.0)

    graph, outputs = export_cpu_graph_and_dump(
        TorchCase(model=Model().eval(), inputs=(torch.ones(4),), name="lifted-constant"), tmp_path
    )

    assert any(op["type"] == "mul" for op in graph["graph"][0]["op"])
    assert any(np.allclose(np.load(info["path"]), 2.0) for info in outputs.values())


def test_cpu_export_retries_device_hints_for_fake_tensor_device_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = TorchCase(model=object(), inputs=(), name="mixed-device")
    expected = ({"graph": [{"op": []}]}, {})
    calls: list[tuple[str, bool]] = []

    def export(case_arg, dump_dir, *, device):  # type: ignore[no-untyped-def]
        del dump_dir
        calls.append((device, case_arg.rewrite_device_hints))
        if not case_arg.rewrite_device_hints:
            raise RuntimeError(
                "Unhandled FakeTensor Device Propagation for aten.mul_.Tensor, found two different devices npu:0, cpu"
            )
        return expected

    monkeypatch.setattr(torch_model, "export_graph_and_dump", export)

    assert torch_model.export_cpu_graph_and_dump(case, tmp_path) == expected
    assert calls == [("cpu", False), ("cpu", True)]


def test_export_cpu_graph_uses_inductor_level_decomposition(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not hasattr(torch.export.ExportedProgram, "run_decompositions"):
        pytest.skip("requires torch.export decomposition support")

    model = torch.nn.Linear(4, 5, bias=False).eval()
    case = TorchCase(model=model, inputs=(torch.ones(2, 3, 4),), name="linear")

    graph, outputs = export_cpu_graph_and_dump(case, tmp_path)
    types = [op["type"] for op in graph["graph"][0]["op"]]

    assert "linear" not in types
    assert "mm" in types
    assert any(info["shape"] == (6, 5) for name, info in outputs.items() if name.startswith("mm"))
