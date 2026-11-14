from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autofusex.npu_examples import (
    _GeGraphSnapshot,
    _ge_output_summaries,
    _load_pre_run_ge_models,
    _match_broadcast_add_reducesum_dumps,
    _fx_tensor_origin,
    _select_runtime_ge_graph,
    example_from_script,
    find_msaccucmp,
    format_run_summary,
    op_name_from_npy,
    runtime_ge_graph_data,
)


def test_example_from_script_supports_known_examples() -> None:
    assert example_from_script("/x/af_add_ge.py").key == "af_add_ge"
    assert example_from_script(Path("af_mul_reducesum.py")).output_origin == "ReduceSum_2"
    spec = example_from_script(Path("af_broadcast_add_reducesum.py"))
    assert spec.key == "af_broadcast_add_reducesum"
    assert spec.input_shapes == ((32, 64), (64,))
    assert example_from_script(Path("af_qwen.py")).output_origin == "ModelOutput"


def test_example_from_script_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported example"):
        example_from_script("unknown.py")


def test_op_name_from_npy_strips_cann_task_suffix() -> None:
    path = "AutomaticBufferFusionOp.AddGreaterEqual.0.44.1784034891410240.output.0.npy"
    assert op_name_from_npy(path) == "AutomaticBufferFusionOp.AddGreaterEqual"


def test_match_broadcast_add_reducesum_dumps() -> None:
    matches = _match_broadcast_add_reducesum_dumps(
        [
            Path("Add.Add.0.43.1.output.0.npy"),
            Path("ReduceSumD.ReduceSum.1.43.2.output.0.npy"),
        ]
    )
    assert matches == {"Add.Add": "Add_1", "ReduceSumD.ReduceSum": "ReduceSum_2"}


def test_find_msaccucmp_uses_cann_root(tmp_path: Path) -> None:
    script = tmp_path / "tools/operator_cmp/compare/msaccucmp.py"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    assert find_msaccucmp(tmp_path) == script


def test_runtime_ge_graph_uses_real_edges_and_fx_origin(tmp_path: Path) -> None:
    def attr(*, text: str | bytes = "", strings: tuple[str | bytes, ...] = ()) -> SimpleNamespace:
        return SimpleNamespace(s=text, list=SimpleNamespace(s=list(strings)))

    data = SimpleNamespace(name="arg0_1", type="Data", input=[], output_desc=[], attr={})
    gather_desc = SimpleNamespace(
        layout="NCHW",
        attr={
            "_datadump_origin_name": attr(text=b"GatherV2"),
            "_datadump_origin_output_index": SimpleNamespace(i=0),
        },
    )
    gather = SimpleNamespace(
        name="GatherV2",
        type="GatherV2",
        input=["arg0_1:0"],
        output_desc=[gather_desc],
        attr={"_datadump_original_op_names": attr(strings=(b"GatherV2",))},
    )
    output = SimpleNamespace(
        name="NetOutput",
        type="NetOutput",
        input=["GatherV2:0"],
        output_desc=[],
        attr={},
    )
    model = SimpleNamespace(graph=[SimpleNamespace(op=[data, gather, output])])
    dump = tmp_path / "GatherV2.GatherV2.4.44.1.output.0.npy"
    np.save(dump, np.zeros((2, 16, 128), dtype=np.float32))

    graph = runtime_ge_graph_data(
        model,
        [dump],
        {"embedding"},
        "/tmp/ge_proto_00000094_graph_0_PreRunAfterBuild.txt",
        {("GatherV2", 0): ("embedding", 0)},
    )
    ops = graph["graph"][0]["op"]

    assert [op["name"] for op in ops] == ["Data.arg0_1", "GatherV2.GatherV2", "NetOutput.NetOutput"]
    assert ops[1]["input"] == ["Data.arg0_1:0"]
    assert ops[2]["input"] == ["GatherV2.GatherV2:0"]
    assert ops[1]["origin_ops"] == ["GatherV2"]
    assert ops[1]["output_desc"][0]["layout"] == "ND"
    assert ops[1]["output_desc"][0]["attr"] == [
        {"key": "_autofusex_cpu_origin_name", "value": {"s": "embedding"}},
        {"key": "_autofusex_cpu_origin_output_index", "value": {"i": 0}},
    ]
    assert graph["metadata"]["topology"] == "CANN GE PreRunAfterBuild"


def test_fx_tensor_origin_decodes_protobuf_bytes() -> None:
    attr = SimpleNamespace(s=b"embedding-aten.embedding.default.OUTPUT.0")
    desc = SimpleNamespace(attr={"_fx_tensor_name": attr})
    assert _fx_tensor_origin(desc, {"embedding"}) == ("embedding", 0)


def test_fx_tensor_origin_resolves_mm_to_exported_matmul() -> None:
    attr = SimpleNamespace(s=b"mm-aten.mm.default.OUTPUT.0")
    desc = SimpleNamespace(attr={"_fx_tensor_name": attr})
    assert _fx_tensor_origin(desc, {"matmul"}) == ("matmul", 0)


def test_runtime_ge_graph_matches_normalized_dump_name(tmp_path: Path) -> None:
    desc = SimpleNamespace(layout="NCHW", attr={})
    op = SimpleNamespace(name="Add/Mul", type="AutomaticBufferFusionOp", input=[], output_desc=[desc], attr={})
    model = SimpleNamespace(graph=[SimpleNamespace(op=[op])])
    dump = tmp_path / "AutomaticBufferFusionOp.Add_Mul.0.1.2.output.0.npy"
    np.save(dump, np.ones((2, 4), dtype=np.float32))

    graph = runtime_ge_graph_data(model, [dump], set(), "/tmp/ge_proto_00000012_graph_0_Build.txt")
    output = graph["graph"][0]["op"][0]["output_desc"][0]

    assert output["data_type"] == "float32"
    assert output["shape"] == {"dim": [2, 4]}
    assert output["layout"] == "ND"


def test_runtime_ge_graph_includes_prefusion_subgraph(tmp_path: Path) -> None:
    def attr(*strings: bytes) -> SimpleNamespace:
        return SimpleNamespace(s=b"", list=SimpleNamespace(s=list(strings)))

    tensor_desc = SimpleNamespace(
        dtype=0,
        shape=SimpleNamespace(dim=[2, 4]),
        layout="ND",
        attr={},
    )
    add = SimpleNamespace(name="Add", type="Add", input=[], output_desc=[tensor_desc], attr={})
    cast = SimpleNamespace(name="Cast", type="Cast", input=["Add:0"], output_desc=[tensor_desc], attr={})
    relu = SimpleNamespace(name="Relu", type="Relu", input=["Cast:0"], output_desc=[tensor_desc], attr={})
    pre_run = SimpleNamespace(graph=[SimpleNamespace(op=[add, cast, relu])])
    fused = SimpleNamespace(
        name="AddRelu",
        type="AutomaticBufferFusionOp",
        input=[],
        output_desc=[tensor_desc],
        attr={"_datadump_original_op_names": attr(b"Add", b"Relu")},
    )
    build = SimpleNamespace(graph=[SimpleNamespace(op=[fused])])
    dump = tmp_path / "AutomaticBufferFusionOp.AddRelu.0.1.2.output.0.npy"
    np.save(dump, np.ones((2, 4), dtype=np.float32))

    graph = runtime_ge_graph_data(
        build,
        [dump],
        set(),
        "/tmp/ge_proto_00000012_graph_0_Build.txt",
        origin_models=[pre_run],
    )

    subgraph = graph["graph"][0]["op"][0]["origin_subgraph"]
    assert subgraph["source"] == "CANN GE PreRunBegin"
    assert [(node["name"], node["type"], node["origin"]) for node in subgraph["nodes"]] == [
        ("Add", "Add", True),
        ("Cast", "Cast", False),
        ("Relu", "Relu", True),
    ]
    assert subgraph["edges"] == [
        {"src": "Add", "dst": "Cast", "dst_input": 0},
        {"src": "Cast", "dst": "Relu", "dst_input": 0},
    ]
    assert subgraph["missing"] == []


def test_prefusion_graph_prefers_optimized_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    early = tmp_path / "ge_proto_00000000_graph_0_PreRunBegin.txt"
    optimized = tmp_path / "ge_proto_00000009_graph_0_RunCustomPassAfterOriginGraphOptimize.txt"
    early.touch()
    optimized.touch()
    monkeypatch.setattr(
        "autofusex.npu_examples._load_runtime_ge_model",
        lambda path: path.name,
    )

    snapshots = _load_pre_run_ge_models(tmp_path)

    assert snapshots == [
        _GeGraphSnapshot(
            source="CANN GE RunCustomPassAfterOriginGraphOptimize",
            model=optimized.name,
        )
    ]


def test_ge_output_shape_falls_back_to_fx_meta() -> None:
    desc = SimpleNamespace(
        dtype=0,
        shape=SimpleNamespace(dim=[]),
        layout="ND",
        attr={"_meta": SimpleNamespace(s=b"Tensor(dtype=torch.float32, shape=torch.Size([8, 4])")},
    )
    op = SimpleNamespace(output_desc=[desc])

    assert _ge_output_summaries(op)[0]["shape"] == [8, 4]


def test_optimized_prefusion_subgraph_does_not_show_eliminated_cast(tmp_path: Path) -> None:
    tensor_desc = SimpleNamespace(dtype=0, shape=SimpleNamespace(dim=[8, 4]), layout="ND", attr={})
    sigmoid = SimpleNamespace(
        name="Sigmoid", type="Sigmoid", input=["MatMul:0"], input_desc=[tensor_desc], output_desc=[tensor_desc], attr={}
    )
    mul = SimpleNamespace(
        name="Mul", type="Mul", input=["Sigmoid:0"], input_desc=[tensor_desc], output_desc=[tensor_desc], attr={}
    )
    tanh = SimpleNamespace(
        name="Tanh", type="Tanh", input=["Mul:0"], input_desc=[tensor_desc], output_desc=[tensor_desc], attr={}
    )
    add = SimpleNamespace(
        name="Add_1",
        type="Add",
        input=["Mul:0", "Tanh:0"],
        input_desc=[tensor_desc, tensor_desc],
        output_desc=[tensor_desc],
        attr={},
    )
    prefusion = SimpleNamespace(graph=[SimpleNamespace(op=[sigmoid, mul, tanh, add])])
    fused = SimpleNamespace(
        name="SigmoidMulTanhAdd_1",
        type="AutomaticBufferFusionOp",
        input=[],
        output_desc=[tensor_desc],
        attr={"_datadump_original_op_names": SimpleNamespace(
            s=b"",
            list=SimpleNamespace(s=[b"Sigmoid", b"Mul", b"Tanh", b"Add_1"]),
        )},
    )
    dump = tmp_path / "AutomaticBufferFusionOp.SigmoidMulTanhAdd_1.0.1.2.output.0.npy"
    np.save(dump, np.ones((8, 4), dtype=np.float32))

    graph = runtime_ge_graph_data(
        SimpleNamespace(graph=[SimpleNamespace(op=[fused])]),
        [dump],
        set(),
        "/tmp/ge_proto_00000012_graph_0_Build.txt",
        origin_models=[_GeGraphSnapshot("CANN GE RunCustomPassAfterOriginGraphOptimize", prefusion)],
    )

    subgraph = graph["graph"][0]["op"][0]["origin_subgraph"]
    assert subgraph["source"] == "CANN GE RunCustomPassAfterOriginGraphOptimize"
    assert [node["name"] for node in subgraph["nodes"]] == ["Sigmoid", "Mul", "Tanh", "Add_1"]
    assert all(node["outputs"][0]["shape"] == [8, 4] for node in subgraph["nodes"])
    assert subgraph["nodes"][0]["inputs"] == [
        {"index": 0, "dtype": "0", "shape": [8, 4], "format": "ND", "src": "MatMul"}
    ]


def test_runtime_ge_graph_is_required(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="refusing to construct topology from dump file order"):
        _select_runtime_ge_graph(tmp_path, [])


def test_run_summary_preserves_incomplete_status() -> None:
    text = format_run_summary(
        {
            "example": "af_qwen",
            "passed": False,
            "html": "/tmp/report.html",
            "out_dir": "/tmp/out",
            "fused_op": "qwen",
            "summary": {
                "status": "INCOMPLETE",
                "passed": 45,
                "compared": 45,
                "total": 83,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
            },
        }
    )
    assert "INCOMPLETE passed=45/45 coverage=45/83" in text
