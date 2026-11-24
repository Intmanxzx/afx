from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

import autofusex.workloads.model.inductor_autofuse as inductor_autofuse
from autofusex.execution.graph import extract_edges
from autofusex.workloads.model.inductor_autofuse import (
    _match_cpu_output,
    _fallback_capture_name,
    _trace_sort_key,
    activate_autofuse_backend,
    build_inductor_autofuse_graph,
)


def test_capture_name_selects_the_runtime_graph_invocation() -> None:
    records = {
        "transpose": {"runtime_gm_id": 7, "runtime_call_index": 0, "outputs": []},
        "transpose__call1": {"runtime_gm_id": 7, "runtime_call_index": 1, "outputs": []},
        "transpose__call2": {"runtime_gm_id": 8, "runtime_call_index": 0, "outputs": []},
    }

    selected = _fallback_capture_name(
        "transpose.default",
        records,
        set(),
        {"gm_id": 7, "call_index": 1},
    )

    assert selected == "transpose__call1"


def test_selected_runtime_capture_keeps_prefusion_canonical_mapping(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0: ExternKernelSchedulerNode(FallbackKernel)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = [StarDep(name='arg0_1', mode=None)]
op0.met_dependencies = []
op0.outputs = [
    buf0: FallbackKernel
    buf0.layout = FixedLayout('npu:0', torch.bfloat16, size=[2, 3], stride=[3, 1])
]
op0.node.kernel = torch.ops.aten.transpose.int
arg0_1_layout = FixedLayout('npu:0', torch.bfloat16, size=[2, 3], stride=[3, 1])
""",
        encoding="utf-8",
    )
    (trace / "output_code.py").write_text("run_intermediate_hooks('transpose', buf0)\n", encoding="utf-8")
    output_path = tmp_path / "transpose.npy"
    np.save(output_path, np.ones((2, 3), dtype=np.float32))

    graph = build_inductor_autofuse_graph(
        tmp_path / "compile",
        {
            "transpose__call1": {
                "runtime_gm_id": 7,
                "runtime_call_index": 1,
                "outputs": [
                    {
                        "index": 0,
                        "path": output_path,
                        "dtype": "bfloat16",
                        "shape": (2, 3),
                        "size": 6,
                    }
                ],
            }
        },
        {},
        prefusion_graph_data={
            "graph": [
                {
                    "op": [
                        {
                            "name": "transpose",
                            "canonical_id": "graph0::transpose",
                            "output_desc": [
                                {"shape": {"dim": [2, 3]}, "data_type": "bfloat16"}
                            ],
                        }
                    ]
                }
            ]
        },
        prefusion_runtime_selection={0: {"gm_id": 7, "call_index": 1}},
    )
    op = next(op for op in graph["graph"][0]["op"] if op["type"] != "Data")

    assert op["name"] == "transpose__call1"
    assert op["origin_ops"] == ["transpose"]
    assert op["canonical_id"] == "graph0::transpose"


def test_activate_autofuse_backend_prefers_torch_npu_bundle(monkeypatch) -> None:
    calls = []
    monkeypatch.delenv("TORCHINDUCTOR_NPU_BACKEND", raising=False)

    def import_module(name):
        calls.append(name)
        return object()

    monkeypatch.setattr(inductor_autofuse.importlib, "import_module", import_module)

    assert activate_autofuse_backend() == "torch_npu._inductor.ascendc"
    assert calls == ["torch_npu._inductor.ascendc"]
    assert os.environ["TORCHINDUCTOR_NPU_BACKEND"] == "ascendc"


def test_activate_autofuse_backend_falls_back_to_standalone(monkeypatch) -> None:
    calls = []

    def import_module(name):
        calls.append(name)
        if name == "torch_npu._inductor.ascendc":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(inductor_autofuse.importlib, "import_module", import_module)

    assert activate_autofuse_backend() == "inductor_npu_ext"
    assert calls == ["torch_npu._inductor.ascendc", "inductor_npu_ext"]


def test_trace_sort_key_uses_numeric_inductor_graph_id(tmp_path: Path) -> None:
    traces = [
        tmp_path / "model__10_inference_10.10",
        tmp_path / "model__2_inference_2.2",
        tmp_path / "model__1_inference_1.1",
        tmp_path / "other_trace",
    ]

    assert [path.name for path in sorted(traces, key=_trace_sort_key)] == [
        "model__1_inference_1.1",
        "model__2_inference_2.2",
        "model__10_inference_10.10",
        "other_trace",
    ]


def test_build_inductor_autofuse_graph_uses_real_ir_and_fusion_exit(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 8})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 8})]
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('npu:0', torch.float32, size=[2, 4], stride=[4, 1])
]
arg0_1_layout = FixedLayout('npu:0', torch.float32, size=[2, 4], stride=[4, 1])
""",
        encoding="utf-8",
    )
    (trace / "fx_graph_readable.py").write_text(
        'def forward(self, arg0_1: "f32[2, 4]"):\n    return arg0_1\n', encoding="utf-8"
    )
    kernel = "autofused_add_relu_deadbeef"
    benchmark = trace / kernel / "benchmark.py"
    benchmark.parent.mkdir()
    benchmark.write_text(
        """
        # Graph fragment:
        #   %arg0_1 : Tensor "f32[2, 4][4, 1]npu:0" = PlaceHolder[target=arg0_1]
        #   %add : Tensor "f32[2, 4][4, 1]npu:0" = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1.0), kwargs = {})
        #   %relu : Tensor "f32[2, 4][4, 1]npu:0" = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
        #   return %relu
        op0: SchedulerNode(ComputedBuffer)
        graph_hint = ascir.HintGraph('graph_hint')
        data = ascir.ops.Data('graph_hint/data', graph_hint)
        data.attr.ir_attr.index = 0
        load = ascir.ops.Load('graph_hint/load', graph_hint)
        load.x = data.y
        load.y.size = [2, 4]
        add = ascir.ops.Add('graph_hint/add', graph_hint)
        add.x1 = load.y
        add.y.size = [2, 4]
        relu = ascir.ops.Relu('graph_hint/relu', graph_hint)
        relu.x = add.y
        relu.y.size = [2, 4]
        output = ascir.ops.Output('graph_hint/output', graph_hint)
        output.x = relu.y
        output.y.dtype = ascir.dtypes.float32
        graph_hint.infer_dtypes()
""",
        encoding="utf-8",
    )

    npu_path = tmp_path / "npu.npy"
    cpu_path = tmp_path / "cpu.npy"
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.save(npu_path, values)
    np.save(cpu_path, values)
    capture_name = f"{kernel}__relu"
    records = {
        capture_name: {
            "name": capture_name,
            "outputs": [
                {"index": 0, "path": npu_path, "dtype": "float32", "shape": (2, 4), "size": 8}
            ],
        }
    }
    second_trace = tmp_path / "compile" / "torchinductor" / "model__1"
    shutil.copytree(trace, second_trace)
    second_name = f"{capture_name}__call1"
    records[second_name] = {
        "name": second_name,
        "outputs": [{"index": 0, "path": npu_path, "dtype": "float32", "shape": (2, 4), "size": 8}],
    }
    cpu_outputs = {
        "relu": {"path": cpu_path, "dtype": "float32", "shape": (2, 4), "size": 8, "type": "relu"}
    }

    graph = build_inductor_autofuse_graph(tmp_path / "compile", records, cpu_outputs)
    ops = graph["graph"][0]["op"]
    fusion = next(op for op in ops if op["type"] == "AutofuseFusion")
    fusions = [op for op in ops if op["type"] == "AutofuseFusion"]

    assert graph["metadata"]["topology"] == "TorchInductor post-fusion / inductor_npu_ext Autofuse"
    assert fusion["name"] == capture_name
    assert fusion["input"] == ["Data.arg0_1:0"]
    assert fusion["origin_ops"] == ["add", "relu"]
    assert [node["type"] for node in fusion["origin_subgraph"]["nodes"]] == ["Data", "Load", "Add", "Relu", "Output"]
    assert fusion["origin_subgraph"]["nodes"][0]["outputs"][0]["shape"] == [2, 4]
    assert fusion["output_desc"][0]["attr"][0]["value"]["s"] == "relu"
    assert [op["name"] for op in fusions] == [capture_name, second_name]
    assert fusions[1]["input"] == ["graph1.Data.arg0_1:0"]


def test_build_inductor_graph_attaches_prefusion_canonical_ids(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op0.writes = [MemoryDep('buf0', c0, {c0: 8})]
op0.unmet_dependencies = [MemoryDep('arg0_1', c0, {c0: 8})]
    op0.met_dependencies = []
""",
        encoding="utf-8",
    )
    (trace / "ir_pre_fusion.txt").write_text(
        """
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 8})]
op0.unmet_dependencies = [MemoryDep('arg0_1', c0, {c0: 8})]
op0.met_dependencies = []
""",
        encoding="utf-8",
    )
    kernel = "autofused_add_relu_deadbeef"
    benchmark = trace / kernel / "benchmark.py"
    benchmark.parent.mkdir()
    benchmark.write_text(
        """
# Topologically Sorted Source Nodes: [add, relu], Original ATen: [aten.add, aten.relu]
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 4]" = PlaceHolder[target=arg0_1]
#   %add : Tensor "f32[2, 4]" = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1.0), kwargs = {})
#   %relu : Tensor "f32[2, 4]" = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
#   return %relu
op0: SchedulerNode(ComputedBuffer)
""",
        encoding="utf-8",
    )
    (trace / "output_code.py").write_text(
        f"run_intermediate_hooks('{kernel}__relu', buf0)\n", encoding="utf-8"
    )
    npu_path = tmp_path / "npu.npy"
    np.save(npu_path, np.ones((2, 4), dtype=np.float32))
    records = {
        f"{kernel}__relu__call1": {
            "outputs": [{"index": 0, "path": npu_path, "dtype": "float32", "shape": (2, 4), "size": 8}]
        }
    }
    prefusion = {
        "graph": [
            {
                "op": [
                    {"name": "add", "canonical_id": "graph0::add"},
                    {"name": "relu", "canonical_id": "graph0::relu"},
                ]
            }
        ]
    }

    graph = build_inductor_autofuse_graph(
        tmp_path / "compile",
        records,
        {},
        prefusion_graph_data=prefusion,
        cpu_graph_data=prefusion,
    )
    fusion = next(op for op in graph["graph"][0]["op"] if op["type"] == "AutofuseFusion")

    assert fusion["canonical_id"] == "graph0::relu"
    assert fusion["origin_node_ids"] == ["graph0::add", "graph0::relu"]
    assert fusion["mapping_method"] == "ir_pre_fusion_canonical_id"
    assert fusion["ir_pre_fusion"]["status"] == "MATCHED"
    assert fusion["ir_pre_fusion"]["scheduler_ids"] == ["op0"]


def test_build_inductor_graph_rejects_ir_canonical_shape_conflict(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 8})]
op0.unmet_dependencies = []
op0.met_dependencies = []
""",
        encoding="utf-8",
    )
    (trace / "ir_pre_fusion.txt").write_text(
        """
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 6})]
op0.unmet_dependencies = []
op0.met_dependencies = []
""",
        encoding="utf-8",
    )
    kernel = "autofused_relu_deadbeef"
    benchmark = trace / kernel / "benchmark.py"
    benchmark.parent.mkdir()
    benchmark.write_text(
        """
# Graph fragment:
#   %relu : Tensor "f32[2, 4]" = call_function[target=torch.ops.aten.relu.default](args = (%arg0_1,), kwargs = {})
#   return %relu
op0: SchedulerNode(ComputedBuffer)
""",
        encoding="utf-8",
    )
    (trace / "output_code.py").write_text(
        f"run_intermediate_hooks('{kernel}__relu', buf0)\n", encoding="utf-8"
    )
    npu_path = tmp_path / "npu.npy"
    np.save(npu_path, np.ones((2, 4), dtype=np.float32))
    graph = build_inductor_autofuse_graph(
        tmp_path / "compile",
        {
            f"{kernel}__relu": {
                "outputs": [
                    {"index": 0, "path": npu_path, "dtype": "float32", "shape": (2, 4), "size": 8}
                ]
            }
        },
        {},
        prefusion_graph_data={
            "graph": [
                {
                    "op": [
                        {
                            "name": "relu",
                            "canonical_id": "graph0::relu",
                            "output_desc": [{"shape": {"dim": [2, 3]}, "data_type": "float32"}],
                        }
                    ]
                }
            ]
        },
    )
    fusion = next(op for op in graph["graph"][0]["op"] if op["type"] == "AutofuseFusion")

    assert "canonical_id" not in fusion
    assert fusion["mapping_method"] == "provenance_conflict"
    assert fusion["provenance_conflicts"][0]["reference_shape"] == [2, 3]


def test_build_inductor_graph_uniquifies_repeated_uncaptured_nodes(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0: ExternKernelSchedulerNode(FallbackKernel)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None)]
op0.outputs = [
    buf0: FallbackKernel
    buf0.layout = FixedLayout('npu:0', torch.float32, size=[2, 4], stride=[4, 1])
]
op0.node.kernel = torch.ops.aten.reshape.default

op1: ExternKernelSchedulerNode(FallbackKernel)
op1.writes = [StarDep(name='buf1', mode=None)]
op1.unmet_dependencies = [StarDep(name='buf0', mode=None)]
op1.met_dependencies = []
op1.outputs = [
    buf1: FallbackKernel
    buf1.layout = FixedLayout('npu:0', torch.float32, size=[8], stride=[1])
]
op1.node.kernel = torch.ops.aten.reshape.default
""",
        encoding="utf-8",
    )

    graph = build_inductor_autofuse_graph(tmp_path / "compile", {}, {})
    computed = [op for op in graph["graph"][0]["op"] if op["type"] != "Data"]

    assert [op["name"] for op in computed] == ["reshape", "reshape__op1"]
    assert computed[1]["input"] == ["reshape:0"]


def test_build_inductor_graph_binds_hook_name_by_output_buffer(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0: ExternKernelSchedulerNode(FallbackKernel)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None)]
op0.node.kernel = torch.ops.aten.reshape.default
arg0_1_layout = FixedLayout('npu:0', torch.float32, size=[2, 4], stride=[4, 1])
buf0_layout = FixedLayout('npu:0', torch.float32, size=[8], stride=[1])
""",
        encoding="utf-8",
    )
    (trace / "output_code.py").write_text("run_intermediate_hooks('view', buf0)\n", encoding="utf-8")
    npu_path = tmp_path / "npu.npy"
    np.save(npu_path, np.arange(8, dtype=np.float32))
    records = {
        "view": {
            "name": "view",
            "outputs": [{"index": 0, "path": npu_path, "dtype": "float32", "shape": (8,), "size": 8}],
        }
    }

    diagnostics = {
        "lowering_fallbacks": [
            {
                "category": "implicit_fallback",
                "node": "logical_view",
                "target": "aten.reshape.default",
                "reason": "TorchInductor created an implicit fallback because no lowering was registered",
            }
        ]
    }
    graph = build_inductor_autofuse_graph(tmp_path / "compile", records, {}, diagnostics)
    computed = [op for op in graph["graph"][0]["op"] if op["type"] != "Data"]

    assert [op["name"] for op in computed] == ["view"]
    assert computed[0]["type"] == "aten.reshape"
    assert computed[0]["unfused_reasons"][0]["category"] == "implicit_fallback"
    assert computed[0]["unfused_reasons"][0]["source_node"] == "logical_view"


def test_build_inductor_graph_skips_fused_scheduler_components(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op0_op1.writes = [MemoryDep('buf1', c0, {c0: 8})]
op0_op1.unmet_dependencies = [MemoryDep('arg0_1', c0, {c0: 8})]
op0_op1.met_dependencies = []
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 8})]
op0.unmet_dependencies = [MemoryDep('arg0_1', c0, {c0: 8})]
op0.met_dependencies = []
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 8})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 8})]
op1.met_dependencies = []

op2: ExternKernelSchedulerNode(FallbackKernel)
op2.writes = [StarDep(name='buf2', mode=None)]
op2.unmet_dependencies = [StarDep(name='buf1', mode=None)]
op2.met_dependencies = [StarDep(name='arg1_1', mode=None), StarDep(name='arg2_1', mode=None)]
op2.node.kernel = torch.ops.aten.convolution.default
""",
        encoding="utf-8",
    )

    graph = build_inductor_autofuse_graph(tmp_path / "compile", {}, {})
    computed = [op for op in graph["graph"][0]["op"] if op["type"] != "Data"]

    assert [op["name"] for op in computed] == ["inductor_op0_op1", "convolution"]
    assert computed[1]["input"] == ["inductor_op0_op1:0", "Data.arg1_1:0", "Data.arg2_1:0"]
    assert not any(op["name"] == "Data.buf1" for op in graph["graph"][0]["op"])


def test_build_inductor_graph_attaches_node_level_unfused_reasons(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0: ExternKernelSchedulerNode(FallbackKernel)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None), StarDep(name='arg1_1', mode=None)]
op0.node.kernel = torch.ops.aten.mul.Tensor

op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 8})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 8})]
op1.met_dependencies = []
""",
        encoding="utf-8",
    )
    (trace / "fx_graph_readable.py").write_text(
        '''
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 4]", arg1_1: "f32[32, 16]"):
        logical_view: "f32[1, 1, 16, 16]" = torch.ops.aten.reshape.default(arg1_1, [1, 1, 16, 16])
        mul: "f32[2, 4, 16, 16]" = torch.ops.aten.mul.Tensor(arg0_1, logical_view)
        return mul
''',
        encoding="utf-8",
    )
    diagnostics = {
        "lowering_fallbacks": [
            {
                "node": "mul",
                "target": "aten.mul.Tensor",
                "reason": "performance of non-contiguous stride (16, 2) maybe worse than eager",
            }
        ],
        "fusion_decision_groups": [
            [
                {
                    "kind": "vertical",
                    "node1": ["op0"],
                    "node2": ["op1"],
                    "result": False,
                    "reason": "axis mapping unsupported",
                    "triton_can_fuse": True,
                }
            ]
        ],
    }

    graph = build_inductor_autofuse_graph(tmp_path / "compile", {}, {}, diagnostics)
    computed = [op for op in graph["graph"][0]["op"] if op["type"] != "Data"]
    mul_reasons = computed[0]["unfused_reasons"]

    assert [item["category"] for item in mul_reasons] == ["lowering_fallback", "fusion_rejection"]
    assert mul_reasons[0]["reason"].startswith("performance of non-contiguous stride")
    assert mul_reasons[1]["direction"] == "output"
    assert mul_reasons[1]["peer"] == computed[1]["name"]
    mul_edges = [edge for edge in extract_edges(graph["graph"][0]["op"]) if edge["dst"] == "mul"]
    assert mul_edges[1]["shape"] == [1, 1, 16, 16]
    assert graph["metadata"]["lowering_fallbacks"] == 1
    assert graph["metadata"]["fusion_decisions"] == 1


def test_match_cpu_output_prefers_exact_exit_and_handles_infinity(tmp_path: Path) -> None:
    npu_path = tmp_path / "npu.npy"
    wrong_path = tmp_path / "add_2.npy"
    expected_path = tmp_path / "add_3.npy"
    np.save(npu_path, np.array([1.0, -np.inf], dtype=np.float32))
    np.save(wrong_path, np.array([2.0, 0.0], dtype=np.float32))
    np.save(expected_path, np.array([1.0, -np.inf], dtype=np.float32))
    records = {
        "fusion": {
            "outputs": [{"index": 0, "path": npu_path, "shape": (2,), "size": 2}],
        }
    }
    cpu_outputs = {
        "add_2": {"path": wrong_path, "shape": (2,), "size": 2, "type": "add"},
        "add_3": {"path": expected_path, "shape": (2,), "size": 2, "type": "add"},
    }
    op = {"name": "fusion", "_exit_name": "add_3", "_exit_type": "add"}

    assert _match_cpu_output(op, records, cpu_outputs, set()) == "add_3"


def test_match_cpu_output_rejects_reordered_shape_with_same_element_count(tmp_path: Path) -> None:
    npu_path = tmp_path / "npu.npy"
    cpu_path = tmp_path / "clone_3.npy"
    np.save(npu_path, np.zeros((3, 2), dtype=np.float32))
    np.save(cpu_path, np.zeros((2, 3), dtype=np.float32))
    records = {
        "clone_3": {
            "outputs": [{"index": 0, "path": npu_path, "shape": (3, 2), "size": 6}],
        }
    }
    cpu_outputs = {
        "clone_3": {"path": cpu_path, "shape": (2, 3), "size": 6, "type": "clone"},
    }
    diagnostics = {}

    origin = _match_cpu_output(
        {"name": "clone_3", "_exit_name": "clone_3", "_exit_type": "clone"},
        records,
        cpu_outputs,
        set(),
        diagnostics,
    )

    assert origin is None
    assert diagnostics["candidate"] == "clone_3"
    assert diagnostics["reference_shape"] == [2, 3]
    assert diagnostics["npu_shape"] == [3, 2]


def test_match_cpu_output_does_not_suffix_match_non_fused_exact_name_conflict(tmp_path: Path) -> None:
    npu_path = tmp_path / "npu.npy"
    exact_path = tmp_path / "clone_6.npy"
    normalized_path = tmp_path / "clone.npy"
    np.save(npu_path, np.zeros((2, 3), dtype=np.float32))
    np.save(exact_path, np.zeros((3, 2), dtype=np.float32))
    np.save(normalized_path, np.zeros((2, 3), dtype=np.float32))
    records = {
        "clone_6": {
            "outputs": [{"index": 0, "path": npu_path, "shape": (2, 3), "size": 6}],
        }
    }
    cpu_outputs = {
        "clone_6": {"path": exact_path, "shape": (3, 2), "size": 6, "type": "clone"},
        "clone": {"path": normalized_path, "shape": (2, 3), "size": 6, "type": "clone"},
    }
    diagnostics = {}

    origin = _match_cpu_output(
        {"name": "clone_6", "_exit_name": "clone_6", "_exit_type": "clone_6"},
        records,
        cpu_outputs,
        set(),
        diagnostics,
    )

    assert origin is None
    assert diagnostics["candidate"] == "clone_6"


def test_match_cpu_output_prefers_candidate_with_matching_upstream_origin(tmp_path: Path) -> None:
    npu_path = tmp_path / "npu.npy"
    first_path = tmp_path / "native_layer_norm_1.npy"
    expected_path = tmp_path / "native_layer_norm_5.npy"
    value = np.zeros((2, 3), dtype=np.float32)
    np.save(npu_path, value)
    np.save(first_path, value)
    np.save(expected_path, value)
    records = {
        "fusion__add_14": {
            "outputs": [{"index": 0, "path": npu_path, "shape": (2, 3), "size": 6}],
        }
    }
    cpu_outputs = {
        "add_14": {"path": first_path, "shape": (2, 3), "size": 6, "type": "add"},
        "native_layer_norm_1": {"path": first_path, "shape": (2, 3), "size": 6, "type": "native_layer_norm"},
        "native_layer_norm_5": {"path": expected_path, "shape": (2, 3), "size": 6, "type": "native_layer_norm"},
    }

    origin = _match_cpu_output(
        {
            "name": "fusion__add_14",
            "_exit_name": "add_14",
            "_exit_type": "add_14",
            "origin_ops": ["sub_2", "add_7"],
            "original_aten_types": ["native_layer_norm"],
        },
        records,
        cpu_outputs,
        set(),
        cpu_inputs={"add_14": {"mul_4"}, "native_layer_norm_1": {"add_4"}, "native_layer_norm_5": {"add_12"}},
        input_origins={"add_12"},
    )

    assert origin == "native_layer_norm_5"


def test_match_cpu_output_prefers_transitive_topology_over_exact_non_fused_name(tmp_path: Path) -> None:
    npu_path = tmp_path / "npu.npy"
    exact_path = tmp_path / "clone_13.npy"
    expected_path = tmp_path / "clone_9.npy"
    value = np.zeros((3, 4), dtype=np.float32)
    np.save(npu_path, value)
    np.save(exact_path, value)
    np.save(expected_path, value)
    records = {
        "clone_13": {
            "outputs": [{"index": 0, "path": npu_path, "shape": (3, 4), "size": 12}],
        }
    }
    cpu_outputs = {
        "clone_9": {"path": expected_path, "shape": (3, 4), "size": 12, "type": "clone"},
        "clone_13": {"path": exact_path, "shape": (3, 4), "size": 12, "type": "clone"},
    }

    origin = _match_cpu_output(
        {"name": "clone_13", "_exit_name": "clone_13", "_exit_type": "clone"},
        records,
        cpu_outputs,
        set(),
        cpu_inputs={
            "clone_9": {"squeeze_2"},
            "squeeze_2": {"permute_26"},
            "clone_13": {"squeeze_3"},
            "squeeze_3": {"permute_38"},
        },
        input_origins={"permute_26"},
    )

    assert origin == "clone_9"


def test_match_cpu_output_rejects_compiler_inserted_non_fused_node(tmp_path: Path) -> None:
    npu_path = tmp_path / "clone_66.npy"
    cpu_path = tmp_path / "clone.npy"
    value = np.zeros((3, 4), dtype=np.float32)
    np.save(npu_path, value)
    np.save(cpu_path, value)
    records = {
        "clone_66": {
            "outputs": [{"index": 0, "path": npu_path, "shape": (3, 4), "size": 12}],
        }
    }
    cpu_outputs = {
        "clone": {"path": cpu_path, "shape": (3, 4), "size": 12, "type": "clone"},
    }
    diagnostics = {}

    origin = _match_cpu_output(
        {"name": "clone_66", "_exit_name": "clone_66", "_exit_type": "clone"},
        records,
        cpu_outputs,
        set(),
        diagnostics,
        cpu_inputs={"clone": {"add"}},
        input_origins={"add_44"},
    )

    assert origin is None
    assert "编译器插入" in diagnostics["reason"]


def test_match_cpu_output_prefers_exact_fusion_exit_over_nearer_internal_node(tmp_path: Path) -> None:
    npu_path = tmp_path / "fusion.npy"
    squeeze_path = tmp_path / "squeeze_16.npy"
    sigmoid_path = tmp_path / "sigmoid.npy"
    value = np.zeros((3, 4), dtype=np.float32)
    np.save(npu_path, value)
    np.save(squeeze_path, value)
    np.save(sigmoid_path, value)
    records = {
        "fusion__sigmoid": {
            "outputs": [{"index": 0, "path": npu_path, "shape": (3, 4), "size": 12}],
        }
    }
    cpu_outputs = {
        "squeeze_16": {"path": squeeze_path, "shape": (3, 4), "size": 12, "type": "squeeze"},
        "sigmoid": {"path": sigmoid_path, "shape": (3, 4), "size": 12, "type": "sigmoid"},
    }

    origin = _match_cpu_output(
        {
            "name": "fusion__sigmoid",
            "_exit_name": "sigmoid",
            "_exit_type": "sigmoid",
            "origin_ops": ["squeeze_16", "sigmoid"],
            "original_aten_types": ["squeeze", "sigmoid"],
        },
        records,
        cpu_outputs,
        set(),
        cpu_inputs={"squeeze_16": {"view_162"}, "view_162": {"bmm_48"}, "sigmoid": {"squeeze_16"}},
        input_origins={"bmm_48"},
    )

    assert origin == "sigmoid"


def test_match_cpu_output_accepts_cann_scalar_shape_normalization(tmp_path: Path) -> None:
    npu_path = tmp_path / "npu.npy"
    cpu_path = tmp_path / "cpu.npy"
    np.save(npu_path, np.array([1.0], dtype=np.float32))
    np.save(cpu_path, np.array(1.0, dtype=np.float32))
    records = {
        "scalar_tensor": {
            "outputs": [{"index": 0, "path": npu_path, "shape": (1,), "size": 1}],
        }
    }
    cpu_outputs = {
        "scalar_tensor": {"path": cpu_path, "shape": (), "size": 1, "type": "scalar_tensor"},
    }

    assert _match_cpu_output(
        {"name": "scalar_tensor", "_exit_name": "scalar_tensor", "_exit_type": "scalar_tensor"},
        records,
        cpu_outputs,
        set(),
    ) == "scalar_tensor"


def test_match_cpu_output_maps_baddbmm_exit_to_cpu_add(tmp_path: Path) -> None:
    npu_path = tmp_path / "baddbmm.npy"
    cpu_path = tmp_path / "add_2.npy"
    np.save(npu_path, np.zeros((2, 4, 4), dtype=np.float32))
    np.save(cpu_path, np.zeros((2, 4, 4), dtype=np.float32))
    records = {
        "baddbmm": {
            "outputs": [{"index": 0, "path": npu_path, "shape": (2, 4, 4), "size": 32}],
        }
    }
    cpu_outputs = {
        "add_2": {"path": cpu_path, "shape": (2, 4, 4), "size": 32, "type": "add"},
    }
    op = {"name": "baddbmm", "_exit_name": "baddbmm", "_exit_type": "baddbmm"}

    assert _match_cpu_output(op, records, cpu_outputs, set()) == "add_2"


def test_match_cpu_output_maps_masked_fill_to_cpu_where(tmp_path: Path) -> None:
    npu_path = tmp_path / "masked_fill.npy"
    cpu_path = tmp_path / "where.npy"
    np.save(npu_path, np.array([[0.0, -np.inf]], dtype=np.float32))
    np.save(cpu_path, np.array([[0.0, -np.inf]], dtype=np.float32))
    records = {
        "masked_fill": {
            "outputs": [{"index": 0, "path": npu_path, "shape": (1, 2), "size": 2}],
        }
    }
    cpu_outputs = {
        "where": {"path": cpu_path, "shape": (1, 2), "size": 2, "type": "where"},
    }
    op = {"name": "masked_fill", "_exit_name": "masked_fill", "_exit_type": "masked_fill"}

    assert _match_cpu_output(op, records, cpu_outputs, set()) == "where"


def test_build_inductor_graph_maps_decomposed_softmax_to_original_aten(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op0.writes = [MemoryDep('buf0', c0, {c0: 8})]
op0.unmet_dependencies = [MemoryDep('arg0_1', c0, {c0: 8})]
op0.met_dependencies = []
""",
        encoding="utf-8",
    )
    kernel = "autofused__softmax_deadbeef"
    benchmark = trace / kernel / "benchmark.py"
    benchmark.parent.mkdir()
    benchmark.write_text(
        """
# Topologically Sorted Source Nodes: [attention], Original ATen: [aten._softmax]
# Graph fragment:
#   %arg0_1 : Tensor "f32[2, 4]" = PlaceHolder[target=arg0_1]
#   %amax : Tensor "f32[2, 1]" = call_function[target=torch.ops.aten.amax.default](args = (%arg0_1, [-1], True), kwargs = {})
#   %div : Tensor "f32[2, 4]" = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %amax), kwargs = {})
#   return %amax,%div
op0: SchedulerNode(ComputedBuffer)
""",
        encoding="utf-8",
    )
    capture_name = f"{kernel}__div"
    (trace / "output_code.py").write_text(
        f"run_intermediate_hooks('{capture_name}::autofusex_output::0', buf0)\n",
        encoding="utf-8",
    )
    npu_path = tmp_path / "softmax.npu.npy"
    cpu_path = tmp_path / "softmax.cpu.npy"
    np.save(npu_path, np.array([-np.inf, 0.5, 2.0, 0.5, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    np.save(cpu_path, np.array([0.0, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    records = {
        capture_name: {
            "outputs": [{"index": 0, "path": npu_path, "dtype": "float32", "shape": (2, 4), "size": 8}]
        }
    }
    cpu_outputs = {
        "_softmax": {"path": cpu_path, "dtype": "float32", "shape": (2, 4), "size": 8, "type": "_softmax"}
    }

    graph = build_inductor_autofuse_graph(tmp_path / "compile", records, cpu_outputs)
    fusion = next(op for op in graph["graph"][0]["op"] if op["type"] == "AutofuseFusion")

    assert fusion["original_aten_types"] == ["_softmax"]
    assert fusion["output_desc"][0]["attr"][0]["key"] == "_autofusex_cpu_origin_name"
    assert fusion["output_desc"][0]["attr"][0]["value"]["s"] == "_softmax"


def test_reused_autofuse_kernel_keeps_fusion_provenance_and_only_live_outputs(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('tmp0', c0, {c0: 8}), MemoryDep('buf0', c0, {c0: 8})]
op0.unmet_dependencies = [MemoryDep('arg0_1', c0, {c0: 8})]
op0.met_dependencies = []
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 8})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 8})]
op1.met_dependencies = []
""",
        encoding="utf-8",
    )
    kernel = "autofused_add_deadbeef"
    benchmark = trace / kernel / "benchmark.py"
    benchmark.parent.mkdir()
    benchmark.write_text(
        """
# %arg0_1 : Tensor "f32[2, 4]" = PlaceHolder[target=arg0_1]
# %add : Tensor "f32[2, 4]" = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1), kwargs = {})
# return %add
op0: SchedulerNode(ComputedBuffer)
""",
        encoding="utf-8",
    )
    first = f"{kernel}__add"
    second = f"{kernel}__add_1"
    (trace / "output_code.py").write_text(
        f"run_intermediate_hooks('{first}::autofusex_output::0', buf0)\n"
        f"run_intermediate_hooks('{second}::autofusex_output::0', buf1)\n",
        encoding="utf-8",
    )
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    records = {}
    cpu_outputs = {}
    for name, origin in ((first, "add"), (second, "add_1")):
        npu_path = tmp_path / f"{origin}.npu.npy"
        cpu_path = tmp_path / f"{origin}.cpu.npy"
        np.save(npu_path, values)
        np.save(cpu_path, values)
        records[name] = {
            "outputs": [{"index": 0, "path": npu_path, "dtype": "float32", "shape": (2, 4), "size": 8}]
        }
        cpu_outputs[origin] = {"path": cpu_path, "shape": (2, 4), "size": 8, "type": "add"}

    graph = build_inductor_autofuse_graph(tmp_path / "compile", records, cpu_outputs)
    fusions = [op for op in graph["graph"][0]["op"] if op["type"] == "AutofuseFusion"]

    assert [op["name"] for op in fusions] == [first, second]
    assert all(len(op["output_desc"]) == 1 for op in fusions)
    assert [op["output_desc"][0]["attr"][0]["value"]["s"] for op in fusions] == ["add", "add_1"]


def test_build_inductor_graph_attaches_fusion_rejections_to_fused_nodes(tmp_path: Path) -> None:
    trace = tmp_path / "compile" / "torchinductor" / "model__0"
    trace.mkdir(parents=True)
    (trace / "ir_post_fusion.txt").write_text(
        """
op0_op1: FusedSchedulerNode(ComputedBuffer)
op0_op1.writes = [MemoryDep('buf0', c0, {c0: 8})]
op0_op1.unmet_dependencies = []
op0_op1.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 8})]

op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf1', c0, {c0: 8})]
op2.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 8})]
op2.met_dependencies = []
""",
        encoding="utf-8",
    )
    (trace / "fx_graph_readable.py").write_text(
        'def forward(self, arg0_1: "f32[2, 4]"):\n    return arg0_1\n', encoding="utf-8"
    )
    kernel = "autofused_add_relu_deadbeef"
    benchmark = trace / kernel / "benchmark.py"
    benchmark.parent.mkdir()
    benchmark.write_text(
        """
        # Graph fragment:
        #   %arg0_1 : Tensor "f32[2, 4][4, 1]npu:0" = PlaceHolder[target=arg0_1]
        #   %add : Tensor "f32[2, 4][4, 1]npu:0" = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1.0), kwargs = {})
        #   %relu : Tensor "f32[2, 4][4, 1]npu:0" = call_function[target=torch.ops.aten.relu.default](args = (%add,), kwargs = {})
        #   return %relu
        op0: SchedulerNode(ComputedBuffer)
        op1: SchedulerNode(ComputedBuffer)
        graph_hint = ascir.HintGraph('graph_hint')
        data = ascir.ops.Data('graph_hint/data', graph_hint)
        data.attr.ir_attr.index = 0
        add = ascir.ops.Add('graph_hint/add', graph_hint)
        relu = ascir.ops.Relu('graph_hint/relu', graph_hint)
        output = ascir.ops.Output('graph_hint/output', graph_hint)
        output.x = relu.y
        graph_hint.infer_dtypes()
""",
        encoding="utf-8",
    )
    npu_path = tmp_path / "npu.npy"
    values = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.save(npu_path, values)
    capture_name = f"{kernel}__relu"
    records = {
        capture_name: {
            "name": capture_name,
            "outputs": [{"index": 0, "path": npu_path, "dtype": "float32", "shape": (2, 4), "size": 8}],
        },
        "exp_1": {
            "name": "exp_1",
            "outputs": [{"index": 0, "path": npu_path, "dtype": "float32", "shape": (2, 4), "size": 8}],
        },
    }
    cpu_outputs = {"relu": {"path": npu_path, "dtype": "float32", "shape": (2, 4), "size": 8, "type": "relu"}}
    diagnostics = {
        "lowering_fallbacks": [],
        "fusion_decision_groups": [
            [
                {
                    "kind": "vertical",
                    "node1": ["op0", "op1"],
                    "node2": ["op2"],
                    "result": False,
                    "reason": "cannot fuse: mixed dtype in fusion",
                    "triton_can_fuse": True,
                }
            ]
        ],
    }

    graph = build_inductor_autofuse_graph(tmp_path / "compile", records, cpu_outputs, diagnostics)
    ops = graph["graph"][0]["op"]
    fusion = next(op for op in ops if op["type"] == "AutofuseFusion")

    assert fusion["fusion_rejections"], "fused kernel should carry further-fusion rejections"
    rejection = fusion["fusion_rejections"][0]
    assert rejection["category"] == "fusion_rejection"
    assert rejection["kind"] == "vertical"
    assert rejection["direction"] == "output"
    assert "cannot fuse" in rejection["reason"]
    assert rejection["peer"] == "inductor_op2"
    # unfused_reasons stays reserved for non-fused nodes
    assert "unfused_reasons" not in fusion or not fusion["unfused_reasons"]
