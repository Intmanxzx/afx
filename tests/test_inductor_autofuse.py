from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

import autofusedebugger.inductor_autofuse as inductor_autofuse
from autofusedebugger.graph import extract_edges
from autofusedebugger.inductor_autofuse import (
    _match_cpu_output,
    activate_autofuse_backend,
    build_inductor_autofuse_graph,
)


def test_activate_autofuse_backend_prefers_torch_npu_bundle(monkeypatch) -> None:
    calls = []

    def import_module(name):
        calls.append(name)
        return object()

    monkeypatch.setattr(inductor_autofuse.importlib, "import_module", import_module)

    assert activate_autofuse_backend() == "torch_npu._inductor.ascendc"
    assert calls == ["torch_npu._inductor.ascendc"]


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
        f"run_intermediate_hooks('{first}::autofusedebugger_output::0', buf0)\n"
        f"run_intermediate_hooks('{second}::autofusedebugger_output::0', buf1)\n",
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
