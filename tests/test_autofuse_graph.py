from __future__ import annotations

from pathlib import Path

from autofusedebugger.autofuse_graph import collect_autofuse_graphs, parse_ascgraph_text


ASCGRAPH = """
Graph: graph_hint_0_B0Y0_S0G0C0

Nodes:
  [0] data : ascir.ops.Data
        .y: f32
            .mem = GM[tensor_id=0, pos=GM]
  [1] load1 : ascir.ops.Load
        .exec_condition = cache_block_split_fused_brc_axis
        .x = {data.y}
        .y: f32
            .mem = UB[tensor_id=1, que_id=0, buf_num=2, reuse_id=0, depth=2, pos=VECIN]
  [2] load2 : ascir.ops.Load
        .x = {data.y}
        .y: f32
            .mem = UB[tensor_id=2, que_id=0, buf_num=2, reuse_id=0, depth=2, pos=VECIN]
  [3] load3 : ascir.ops.Load
        .x = {data.y}
        .y: f32
            .mem = UB[tensor_id=3, que_id=0, buf_num=2, reuse_id=1, depth=2, pos=VECIN]
  [4] add : ascir.ops.Add
        .x = {load1.y, load3.y}
        .y: f32
            .mem = UB[tensor_id=4, que_id=1, buf_num=1, reuse_id=1, depth=2, pos=VECCALC]
            .tmp_buf = {{ id=1 }}
  [5] relu : ascir.ops.Relu
        .x = {add.y}
        .y: f32
            .mem = UB[tensor_id=5, que_id=1, buf_num=2, reuse_id=2, depth=2, pos=VECOUT]
--------------------------------------------------------------------------------
"""


def test_parse_ascgraph_classifies_load_queue_usage() -> None:
    graph = parse_ascgraph_text(ASCGRAPH, "/tmp/ascgraph_AutoFuseAfterOptimize_0.txt")

    loads = {node["name"]: node["load_class"] for node in graph["nodes"] if node["type"] == "Load"}
    assert loads == {"load1": "shared", "load2": "shared", "load3": "reused"}
    queue_classes = {node["name"]: node["queue_class"] for node in graph["nodes"] if "queue_class" in node}
    assert queue_classes == {
        "load1": "shared",
        "load2": "shared",
        "load3": "reused",
        "add": "reused",
        "relu": "reused",
    }
    assert graph["edges"] == [
        {"src": 0, "dst": 1},
        {"src": 0, "dst": 2},
        {"src": 0, "dst": 3},
        {"src": 1, "dst": 4},
        {"src": 3, "dst": 4},
        {"src": 4, "dst": 5},
    ]
    assert [(queue["id"], queue["uses"], queue["load_nodes"]) for queue in graph["queues"]] == [
        (0, [1, 2, 3], [1, 2, 3]),
        (1, [4, 5], []),
    ]
    assert graph["queues"][0]["instances"] == [
        {
            "reuse_id": 0,
            "depth": 2,
            "buf_num": 2,
            "queue_class": "shared",
            "producer_nodes": [1, 2],
            "consumer_nodes": [4],
            "start": 1,
            "end": 4,
        },
        {
            "reuse_id": 1,
            "depth": 2,
            "buf_num": 2,
            "queue_class": "reused",
            "producer_nodes": [3],
            "consumer_nodes": [4],
            "start": 3,
            "end": 4,
        },
    ]
    assert graph["queues"][1]["instances"][0]["producer_nodes"] == [4]
    assert graph["queues"][1]["instances"][0]["consumer_nodes"] == [5]
    assert graph["buffers"] == [
        {
            "id": 1,
            "uses": [4],
            "instances": [
                {
                    "kind": "tmp_buf",
                    "source_node": 4,
                    "producer_nodes": [4],
                    "consumer_nodes": [],
                    "start": 4,
                    "end": 4,
                }
            ],
        }
    ]
    nodes = {node["name"]: node for node in graph["nodes"]}
    assert nodes["load1"]["exec_condition"] == "cache_block_split_fused_brc_axis"
    assert nodes["load1"]["broadcast_cache"] is True
    assert nodes["add"]["exec_condition"] == "no_cache"
    assert nodes["add"]["broadcast_cache"] is False


def test_collect_autofuse_graphs_reads_all_matching_suffixes(tmp_path: Path) -> None:
    kernel = "autofused_add_relu_0123456789abcdef0123456789abcdef"
    kernel_dir = tmp_path / "compile" / kernel / "choice_0"
    kernel_dir.mkdir(parents=True)
    first = kernel_dir / "ascgraph_00001_graph_a_AutoFuseAfterOptimize_0.txt"
    second = kernel_dir / "ascgraph_00002_graph_b_AutoFuseAfterOptimize_1.txt"
    first.write_text(ASCGRAPH, encoding="utf-8")
    second.write_text(ASCGRAPH.replace("graph_hint_0_B0Y0_S0G0C0", "graph_b"), encoding="utf-8")

    graphs = collect_autofuse_graphs(tmp_path)

    assert [graph["name"] for graph in graphs] == ["graph_hint_0_B0Y0_S0G0C0", "graph_b"]
    assert [graph["kernel_name"] for graph in graphs] == [kernel, kernel]
    assert graphs[0]["variant_name"] == "ascgraph_00001_graph_a_AutoFuseAfterOptimize_0"
    assert graphs[0]["source_relative"].startswith(f"compile/{kernel}/choice_0/")


def test_parse_ascgraph_builds_buffer_output_interval() -> None:
    graph = parse_ascgraph_text(
        """
Graph: buffer_graph
Nodes:
  [0] data : ascir.ops.Data
        .y: f32
            .mem = GM[tensor_id=0]
  [1] add : ascir.ops.Add
        .x = {data.y}
        .y: f32
            .mem = UB[tensor_id=1, buf_id=3]
  [2] relu : ascir.ops.Relu
        .x = {add.y}
        .y: f32
            .mem = GM[tensor_id=2]
--------------------------------------------------------------------------------
"""
    )

    assert graph["buffers"] == [
        {
            "id": 3,
            "uses": [1],
            "instances": [
                {
                    "kind": "output",
                    "source_node": 1,
                    "producer_nodes": [1],
                    "consumer_nodes": [2],
                    "start": 1,
                    "end": 2,
                }
            ],
        }
    ]
