from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autofusex.graph import build_manifest


def test_build_manifest_extracts_edges(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "graph": [
                    {
                        "op": [
                            {
                                "name": "A",
                                "type": "Data",
                                "output_desc": [{"shape": {"dim": [2]}, "data_type": "float32"}],
                            },
                            {
                                "name": "B",
                                "type": "Add",
                                "input": ["A:0"],
                                "output_desc": [{"shape": {"dim": [2]}, "data_type": "float32"}],
                                "origin_subgraph": {
                                    "source": "CANN GE PreRunBegin",
                                    "nodes": [{"name": "InnerAdd", "type": "Add", "outputs": []}],
                                    "edges": [],
                                },
                            },
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(
        side="cpu",
        graph_path=graph_path,
        dump_dir=tmp_path,
        out_path=tmp_path / "manifest.json",
        model_name="m",
    )

    assert manifest["edges"] == [{"src": "A", "dst": "B", "dst_input": 0}]
    assert manifest["nodes"][1]["origin_subgraph"]["nodes"][0]["name"] == "InnerAdd"


def test_build_manifest_preserves_ir_mapping_diagnostics(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "graph": [
                    {
                        "op": [
                            {
                                "name": "fused_relu",
                                "type": "AutofuseFusion",
                                "mapping_method": "ir_pre_fusion_canonical_id",
                                "ir_pre_fusion": {
                                    "status": "MATCHED",
                                    "scheduler_ids": ["op0"],
                                    "writes": ["buf0"],
                                    "dependencies": ["arg0_1"],
                                },
                                "output_desc": [
                                    {
                                        "shape": {"dim": [2, 4]},
                                        "data_type": "float32",
                                        "attr": [
                                            {"key": "_autofusex_cpu_origin_name", "value": {"s": "relu"}},
                                            {"key": "_autofusex_cpu_origin_output_index", "value": {"i": 0}},
                                            {"key": "_autofusex_mapping_method", "value": {"s": "ir_pre_fusion_canonical_id"}},
                                            {"key": "_autofusex_prefusion_ir_scheduler_ids", "value": {"s": "op0"}},
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    np.save(tmp_path / "fused_relu.output.0.npy", np.zeros((2, 4), dtype=np.float32))

    manifest = build_manifest(
        side="npu_autofuse",
        graph_path=graph_path,
        dump_dir=tmp_path,
        out_path=tmp_path / "manifest.json",
        model_name="m",
    )

    assert manifest["nodes"][0]["mapping_method"] == "ir_pre_fusion_canonical_id"
    assert manifest["nodes"][0]["ir_pre_fusion"]["scheduler_ids"] == ["op0"]
    assert manifest["fusions"][0]["mapping_method"] == "ir_pre_fusion_canonical_id"
    assert manifest["fusions"][0]["prefusion_ir_scheduler_ids"] == ["op0"]


def test_build_manifest_prefers_exact_output_file(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "graph": [
                    {
                        "op": [
                            {
                                "name": "mul",
                                "type": "Mul",
                                "output_desc": [{"shape": {"dim": [1]}, "data_type": "float32"}],
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "matmul.output.0.npy").write_bytes(b"wrong")
    (tmp_path / "mul.output.0.npy").write_bytes(b"right")

    manifest = build_manifest(
        side="cpu",
        graph_path=graph_path,
        dump_dir=tmp_path,
        out_path=tmp_path / "manifest.json",
        model_name="m",
    )

    assert manifest["nodes"][0]["outputs"][0]["path"] == "mul.output.0.npy"


def test_build_manifest_uses_prefusion_origin_id_for_fusion_output(tmp_path: Path) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "graph": [
                    {
                        "op": [
                            {
                                "name": "autofused_softmax__div",
                                "type": "AutofuseFusion",
                                "origin_ops": ["amax", "sub", "exp", "sum", "div"],
                                "origin_node_ids": [
                                    "graph0::amax",
                                    "graph0::sub",
                                    "graph0::exp",
                                    "graph0::sum",
                                    "graph0::div",
                                ],
                                "output_desc": [
                                    {
                                        "shape": {"dim": [2]},
                                        "data_type": "float32",
                                        "attr": [
                                            {
                                                "key": "_autofusex_prefusion_origin_id",
                                                "value": {"s": "graph0::div"},
                                            },
                                            {
                                                "key": "_autofusex_cpu_origin_name",
                                                "value": {"s": "div"},
                                            },
                                            {
                                                "key": "_autofusex_cpu_origin_output_index",
                                                "value": {"i": 0},
                                            },
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    np.save(tmp_path / "autofused_softmax__div.output.0.npy", np.ones(2, dtype=np.float32))

    manifest = build_manifest(
        side="npu_autofuse",
        graph_path=graph_path,
        dump_dir=tmp_path,
        out_path=tmp_path / "manifest.json",
    )

    assert manifest["fusions"][0]["origin_canonical_id"] == "graph0::div"
