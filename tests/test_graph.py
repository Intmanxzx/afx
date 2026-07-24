from __future__ import annotations

import json
from pathlib import Path

from autofusedebugger.graph import build_manifest


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
