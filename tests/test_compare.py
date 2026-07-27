from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autofusedebugger.compare import (
    TensorRef,
    _metrics,
    compare_manifests,
    compare_pair,
    compare_three_manifests,
    resolve_tolerances,
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_compare_fused_output_to_cpu_origin(tmp_path: Path) -> None:
    np.save(tmp_path / "cpu_relu.npy", np.array([1.0, 2.0], dtype=np.float32))
    np.save(tmp_path / "npu_fused.npy", np.array([1.0, 2.001], dtype=np.float32))

    cpu = {
        "schema": "autofusedebugger-manifest/1",
        "side": "cpu",
        "model": {"name": "m"},
        "artifacts": {"root": str(tmp_path)},
        "nodes": [
            {
                "name": "Relu_1",
                "type": "Relu",
                "outputs": [{"index": 0, "path": "cpu_relu.npy", "format": "ND"}],
            }
        ],
        "edges": [],
        "fusions": [],
    }
    npu = {
        "schema": "autofusedebugger-manifest/1",
        "side": "npu",
        "model": {"name": "m"},
        "artifacts": {"root": str(tmp_path)},
        "nodes": [
            {
                "name": "autofuse_0_Add_Relu",
                "type": "AscBackend",
                "outputs": [{"index": 0, "path": "npu_fused.npy", "format": "ND"}],
            }
        ],
        "edges": [],
        "fusions": [
            {
                "fused_op": "autofuse_0_Add_Relu",
                "fused_index": 0,
                "origin_op": "Relu_1",
                "origin_index": 0,
            }
        ],
    }
    cpu_path = tmp_path / "cpu.json"
    npu_path = tmp_path / "npu.json"
    _write_json(cpu_path, cpu)
    _write_json(npu_path, npu)

    report = compare_manifests(cpu_path, npu_path, rtol=1e-2, atol=1e-2)
    assert report["summary"]["failed"] == 0
    assert report["results"][0]["cpu_op"] == "Relu_1"
    assert report["graphs"]["cpu"]["nodes"][0]["name"] == "Relu_1"
    assert report["graphs"]["npu"]["fusions"][0]["origin_op"] == "Relu_1"


def test_fusion_internal_cpu_outputs_are_not_observable(tmp_path: Path) -> None:
    for name in ("addmm", "mul", "relu"):
        np.save(tmp_path / f"{name}.npy", np.array([1.0, 2.0], dtype=np.float32))
    np.save(tmp_path / "npu_fused.npy", np.array([1.0, 2.0], dtype=np.float32))

    base = {
        "schema": "autofusedebugger-manifest/1",
        "model": {"name": "m"},
        "artifacts": {"root": str(tmp_path)},
        "edges": [],
    }
    cpu = dict(
        base,
        side="cpu",
        nodes=[
            {"name": name, "type": name, "outputs": [{"index": 0, "path": f"{name}.npy"}]}
            for name in ("addmm", "mul", "relu")
        ],
        fusions=[],
    )
    npu = dict(
        base,
        side="npu",
        nodes=[
            {
                "name": "autofused_addmm_mul_relu",
                "type": "AutofuseFusion",
                "origin_ops": ["addmm", "mul", "relu"],
                "outputs": [{"index": 0, "path": "npu_fused.npy"}],
            }
        ],
        fusions=[
            {
                "fused_op": "autofused_addmm_mul_relu",
                "fused_index": 0,
                "origin_op": "relu",
                "origin_index": 0,
            }
        ],
    )
    _write_json(tmp_path / "cpu.json", cpu)
    _write_json(tmp_path / "npu.json", npu)

    report = compare_manifests(tmp_path / "cpu.json", tmp_path / "npu.json", rtol=0, atol=0)

    assert report["summary"]["status"] == "PASS"
    assert report["summary"]["compared"] == 1
    assert report["summary"]["observable_total"] == 1
    assert report["summary"]["coverage"] == 1.0
    assert report["summary"]["not_observable"] == 2
    assert report["summary"]["unmapped"] == 0
    internal = [row for row in report["results"] if row["status"] == "NOT_OBSERVABLE_ON_NPU"]
    assert {row["cpu_op"] for row in internal} == {"addmm", "mul"}
    assert all(row["mapping_method"] == "fusion_internal" for row in internal)


def test_shape_mismatch_is_reported(tmp_path: Path) -> None:
    np.save(tmp_path / "cpu.npy", np.array([1.0, 2.0], dtype=np.float32))
    np.save(tmp_path / "npu.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32))
    base = {
        "schema": "autofusedebugger-manifest/1",
        "model": {"name": "m"},
        "artifacts": {"root": str(tmp_path)},
        "nodes": [],
        "edges": [],
        "fusions": [],
    }
    cpu = dict(base, side="cpu", nodes=[{"name": "A", "type": "Add", "outputs": [{"index": 0, "path": "cpu.npy"}]}])
    npu = dict(base, side="npu", nodes=[{"name": "A", "type": "Add", "outputs": [{"index": 0, "path": "npu.npy"}]}])
    _write_json(tmp_path / "cpu.json", cpu)
    _write_json(tmp_path / "npu.json", npu)
    report = compare_manifests(tmp_path / "cpu.json", tmp_path / "npu.json", rtol=0, atol=0)
    assert report["summary"]["failed"] == 1
    assert report["results"][0]["status"].startswith("SHAPE_MISMATCH")


def test_graph_summary_includes_input_dtype_and_shape(tmp_path: Path) -> None:
    np.save(tmp_path / "a.npy", np.array([1.0, 2.0], dtype=np.float32))
    np.save(tmp_path / "b.npy", np.array([2.0, 3.0], dtype=np.float32))
    manifest = {
        "schema": "autofusedebugger-manifest/1",
        "side": "cpu",
        "model": {"name": "m"},
        "artifacts": {"root": str(tmp_path)},
        "nodes": [
            {
                "name": "A",
                "type": "Data",
                "outputs": [{"index": 0, "path": "a.npy", "format": "ND", "dtype": "float32", "shape": [2]}],
            },
            {
                "name": "B",
                "type": "Add",
                "outputs": [{"index": 0, "path": "b.npy", "format": "ND", "dtype": "float32", "shape": [2]}],
                "origin_subgraph": {
                    "source": "CANN GE PreRunBegin",
                    "nodes": [{"name": "InnerAdd", "type": "Add", "outputs": []}],
                    "edges": [],
                },
            },
        ],
        "edges": [{"src": "A", "dst": "B", "dst_input": 0}],
        "fusions": [],
    }
    _write_json(tmp_path / "cpu.json", manifest)
    _write_json(tmp_path / "npu.json", dict(manifest, side="npu"))

    report = compare_manifests(tmp_path / "cpu.json", tmp_path / "npu.json", rtol=0, atol=0)

    node_b = report["graphs"]["cpu"]["nodes"][1]
    assert node_b["inputs"] == [
        {"src": "A", "index": 0, "format": "ND", "dtype": "float32", "shape": [2]}
    ]
    assert node_b["origin_subgraph"]["source"] == "CANN GE PreRunBegin"


def test_unmatched_npu_output_is_reported_as_incomplete(tmp_path: Path) -> None:
    np.save(tmp_path / "cpu.npy", np.array([1.0], dtype=np.float32))
    np.save(tmp_path / "npu.npy", np.array([1.0], dtype=np.float32))
    base = {
        "schema": "autofusedebugger-manifest/1",
        "model": {"name": "m"},
        "artifacts": {"root": str(tmp_path)},
        "edges": [],
        "fusions": [],
    }
    cpu = dict(
        base,
        side="cpu",
        nodes=[{"name": "embedding", "type": "embedding", "outputs": [{"index": 0, "path": "cpu.npy"}]}],
    )
    npu = dict(
        base,
        side="npu",
        nodes=[{"name": "GatherV2.GatherV2", "type": "GatherV2", "outputs": [{"index": 0, "path": "npu.npy"}]}],
    )
    _write_json(tmp_path / "cpu.json", cpu)
    _write_json(tmp_path / "npu.json", npu)

    report = compare_manifests(tmp_path / "cpu.json", tmp_path / "npu.json", rtol=0, atol=0)

    assert report["summary"] == {
        "total": 1,
        "observable_total": 1,
        "compared": 0,
        "unmatched": 1,
        "unmapped": 1,
        "not_captured": 0,
        "not_observable": 0,
        "dumped": 1,
        "coverage": 0.0,
        "status": "INCOMPLETE",
        "passed": 0,
        "failed": 0,
        "strict_status": "INCOMPLETE",
        "strict_passed": 0,
        "strict_failed": 0,
        "cann_status": "INCOMPLETE",
        "cann_passed": 0,
        "cann_failed": 0,
        "dual_status": "INCOMPLETE",
        "dual_passed": 0,
        "dual_failed": 0,
        "standard_disagreements": 0,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
    }
    assert report["results"][0]["status"] == "UNMATCHED_CPU_OP"
    assert report["results"][0]["compared"] is False


def test_missing_npu_dump_is_distinguished_from_missing_cpu_mapping(tmp_path: Path) -> None:
    base = {
        "schema": "autofusedebugger-manifest/1",
        "model": {"name": "m"},
        "artifacts": {"root": str(tmp_path)},
        "edges": [],
        "fusions": [],
    }
    cpu = dict(base, side="cpu", nodes=[])
    npu = dict(
        base,
        side="npu",
        nodes=[{"name": "view", "type": "aten.reshape", "outputs": [{"index": 0, "path": None}]}],
    )
    _write_json(tmp_path / "cpu.json", cpu)
    _write_json(tmp_path / "npu.json", npu)

    report = compare_manifests(tmp_path / "cpu.json", tmp_path / "npu.json", rtol=0, atol=0)

    assert report["results"][0]["status"] == "NPU_OUTPUT_NOT_CAPTURED"
    assert report["summary"]["not_captured"] == 1
    assert report["summary"]["unmapped"] == 0


def test_metrics_handles_equal_infinity_without_warning() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = _metrics(np.array([1.0, -np.inf]), np.array([1.0, -np.inf]), rtol=0, atol=0)

    assert result["passed"] is True
    assert result["cann_passed"] is False
    assert result["cann_cosine_similarity"] is None
    assert result["mismatched"] == 0
    assert result["finite_pair_count"] == 1
    assert result["max_error_index"] == [0]
    assert result["nonfinite"]["cpu"]["neg_inf"] == 1
    assert result["nonfinite"]["npu"]["neg_inf"] == 1
    assert result["nonfinite"]["aligned"]["neg_inf"] == 1
    assert result["nonfinite"]["different_positions"] == 0
    assert result["first_nonfinite_index"] == [1]
    assert result["first_nonfinite_cpu"] == "-Inf"
    assert result["first_nonfinite_npu"] == "-Inf"
    assert result["cann_unavailable_reason"] == "NONFINITE_INPUT"
    json.dumps(result, allow_nan=False)


def test_dtype_aware_default_tolerances() -> None:
    assert resolve_tolerances("float32", "float32") == (1e-5, 1e-5, "dtype")
    assert resolve_tolerances("float32", "float16") == (1e-3, 1e-3, "dtype")
    assert resolve_tolerances("bfloat16", "float32") == (1e-2, 1e-2, "dtype")
    assert resolve_tolerances("int64", "int32") == (0.0, 0.0, "dtype")


def test_compare_pair_uses_strict_float32_default_and_records_threshold(tmp_path: Path) -> None:
    cpu_path = tmp_path / "cpu.npy"
    npu_path = tmp_path / "npu.npy"
    np.save(cpu_path, np.array([1.0], dtype=np.float32))
    np.save(npu_path, np.array([1.00003], dtype=np.float32))

    row = compare_pair(
        TensorRef("cpu", 0, cpu_path, "ND", "Add"),
        TensorRef("npu", 0, npu_path, "ND", "Add"),
    )

    assert row["passed"] is False
    assert row["strict_passed"] is False
    assert (row["rtol"], row["atol"], row["tolerance_source"]) == (1e-5, 1e-5, "dtype")


def test_compare_pair_embeds_elementwise_tensor_data(tmp_path: Path) -> None:
    cpu_path = tmp_path / "cpu.npy"
    npu_path = tmp_path / "npu.npy"
    np.save(cpu_path, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    np.save(npu_path, np.array([[1.0, 2.1], [3.0, 4.0]], dtype=np.float32))

    row = compare_pair(
        TensorRef("cpu", 0, cpu_path, "ND", "Add"),
        TensorRef("npu", 0, npu_path, "ND", "Add"),
    )

    payload = row["elementwise_data"]
    assert payload["shape"] == [2, 2]
    assert payload["logical_dtype"] == "float32"
    assert payload["cpu"]["encoding"] == "base64"
    assert payload["npu"]["encoding"] == "base64"
    assert payload["strict_mask"]["dtype"] == "uint8"


def test_compare_three_manifests_returns_all_pairwise_results(tmp_path: Path) -> None:
    value = np.array([1.0, 2.0], dtype=np.float32)
    np.save(tmp_path / "cpu.npy", value)
    np.save(tmp_path / "eager.npy", value)
    np.save(tmp_path / "autofuse.npy", value)

    def manifest(side: str, output: str) -> dict:
        return {
            "schema": "autofusedebugger-manifest/1",
            "side": side,
            "model": {"name": "m"},
            "artifacts": {"root": str(tmp_path)},
            "nodes": [{"name": "add", "type": "Add", "outputs": [{"index": 0, "path": output}]}],
            "edges": [],
            "fusions": [],
        }

    paths = []
    for name, side, output in (("cpu", "cpu", "cpu.npy"), ("eager", "npu_eager", "eager.npy"), ("autofuse", "npu", "autofuse.npy")):
        path = tmp_path / f"{name}.json"
        _write_json(path, manifest(side, output))
        paths.append(path)

    report = compare_three_manifests(*paths, rtol=0, atol=0)
    assert report["comparison_mode"] == "three_way"
    assert report["comparison_order"] == ["cpu_npu_autofuse", "cpu_npu_eager", "npu_eager_npu_autofuse"]
    assert all(report["comparisons"][key]["summary"]["dual_failed"] == 0 for key in report["comparison_order"])
    cpu_autofuse = report["comparisons"]["cpu_npu_autofuse"]["results"][0]["elementwise_data"]
    cpu_eager = report["comparisons"]["cpu_npu_eager"]["results"][0]["elementwise_data"]
    eager_autofuse = report["comparisons"]["npu_eager_npu_autofuse"]["results"][0]["elementwise_data"]
    assert cpu_autofuse["cpu"] is cpu_eager["cpu"]
    assert cpu_eager["npu"] is eager_autofuse["cpu"]
    assert cpu_autofuse["npu"] is eager_autofuse["npu"]


def test_compare_pair_preserves_bfloat16_logical_dtype_with_float32_storage(tmp_path: Path) -> None:
    cpu_path = tmp_path / "cpu.npy"
    npu_path = tmp_path / "npu.npy"
    np.save(cpu_path, np.array([1.0], dtype=np.float32))
    np.save(npu_path, np.array([1.0], dtype=np.float32))

    row = compare_pair(
        TensorRef("cpu", 0, cpu_path, "ND", "Add", "bfloat16"),
        TensorRef("npu", 0, npu_path, "ND", "Add", "bfloat16"),
    )

    assert row["dtype"] == "bfloat16"
    assert row["storage_dtype"] == "float32"
    assert (row["rtol"], row["atol"]) == (1e-2, 1e-2)


def test_compare_pair_uses_lower_precision_dtype_and_allows_override(tmp_path: Path) -> None:
    cpu_path = tmp_path / "cpu.npy"
    npu_path = tmp_path / "npu.npy"
    np.save(cpu_path, np.array([1.0], dtype=np.float32))
    np.save(npu_path, np.array([1.0], dtype=np.float16))
    cpu = TensorRef("cpu", 0, cpu_path, "ND", "Add")
    npu = TensorRef("npu", 0, npu_path, "ND", "Add")

    automatic = compare_pair(cpu, npu)
    overridden = compare_pair(cpu, npu, rtol=2e-2, atol=3e-2)

    assert (automatic["rtol"], automatic["atol"], automatic["tolerance_source"]) == (1e-3, 1e-3, "dtype")
    assert (overridden["rtol"], overridden["atol"], overridden["tolerance_source"]) == (2e-2, 3e-2, "override")


def test_integer_outputs_require_exact_match(tmp_path: Path) -> None:
    cpu_path = tmp_path / "cpu.npy"
    npu_path = tmp_path / "npu.npy"
    np.save(cpu_path, np.array([1], dtype=np.int32))
    np.save(npu_path, np.array([2], dtype=np.int32))

    row = compare_pair(
        TensorRef("cpu", 0, cpu_path, "ND", "Add"),
        TensorRef("npu", 0, npu_path, "ND", "Add"),
    )

    assert row["passed"] is False
    assert row["cann_passed"] is False
    assert (row["rtol"], row["atol"]) == (0.0, 0.0)


def test_cann_aggregate_can_pass_when_strict_outlier_check_fails() -> None:
    golden = np.linspace(1.0, 2.0, 1000, dtype=np.float64)
    actual = golden.copy()
    actual[500] += 1.0

    result = _metrics(golden, actual, rtol=1e-5, atol=1e-5)

    assert result["strict_passed"] is False
    assert result["cann_passed"] is True
    assert result["cann_cosine_similarity"] > 0.99
    assert result["relative_euclidean_distance"] < 0.05
    assert result["kl_divergence"] < 0.005


def test_cann_zero_vector_check_can_fail_when_strict_atol_passes() -> None:
    golden = np.zeros(1000, dtype=np.float32)
    actual = np.full(1000, 1e-6, dtype=np.float32)

    result = _metrics(golden, actual, rtol=1e-5, atol=1e-5)

    assert result["strict_passed"] is True
    assert result["cann_passed"] is False
    assert result["cann_cosine_similarity"] == 0.0
    assert result["relative_euclidean_distance"] == 1.0
