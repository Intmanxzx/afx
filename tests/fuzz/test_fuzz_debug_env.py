from pathlib import Path

from autofusex.workloads.fuzz.scheduling.supervisor import _debug_env
from autofusex.workloads.fuzz.torch_backend import _ensure_backend_dfx_flag


def test_backend_dfx_flag_preserves_existing_mlir_route():
    flags = _ensure_backend_dfx_flag('--mlir_codegen_config=fallback;--codegen_compile_debug=true', 'mlir')
    assert '--mlir_codegen_config=fallback' in flags
    assert '--mlir_codegen_config=mlir' not in flags
    assert '--codegen_compile_debug=true' in flags


def test_backend_dfx_flag_adds_route_without_dropping_dump_options():
    flags = _ensure_backend_dfx_flag('--codegen_compile_debug=true;--debug_dir=/tmp/dump', 'mlir')
    assert '--mlir_codegen_config=mlir' in flags
    assert '--codegen_compile_debug=true' in flags
    assert '--debug_dir=/tmp/dump' in flags


def test_autofuse_debug_environment_is_local_and_backend_scoped(tmp_path):
    env = _debug_env({"AUTOFUSE_DFX_FLAGS": "--existing=1"}, tmp_path / "scratch", tmp_path / "debug", "af-mlir")
    assert "DUMP_GE_GRAPH" not in env
    assert "DUMP_GRAPH_LEVEL" not in env
    assert "DUMP_GRAPH_PATH" not in env
    assert env["TORCH_COMPILE_DEBUG"] == "1"
    assert env["TORCH_COMPILE_DEBUG_DIR"] == str(tmp_path / "debug")
    assert "--codegen_compile_debug=true" in env["AUTOFUSE_DFX_FLAGS"]
    assert "--debug_dir=" + str(tmp_path / "debug") in env["AUTOFUSE_DFX_FLAGS"]
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(Path(tmp_path / "scratch") / "inductor")


def test_cpu_debug_environment_does_not_enable_npu_graph_dump(tmp_path):
    env = _debug_env({}, tmp_path / "scratch", tmp_path / "debug", "cpu")
    assert "DUMP_GE_GRAPH" not in env
    assert "TORCH_COMPILE_DEBUG" not in env
