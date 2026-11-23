from __future__ import annotations

from pathlib import Path

from autofusex.execution.runner import run_debug_cpu_command


def test_run_debug_cpu_sets_environment(tmp_path: Path) -> None:
    ret = run_debug_cpu_command(
        "python -c \"import os; print(os.environ['TORCHINDUCTOR_NPU_EXT_DEBUG'])\"",
        tmp_path,
    )
    assert ret == 0
    assert (tmp_path / "run.stdout").read_text(encoding="utf-8").strip() == "cpu"
