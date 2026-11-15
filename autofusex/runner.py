from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from autofusex.camodel import camodel_env


def _merge_env(env_items: list[str] | None = None, base: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if base:
        env.update(base)
    for item in env_items or []:
        if "=" not in item:
            raise ValueError(f"env item must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"env key must be non-empty: {item}")
        env[key] = value
    return env


def run_command(
    command: str,
    out_dir: str | Path,
    env_items: list[str] | None = None,
    base_env: dict[str, str] | None = None,
) -> int:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = _merge_env(env_items, base_env)
    expanded = command.format(out_dir=str(out))
    proc = subprocess.run(
        expanded,
        shell=True,
        cwd=str(out),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (out / "run.stdout").write_text(proc.stdout, encoding="utf-8")
    (out / "run.stderr").write_text(proc.stderr, encoding="utf-8")
    (out / "run.command").write_text(expanded + "\n", encoding="utf-8")
    return int(proc.returncode)


def run_camodel_command(
    command: str,
    out_dir: str | Path,
    cann_root: str | Path,
    soc: str | None,
    env_items: list[str] | None = None,
) -> int:
    return run_command(command, out_dir, env_items, camodel_env(cann_root, soc))


def run_debug_cpu_command(
    command: str,
    out_dir: str | Path,
    env_items: list[str] | None = None,
) -> int:
    return run_command(
        command,
        out_dir,
        env_items,
        {
            "TORCHINDUCTOR_NPU_EXT_DEBUG": "cpu",
            "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1",
            "TORCH_COMPILE_DEBUG": "1",
        },
    )


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)
