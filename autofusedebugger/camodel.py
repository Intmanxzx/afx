from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def simulator_root(cann_root: str | Path) -> Path:
    root = Path(cann_root).expanduser().resolve()
    candidate = root / "x86_64-linux" / "simulator"
    if candidate.is_dir():
        return candidate
    if root.name == "simulator" and root.is_dir():
        return root
    raise NotADirectoryError(f"CANN simulator directory not found under {root}")


def discover_soc_dirs(cann_root: str | Path) -> list[Path]:
    root = simulator_root(cann_root)
    result: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        has_lib = (child / "lib").is_dir()
        has_camodel = (child / "camodel").is_dir()
        if has_lib or has_camodel:
            result.append(child)
    return result


def resolve_soc_dir(cann_root: str | Path, soc: str | None) -> Path:
    dirs = discover_soc_dirs(cann_root)
    if not dirs:
        raise NotADirectoryError(f"no simulator SoC directory found under {simulator_root(cann_root)}")
    if soc is None:
        camodel_dirs = [p for p in dirs if (p / "camodel").is_dir()]
        return camodel_dirs[0] if camodel_dirs else dirs[0]
    for item in dirs:
        if item.name == soc:
            return item
    available = ", ".join(p.name for p in dirs)
    raise FileNotFoundError(f"simulator SoC {soc!r} not found; available: {available}")


def camodel_env(cann_root: str | Path, soc: str | None = None) -> dict[str, str]:
    root = Path(cann_root).expanduser().resolve()
    sim_root = simulator_root(root)
    soc_dir = resolve_soc_dir(root, soc)
    paths: list[Path] = []
    for candidate in (soc_dir / "lib", soc_dir / "camodel", sim_root / "lib", sim_root / "bin"):
        if candidate.is_dir():
            paths.append(candidate)
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [str(p) for p in paths]
    if old_ld:
        ld_parts.append(old_ld)
    return {
        "ASCEND_TOOLKIT_HOME": str(root),
        "ASCEND_HOME_PATH": str(root),
        "AUTOFUSE_CAMODEL_SIMULATOR_ROOT": str(sim_root),
        "AUTOFUSE_CAMODEL_SOC_DIR": str(soc_dir),
        "LD_LIBRARY_PATH": ":".join(ld_parts),
    }


def camodel_info(cann_root: str | Path) -> dict[str, Any]:
    root = Path(cann_root).expanduser().resolve()
    sim_root = simulator_root(root)
    socs = []
    for item in discover_soc_dirs(root):
        socs.append(
            {
                "name": item.name,
                "path": str(item),
                "has_lib": (item / "lib").is_dir(),
                "has_camodel": (item / "camodel").is_dir(),
                "has_sim": (item / "camodel" / "sim").is_file(),
                "has_run_csh": (item / "camodel" / "run.csh").is_file(),
            }
        )
    return {"cann_root": str(root), "simulator_root": str(sim_root), "socs": socs}


def format_info(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
