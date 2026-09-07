from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

from autofusedebugger.adapters import detect_framework, get_backend_adapter, get_framework_adapter
from autofusedebugger.progress import progress_message


_PYTHON_EXECUTABLE_RE = re.compile(r"python(?:\d+(?:\.\d+)*)?$")
_PYTHON_OPTIONS_WITH_VALUE = {"-W", "-X"}


def parse_python_script_invocation(argv: Sequence[str]) -> tuple[Path, tuple[str, ...]] | None:
    """Parse the convenience form: ``autofusedebugger python3 model.py ...``."""
    if not argv or not _PYTHON_EXECUTABLE_RE.fullmatch(Path(argv[0]).name):
        return None

    tokens = list(argv[1:])
    script_index: int | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"-c", "-m"}:
            raise ValueError("direct mode accepts a Python script path, not -c or -m")
        if token == "--":
            index += 1
            if index < len(tokens):
                script_index = index
            break
        if token in _PYTHON_OPTIONS_WITH_VALUE:
            if index + 1 >= len(tokens):
                raise ValueError(f"Python option {token} requires a value")
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        script_index = index
        break

    if script_index is None:
        raise ValueError("direct mode requires a Python script, for example: python3 model.py")

    script = Path(tokens[script_index]).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Python script not found: {script}")
    return script, tuple(tokens[script_index + 1 :])


def default_direct_output_paths(script: str | Path, cwd: str | Path | None = None) -> tuple[Path, Path]:
    """Return the artifact directory and HTML path for direct mode."""
    root = Path(cwd or Path.cwd()).resolve()
    stem = Path(script).stem or "model"
    return root / f"{stem}_autofusedebugger", root / f"{stem}_autofusedebugger.html"


def run_framework_script(
    *,
    script: str | Path,
    out_dir: str | Path,
    framework: str = "auto",
    backend: str = "inductor_autofuse",
    script_args: tuple[str, ...] = (),
    capture_call: int = 0,
    module_pattern: str | None = None,
    name: str | None = None,
    fullgraph: bool = False,
    dynamic: bool = False,
    rewrite_device_hints: bool = False,
    html_path: str | Path | None = None,
    cann_root: str | Path | None = None,
    device: str = "npu:0",
    rtol: float | None = None,
    atol: float | None = None,
    save_elementwise_data: bool = False,
) -> dict[str, Any]:
    path = Path(script).resolve()
    old_path = list(sys.path)
    old_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_items = [item for item in (old_pythonpath or "").split(os.pathsep) if item]
    script_dir = str(path.parent)
    sys.path.insert(0, str(path.parent))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [script_dir, *(item for item in pythonpath_items if item != script_dir)]
    )
    try:
        resolved_framework = framework
        if framework == "auto":
            resolved_framework = detect_framework(path)
            progress_message("INFO", f"framework auto-detected: {resolved_framework} script={path}")
        frontend = get_framework_adapter(resolved_framework)
        target = get_backend_adapter(backend)
        if not target.supports(resolved_framework):
            raise ValueError(
                f"backend {backend} does not support framework {resolved_framework} "
                f"(detected from {path}); choose a compatible backend or pass --framework explicitly"
            )
        invocation = frontend.capture(
            script=path,
            script_args=script_args,
            capture_call=capture_call,
            module_pattern=module_pattern,
        )
        if not target.supports(invocation.framework):
            raise ValueError(f"backend {backend} does not support framework {invocation.framework}")
        return target.run(
            invocation,
            out_dir=out_dir,
            html_path=html_path,
            cann_root=cann_root,
            device=device,
            name=name,
            fullgraph=fullgraph,
            dynamic=dynamic,
            rewrite_device_hints=rewrite_device_hints,
            rtol=rtol,
            atol=atol,
            save_elementwise_data=save_elementwise_data,
        )
    finally:
        sys.path[:] = old_path
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath
