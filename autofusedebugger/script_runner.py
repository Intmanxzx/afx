from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from autofusedebugger.adapters import get_backend_adapter, get_framework_adapter


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
        frontend = get_framework_adapter(framework)
        target = get_backend_adapter(backend)
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
        )
    finally:
        sys.path[:] = old_path
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath
