from __future__ import annotations

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
    fullgraph: bool = True,
    dynamic: bool = False,
    html_path: str | Path | None = None,
    cann_root: str | Path | None = None,
    device: str = "npu:0",
    rtol: float | None = None,
    atol: float | None = None,
) -> dict[str, Any]:
    frontend = get_framework_adapter(framework)
    target = get_backend_adapter(backend)
    invocation = frontend.capture(
        script=script,
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
        rtol=rtol,
        atol=atol,
    )
