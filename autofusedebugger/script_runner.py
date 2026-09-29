from __future__ import annotations

import os
import re
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from autofusedebugger.agent import AgentConfig
from autofusedebugger.adapters import (
    detect_framework,
    find_cann_ge_root,
    get_backend_adapter,
    get_framework_adapter,
)
from autofusedebugger.normalization import normalize_script, recover_script_error
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
    capture_call: int | None = None,
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
    with_cpu: bool = False,
    agent_normalize: bool = False,
    agent_recover: bool = False,
    agent_endpoint: str | None = None,
    agent_transport: str | None = None,
    agent_command: str | None = None,
    agent_model: str | None = None,
    agent_timeout: float | None = None,
    agent_output: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(script).resolve()
    output_root = Path(out_dir).resolve()
    effective_capture_call = 0 if capture_call is None else capture_call
    effective_module_pattern = module_pattern
    normalized: dict[str, Any] | None = None
    resolved_framework = framework
    agent_config: AgentConfig | None = None
    cann_ge_mode = find_cann_ge_root(path) is not None
    normalization_path = Path(agent_output).resolve() if agent_output else output_root / "agent_normalization.json"
    recovery_path = output_root / "agent_recovery.json"
    phase = "framework detection"
    old_path = list(sys.path)
    old_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_items = [item for item in (old_pythonpath or "").split(os.pathsep) if item]
    import_roots = [str(path.parent)]
    cann_ge_root = find_cann_ge_root(path)
    if cann_ge_root is not None and str(cann_ge_root) not in import_roots:
        import_roots.append(str(cann_ge_root))
    for import_root in reversed(import_roots):
        if import_root not in sys.path:
            sys.path.insert(0, import_root)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [*import_roots, *(item for item in pythonpath_items if item not in import_roots)]
    )
    try:
        if framework == "auto":
            resolved_framework = detect_framework(path)
            progress_message("INFO", f"framework auto-detected: {resolved_framework} script={path}")
        if agent_normalize or agent_recover:
            phase = "Agent normalization"
            agent_config = AgentConfig.from_env(
                endpoint=agent_endpoint,
                transport=agent_transport,
                command=agent_command,
                model=agent_model,
                timeout=agent_timeout,
            )
            normalized = normalize_script(
                script=path,
                framework=resolved_framework,
                script_args=script_args,
                config=agent_config,
                output=normalization_path,
            )
            progress_message(
                "INFO",
                f"Agent normalization saved: {normalization_path} "
                f"status={normalized['status']} confidence={normalized['diagnostics'].get('confidence', '-')}",
            )
            if normalized["status"] == "needs_user_input":
                progress_message("WARNING", "Agent normalization requires user input; continue with local capture fallback")
            entrypoint = normalized["source"].get("entrypoint", {})
            # CANN GE has a native run_af boundary. Keep its adapter-owned
            # entry selection instead of applying a static Agent guess.
            if capture_call is None and not cann_ge_mode:
                suggested_call = normalized["execution"].get("capture_call", entrypoint.get("capture_call"))
                if isinstance(suggested_call, int) and suggested_call >= 0:
                    effective_capture_call = suggested_call
            if not cann_ge_mode and module_pattern is None and isinstance(entrypoint.get("module_pattern"), str):
                effective_module_pattern = entrypoint["module_pattern"]
        frontend = get_framework_adapter(resolved_framework)
        target = get_backend_adapter(backend)
        if not target.supports(resolved_framework):
            raise ValueError(
                f"backend {backend} does not support framework {resolved_framework} "
                f"(detected from {path}); choose a compatible backend or pass --framework explicitly"
            )
        phase = "framework adapter capture"
        invocation = frontend.capture(
            # The Agent may select an entry among several module calls, but
            # the adapter remains responsible for actually executing it.
            script=path,
            script_args=script_args,
            capture_call=effective_capture_call,
            module_pattern=effective_module_pattern,
        )
        if normalized is not None:
            metadata = dict(invocation.metadata)
            metadata["agent_normalization"] = {
                "path": str(normalization_path),
                "status": normalized["status"],
                "confidence": normalized["diagnostics"].get("confidence"),
                "entrypoint": normalized["source"].get("entrypoint", {}),
            }
            invocation = replace(invocation, metadata=metadata)
        if not target.supports(invocation.framework):
            raise ValueError(f"backend {backend} does not support framework {invocation.framework}")
        phase = "backend execution"
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
            with_cpu=with_cpu,
        )
    except Exception as exc:
        if agent_recover:
            try:
                if agent_config is None:
                    agent_config = AgentConfig.from_env(
                        endpoint=agent_endpoint,
                        transport=agent_transport,
                        command=agent_command,
                        model=agent_model,
                        timeout=agent_timeout,
                    )
                recover_script_error(
                    script=path,
                    framework=resolved_framework,
                    phase=phase,
                    error=exc,
                    normalized=normalized,
                    config=agent_config,
                    output=recovery_path,
                    traceback_text=traceback.format_exc(),
                )
                progress_message("INFO", f"Agent recovery plan saved: {recovery_path}")
            except Exception as recovery_exc:
                progress_message("WARNING", f"Agent recovery request failed: {recovery_exc}")
            raise RuntimeError(f"{exc}; see Agent recovery plan: {recovery_path}") from exc
        raise
    finally:
        sys.path[:] = old_path
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath
