from __future__ import annotations

import copy
import re
import runpy
import sys
import threading
from pathlib import Path
from typing import Any

from autofusedebugger.adapters.base import CapturedInvocation, FrameworkAdapter


class _CaptureComplete(BaseException):
    pass


def _snapshot(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_snapshot(item, torch) for item in value)
    if isinstance(value, list):
        return [_snapshot(item, torch) for item in value]
    if isinstance(value, dict):
        return {key: _snapshot(item, torch) for key, item in value.items()}
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def _module_identity(module: Any) -> str:
    module_type = type(module)
    return f"{module_type.__module__}.{module_type.__qualname__}"


def _original_module(module: Any) -> Any:
    return getattr(module, "_orig_mod", module)


class PyTorchAdapter(FrameworkAdapter):
    name = "pytorch"
    implemented = True

    def capture(
        self,
        *,
        script: str | Path,
        script_args: tuple[str, ...] = (),
        capture_call: int = 0,
        module_pattern: str | None = None,
    ) -> CapturedInvocation:
        if capture_call < 0:
            raise ValueError("capture_call must be >= 0")
        path = Path(script).resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))

        import torch

        pattern = re.compile(module_pattern) if module_pattern else None
        original_call = torch.nn.Module.__call__
        state = threading.local()
        captured: list[CapturedInvocation] = []
        eligible_call = 0

        def intercepted_call(module, *args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal eligible_call
            depth = getattr(state, "depth", 0)
            if depth == 0:
                original = _original_module(module)
                identity = _module_identity(original)
                if pattern is None or pattern.search(identity) or pattern.search(type(original).__name__):
                    current_call = eligible_call
                    eligible_call += 1
                    if current_call == capture_call:
                        captured.append(
                            CapturedInvocation(
                                framework=self.name,
                                model=original,
                                args=tuple(_snapshot(args, torch)),
                                kwargs=dict(_snapshot(kwargs, torch)),
                                name=type(original).__name__,
                                metadata={
                                    "capture_call": current_call,
                                    "module": identity,
                                    "script": str(path),
                                },
                            )
                        )
                        raise _CaptureComplete
            state.depth = depth + 1
            try:
                return original_call(module, *args, **kwargs)
            finally:
                state.depth = depth

        old_argv = sys.argv
        old_path = list(sys.path)
        torch.nn.Module.__call__ = intercepted_call
        sys.argv = [str(path), *script_args]
        sys.path.insert(0, str(path.parent))
        try:
            try:
                runpy.run_path(str(path), run_name="__main__")
            except _CaptureComplete:
                pass
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise RuntimeError(f"model script exited with status {exc.code}") from exc
        finally:
            torch.nn.Module.__call__ = original_call
            sys.argv = old_argv
            sys.path[:] = old_path

        if not captured:
            selector = f" matching {module_pattern!r}" if module_pattern else ""
            raise RuntimeError(
                f"did not observe PyTorch module call {capture_call}{selector} while executing {path}"
            )
        return captured[0]
