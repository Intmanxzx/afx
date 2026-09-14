from __future__ import annotations

import copy
import inspect
import os
import re
import runpy
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from autofusedebugger.adapters.base import CapturedInvocation, FrameworkAdapter
from autofusedebugger.progress import progress_message


class _CaptureComplete(BaseException):
    pass


def _snapshot(value: Any, torch: Any) -> Any:
    from torch.utils import _pytree

    def copy_leaf(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return item.detach().cpu().clone()
        try:
            return copy.deepcopy(item)
        except Exception:
            return item

    return _pytree.tree_map(copy_leaf, value)


def _module_identity(module: Any) -> str:
    module_type = type(module)
    return f"{module_type.__module__}.{module_type.__qualname__}"


def _original_module(module: Any) -> Any:
    return getattr(module, "_orig_mod", module)


def _execution_metadata(module: Any, torch: Any) -> dict[str, Any]:
    autocast: dict[str, dict[str, Any]] = {}
    for device_type in ("cpu", "cuda", "npu"):
        try:
            enabled = bool(torch.is_autocast_enabled(device_type))
        except Exception:
            continue
        if not enabled:
            continue
        entry: dict[str, Any] = {"enabled": True}
        try:
            entry["dtype"] = str(torch.get_autocast_dtype(device_type))
        except Exception:
            pass
        try:
            entry["cache_enabled"] = bool(torch.is_autocast_cache_enabled())
        except Exception:
            pass
        autocast[device_type] = entry
    return {
        "training": bool(getattr(module, "training", False)),
        "grad_enabled": bool(torch.is_grad_enabled()),
        "autocast": autocast,
    }


def _find_cann_ge_root(script: Path) -> Path | None:
    """Find the root of the smoke-testing/cann-ge package layout, if present."""
    for candidate in (script.parent, *script.parents):
        if (candidate / "libs" / "__init__.py").is_file() and (candidate / "feature").is_dir():
            return candidate
    return None


def _cann_ge_entry_arguments(
    entry: Any, script: Path, root: Path, case_dir: Path
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Build the arguments expected by a cann-ge ``tc_*`` case function.

    The framework normally supplies these values from ``xfusion_fmk.py``.  We
    only need safe placeholders here: the module call interception stops the
    case before its own result/profiling checks run.
    """
    values = {
        "kernel_path": str(root / "kernel_meta*" / "kernel_meta"),
        "case_path": str(case_dir),
        "case_name": script.stem,
        "soc_version": os.environ.get("SOC_VERSION", "Ascend910B1"),
        "device_id": os.environ.get("ASCEND_DEVICE_ID", os.environ.get("DEVICE_ID", "0")),
    }
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    signature = inspect.signature(entry)
    for parameter in signature.parameters.values():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        value = values.get(parameter.name)
        if value is None:
            if parameter.default is not inspect.Parameter.empty:
                continue
            raise RuntimeError(
                f"cannot invoke cann-ge entry {entry.__name__!r}: unsupported required parameter "
                f"{parameter.name!r}; pass a create_case() adapter for this script"
            )
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            keyword[parameter.name] = value
        else:
            positional.append(value)
    return tuple(positional), keyword


def _cann_ge_entries(namespace: dict[str, Any]) -> list[tuple[str, Any]]:
    return sorted(
        (name, value)
        for name, value in namespace.items()
        if name.startswith("tc_") and callable(value) and not name.startswith("__")
    )


def _select_cann_ge_entry(namespace: dict[str, Any], script: Path) -> tuple[str, Any] | None:
    entries = _cann_ge_entries(namespace)
    preferred = next(((name, entry) for name, entry in entries if name == script.stem), None)
    if preferred is not None:
        return preferred
    if len(entries) == 1:
        return entries[0]
    if len(entries) > 1:
        names = ", ".join(name for name, _ in entries)
        raise RuntimeError(
            f"did not observe a PyTorch module call while executing {script}; cann-ge script has "
            f"multiple tc_* entries ({names}); use one case per script or provide create_case()"
        )
    return None


def _prepare_cann_ge_namespace(namespace: dict[str, Any], entry: Any) -> None:
    """Disable framework-only version gates while capturing a model call."""
    entry_globals = getattr(entry, "__globals__", namespace)
    if callable(entry_globals.get("check_torch_version_for_inductor")):
        entry_globals["check_torch_version_for_inductor"] = lambda: True


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
                                    "execution": _execution_metadata(original, torch),
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
        old_cwd = Path.cwd()
        cann_ge_root = _find_cann_ge_root(path)
        import_roots = [str(path.parent)]
        if cann_ge_root is not None:
            import_roots.insert(0, str(cann_ge_root))
        torch.nn.Module.__call__ = intercepted_call
        sys.argv = [str(path), *script_args]
        for import_root in reversed(import_roots):
            if import_root not in sys.path:
                sys.path.insert(0, import_root)
        if cann_ge_root is not None:
            os.chdir(cann_ge_root)
        try:
            namespace: dict[str, Any] = {}
            try:
                namespace = runpy.run_path(str(path), run_name="__main__")
            except _CaptureComplete:
                pass
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise RuntimeError(f"model script exited with status {exc.code}") from exc
            if not captured and cann_ge_root is not None:
                selected_entry = _select_cann_ge_entry(namespace, path)
                if selected_entry is not None:
                    entry_name, entry = selected_entry
                    _prepare_cann_ge_namespace(namespace, entry)
                    progress_message("INFO", f"cann-ge case entry detected: {entry_name}")
                    with tempfile.TemporaryDirectory(prefix="autofusedebugger_cann_ge_") as temp_root:
                        case_dir = Path(temp_root) / "case"
                        case_dir.mkdir(parents=True)
                        entry_args, entry_kwargs = _cann_ge_entry_arguments(entry, path, cann_ge_root, case_dir)
                        try:
                            entry(*entry_args, **entry_kwargs)
                        except _CaptureComplete:
                            pass
        finally:
            torch.nn.Module.__call__ = original_call
            sys.argv = old_argv
            sys.path[:] = old_path
            os.chdir(old_cwd)

        if not captured:
            selector = f" matching {module_pattern!r}" if module_pattern else ""
            raise RuntimeError(
                f"did not observe PyTorch module call {capture_call}{selector} while executing {path}"
            )
        return captured[0]
