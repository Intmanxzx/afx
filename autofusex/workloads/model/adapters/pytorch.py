from __future__ import annotations

import copy
import functools
import importlib
import importlib.util
import inspect
import multiprocessing as mp
import os
import pickle
import re
import runpy
import sys
import tempfile
import threading
from queue import Empty, Full
from pathlib import Path
from typing import Any

from autofusex.workloads.model.adapters.base import CapturedInvocation, FrameworkAdapter
from autofusex.workloads.model.adapters.detection import find_cann_ge_root
from autofusex.progress import progress_message


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
    parameters = list(signature.parameters.values())
    # Some Baize releases implement set_timeout without functools.wraps.  The
    # decorated tc_* entry then exposes only (*args, **kwargs), although the
    # framework contract still requires the five arguments below.  Since this
    # adapter is invoking a cann-ge entry, fall back to that known contract
    # instead of silently dropping every argument.
    opaque_wrapper = parameters and all(
        parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
    )
    if opaque_wrapper:
        return (
            tuple(values[name] for name in ("kernel_path", "case_path", "case_name", "soc_version", "device_id")),
            {},
        )
    for parameter in parameters:
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


def _cann_ge_run_af_parts(
    run_af: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, Any, str, Path | None, str | None] | None:
    """Extract the PyTorch model and input specification from a ``run_af`` call."""
    bound_arguments: dict[str, Any] = {}
    try:
        bound_arguments = dict(inspect.signature(run_af).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        pass
    bound_config = bound_arguments.get("config", kwargs.get("config"))
    if bound_config is None and len(args) == 1:
        candidate = args[0]
        if hasattr(candidate, "inputs_param") and hasattr(candidate, "build_graph"):
            bound_config = candidate
    if bound_config is not None:
        inputs_param = getattr(bound_config, "inputs_param", None)
        build_graph = getattr(bound_config, "build_graph", None)
        framework = str(getattr(bound_config, "framework", "tensorflow")).lower()
        case_path = getattr(bound_config, "case_path", None)
        case_name = getattr(bound_config, "case_name", None)
    else:
        inputs_param = bound_arguments.get("inputs_param", args[0] if len(args) > 0 else None)
        build_graph = bound_arguments.get("build_graph", args[1] if len(args) > 1 else None)
        case_path = bound_arguments.get("case_path", args[3] if len(args) > 3 else None)
        case_name = bound_arguments.get("case_name", args[4] if len(args) > 4 else None)
        framework = str(bound_arguments.get("framework", "tensorflow")).lower()
    if framework not in {"pytorch", "torch"}:
        try:
            import torch

            if isinstance(build_graph, torch.nn.Module):
                framework = "pytorch"
        except ImportError:
            pass
    if framework not in {"pytorch", "torch"}:
        return None
    if not isinstance(inputs_param, dict) or build_graph is None:
        return None
    return inputs_param, build_graph, framework, Path(case_path) if case_path else None, case_name


def _cann_ge_run_af_globals(run_af: Any, namespaces: list[dict[str, Any]]) -> dict[str, Any]:
    """Find ``gen_inputs`` behind a direct import or framework re-export."""
    candidates = [run_af]
    try:
        unwrapped = inspect.unwrap(run_af)
    except (ValueError, TypeError):
        unwrapped = run_af
    if unwrapped is not run_af:
        candidates.append(unwrapped)
    for candidate in candidates:
        globalns = getattr(candidate, "__globals__", None)
        if isinstance(globalns, dict) and callable(globalns.get("gen_inputs")):
            return globalns
    for namespace in namespaces:
        if callable(namespace.get("gen_inputs")):
            return namespace
    module_name = str(getattr(unwrapped, "__module__", ""))
    if module_name:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            module = None
        if module is not None and callable(getattr(module, "gen_inputs", None)):
            return vars(module)
    return getattr(unwrapped, "__globals__", {})


def _cann_ge_execution_inputs(generated: Any, input_names: list[str]) -> list[Any]:
    """Normalize Baize inputs to the final iteration used for its output."""
    if isinstance(generated, dict):
        return [generated[name] for name in input_names]
    if not isinstance(generated, (list, tuple)):
        return [generated]
    if not generated:
        return []
    # A single iteration is represented as [input0, input1, ...], while the
    # normal generator returns [[input0, input1, ...], ...].
    first = generated[0]
    if len(generated) == len(input_names) and not isinstance(first, (list, tuple, dict)):
        return list(generated)
    final = generated[-1]
    if isinstance(final, dict):
        return [final[name] for name in input_names]
    if isinstance(final, (list, tuple)):
        return list(final)
    return [final]


def _cann_ge_load_saved_inputs(run_af: Any, case_dir: Path, input_names: list[str]) -> list[Any] | None:
    """Load the framework's persisted inputs when a previous run provided them."""
    input_path = case_dir / "inputs.txt"
    if not input_path.is_file():
        return None
    globalns = _cann_ge_run_af_globals(run_af, [])
    loader = globalns.get("_pickle_load_bf16_safe")
    try:
        if callable(loader):
            saved = loader(str(input_path))
        else:
            import pickle

            with input_path.open("rb") as stream:
                saved = pickle.load(stream)
    except Exception as exc:
        raise RuntimeError(f"failed to load cann-ge inputs from {input_path}: {exc}") from exc
    return _cann_ge_execution_inputs(saved, input_names)


def _cann_ge_to_torch_inputs(values: list[Any], torch: Any) -> tuple[Any, ...]:
    """Convert Baize-generated NumPy inputs to CPU tensors for the debugger."""
    import numpy as np

    converted: list[Any] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            converted.append(value.detach().cpu().clone())
            continue
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "O":
                converted.append(torch.tensor(value.tolist()))
                continue
            try:
                converted.append(torch.from_numpy(value))
            except (TypeError, RuntimeError):
                converted.append(torch.tensor(value))
            continue
        converted.append(torch.as_tensor(value))
    return tuple(converted)


def _cann_ge_model(build_graph: Any, input_names: list[str], torch: Any) -> Any:
    """Return an nn.Module for either a Model instance or a graph callable."""
    if isinstance(build_graph, torch.nn.Module):
        return build_graph

    class CallableGraphModule(torch.nn.Module):
        def __init__(self, callable_graph: Any, names: list[str]) -> None:
            super().__init__()
            self.callable_graph = callable_graph
            self.input_names = tuple(names)
            try:
                parameters = inspect.signature(callable_graph).parameters.values()
            except (TypeError, ValueError):
                parameters = ()
            self.keyword_inputs = bool(
                any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
                and not any(
                    parameter.kind
                    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    for parameter in parameters
                )
            )

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            if self.keyword_inputs:
                values = dict(zip(self.input_names, args))
                values.update(kwargs)
                return self.callable_graph(**values)
            return self.callable_graph(*args, **kwargs)

    return CallableGraphModule(build_graph, input_names)


def _install_cann_ge_run_af_capture(
    namespace: dict[str, Any],
    entry: Any,
    script: Path,
    captured: list[CapturedInvocation],
    capture_call: int,
    case_dir: Path,
    torch: Any,
) -> Any:
    """Patch a cann-ge ``run_af`` import so the debugger owns execution.

    A PyTorch cann-ge case passes the real model and its input specification to
    ``run_af`` before Baize starts benchmark workers.  Capturing at this point
    avoids inheriting the debugger hook into those workers entirely.
    """
    if capture_call != 0:
        return None
    entry_globals = getattr(entry, "__globals__", namespace)
    originals: dict[str, Any] = {}
    candidate_namespaces: list[dict[str, Any]] = [namespace]
    if entry_globals is not namespace:
        candidate_namespaces.append(entry_globals)

    def is_run_af(value: Any) -> bool:
        if not callable(value):
            return False
        try:
            unwrapped = inspect.unwrap(value)
        except (ValueError, TypeError):
            unwrapped = value
        return getattr(unwrapped, "__name__", "") == "run_af"

    for candidate_namespace in candidate_namespaces:
        for name, value in list(candidate_namespace.items()):
            if not is_run_af(value):
                continue

            @functools.wraps(value)
            def capture_run_af(*args: Any, _original: Any = value, **kwargs: Any) -> Any:
                parts = _cann_ge_run_af_parts(_original, args, kwargs)
                if parts is None:
                    return _original(*args, **kwargs)
                inputs_param, build_graph, _, raw_case_path, raw_case_name = parts
                input_names = list(inputs_param)
                generator = _cann_ge_run_af_globals(_original, candidate_namespaces).get("gen_inputs")
                if not callable(generator):
                    raise RuntimeError("cann-ge PyTorch run_af does not expose gen_inputs")
                try:
                    generator_signature = inspect.signature(generator)
                    accepts_framework = "framework" in generator_signature.parameters or any(
                        parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in generator_signature.parameters.values()
                    )
                except (TypeError, ValueError):
                    accepts_framework = True
                generated = (
                    generator(inputs_param, framework="pytorch")
                    if accepts_framework
                    else generator(inputs_param)
                )
                input_values = _cann_ge_execution_inputs(generated, input_names)
                saved_case_dir = (raw_case_path or case_dir) / str(raw_case_name or script.stem)
                saved_values = _cann_ge_load_saved_inputs(_original, saved_case_dir, input_names)
                input_source = "cann-ge gen_inputs"
                if saved_values is not None:
                    input_values = saved_values
                    input_source = str(saved_case_dir / "inputs.txt")
                invocation_model = _cann_ge_model(build_graph, input_names, torch)
                invocation_model.eval()
                execution = _execution_metadata(invocation_model, torch)
                execution["training"] = False
                execution["grad_enabled"] = False
                captured.append(
                    CapturedInvocation(
                        framework="pytorch",
                        model=invocation_model,
                        args=_cann_ge_to_torch_inputs(input_values, torch),
                        kwargs={},
                        name=type(build_graph).__name__,
                        metadata={
                            "capture_call": capture_call,
                            "module": _module_identity(invocation_model),
                            "script": str(script),
                            "execution": execution,
                            "capture_process": "parent",
                            "capture_method": "cann_ge_run_af_boundary",
                            "input_source": input_source,
                            "input_iteration_count": len(generated)
                            if isinstance(generated, (list, tuple))
                            else 1,
                        },
                    )
                )
                progress_message(
                    "INFO",
                    "cann-ge PyTorch model and inputs captured at run_af boundary; "
                    "Baize benchmark execution is bypassed",
                )
                raise _CaptureComplete

            originals[f"{id(candidate_namespace)}:{name}"] = (candidate_namespace, name, value)
            candidate_namespace[name] = capture_run_af

    if not originals:
        return None

    def restore() -> None:
        for namespace, name, value in originals.values():
            namespace[name] = value

    return restore


def _run_cann_ge_script(path: Path) -> dict[str, Any]:
    """Execute a cann-ge case as an importable module for subprocess pickling.

    Baize may serialize a model created by the case function.  Running the
    source with ``runpy(..., run_name="__main__")`` makes classes defined by
    the source appear as ``__main__.Model``, which the Baize subprocess cannot
    import.  The file stem is also the module name that a child process can
    import from the script directory inherited through ``PYTHONPATH``.
    """
    module_name = path.stem
    if not module_name.isidentifier():
        raise RuntimeError(
            f"cann-ge script {path} must have a valid Python module filename for subprocess execution"
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import cann-ge script {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return vars(module)


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
        capture_pid = os.getpid()
        child_capture_queue: Any = None

        def intercepted_call(module, *args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal eligible_call
            depth = getattr(state, "depth", 0)
            if os.getpid() != capture_pid:
                # Baize uses fork for some torch benchmarks.  The hook is
                # inherited by the child, but _CaptureComplete must never
                # escape into Baize's worker because it only catches Exception.
                if depth == 0 and child_capture_queue is not None and not getattr(state, "sent", False):
                    original = _original_module(module)
                    identity = _module_identity(original)
                    if pattern is None or pattern.search(identity) or pattern.search(type(original).__name__):
                        try:
                            child_capture_queue.put_nowait(pickle.dumps(
                                CapturedInvocation(
                                    framework=self.name,
                                    model=original,
                                    args=tuple(_snapshot(args, torch)),
                                    kwargs=dict(_snapshot(kwargs, torch)),
                                    name=type(original).__name__,
                                    metadata={
                                        "capture_call": capture_call,
                                        "module": identity,
                                        "script": str(path),
                                        "execution": _execution_metadata(original, torch),
                                        "capture_process": "fork_child",
                                    },
                                )
                            ))
                            state.sent = True
                        except (Full, OSError, ValueError):
                            pass
                state.depth = depth + 1
                try:
                    return original_call(module, *args, **kwargs)
                finally:
                    state.depth = depth
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
        cann_ge_root = find_cann_ge_root(path)
        import_roots = [str(path.parent)]
        if cann_ge_root is not None:
            import_roots.insert(0, str(cann_ge_root))
            # This queue is inherited by Linux fork workers.  It allows a
            # framework-owned worker to return the captured model invocation
            # without aborting that worker's benchmark execution.
            child_capture_queue = mp.Queue(maxsize=1)
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
                if cann_ge_root is not None:
                    namespace = _run_cann_ge_script(path)
                else:
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
                    with tempfile.TemporaryDirectory(prefix="autofusex_cann_ge_") as temp_root:
                        case_dir = Path(temp_root) / "case"
                        case_dir.mkdir(parents=True)
                        entry_args, entry_kwargs = _cann_ge_entry_arguments(entry, path, cann_ge_root, case_dir)
                        run_af_restore = _install_cann_ge_run_af_capture(
                            namespace,
                            entry,
                            path,
                            captured,
                            capture_call,
                            case_dir,
                            torch,
                        )
                        if run_af_restore is not None:
                            progress_message(
                                "INFO",
                                "cann-ge PyTorch run_af interception enabled",
                            )
                        try:
                            entry(*entry_args, **entry_kwargs)
                        except _CaptureComplete:
                            pass
                        finally:
                            if run_af_restore is not None:
                                run_af_restore()
                    if not captured and child_capture_queue is not None:
                        try:
                            # Serialize tensors by value: Torch's queue reducers
                            # otherwise need the worker's FD server after it exits.
                            captured.append(pickle.loads(child_capture_queue.get(timeout=2)))
                        except (Empty, EOFError, OSError):
                            pass
        finally:
            torch.nn.Module.__call__ = original_call
            sys.argv = old_argv
            sys.path[:] = old_path
            os.chdir(old_cwd)
            if child_capture_queue is not None:
                child_capture_queue.close()

        if not captured:
            selector = f" matching {module_pattern!r}" if module_pattern else ""
            raise RuntimeError(
                f"did not observe PyTorch module call {capture_call}{selector} while executing {path}"
            )
        return captured[0]
