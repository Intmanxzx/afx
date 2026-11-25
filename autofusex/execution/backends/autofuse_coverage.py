"""Runtime-only Autofuse coverage probe for one target compilation.

The probe observes the real AscendC compiler boundary without changing any
installed torch/torch_npu source.  It is deliberately scoped to a single
worker invocation; candidate failures remain ordinary backend failures while
a normally completed scheduler with no candidate is an un-evaluable case.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest.mock import patch
import builtins
from contextlib import ExitStack


class NoAutofuseCoverage(RuntimeError):
    """Raised only after scheduler codegen completed with no AF candidate."""


@dataclass
class Coverage:
    candidates: int = 0
    scheduler_codegen_entered: int = 0
    scheduler_codegen_succeeded: int = 0
    codegen_succeeded: int = 0
    installed: bool = False
    no_coverage: bool = False
    events: list = field(default_factory=list)

    def event(self, name):
        if len(self.events) < 64:
            self.events.append(name)


@contextmanager
def observe_autofuse():
    """Patch the actual exported AF entry points for this worker call.

    ``installed`` is false when a runtime does not expose the AscendC hooks;
    callers must then classify the result as unknown/environmental rather than
    silently treating it as no coverage.
    """
    state = Coverage()
    try:
        # Importing torch_npu._inductor initializes the selected backend.
        # Select AscendC before observation, just as torch.compile will;
        # otherwise fresh workers initialize the default Triton backend.
        import os
        with patch.dict(os.environ, TORCHINDUCTOR_NPU_BACKEND='ascendc'):
            from torch_npu._inductor.ascendc import codegen as af_codegen
            from torch._inductor.scheduler import Scheduler
    except (ImportError, AttributeError):
        yield state
        return

    original_af = getattr(af_codegen, 'codegen_kernel_def', None)
    original_scheduler = getattr(Scheduler, 'codegen', None)
    if not callable(original_af) or not callable(original_scheduler):
        yield state
        return

    state.installed = True
    sources = {}
    try:
        from torch_npu._inductor.ascendc.codegen import _asc_codegen
    except ImportError:
        _asc_codegen = None

    def capture_compile(source, filename, mode, *args, **kwargs):
        code = builtins.compile(source, filename, mode, *args, **kwargs)
        if isinstance(source, str) and 'from autofuse.pyautofuse import' in source:
            sources[code] = source.splitlines()
        return code

    def traced_af(graph, *args, **kwargs):
        state.candidates += 1
        state.event('AF_CANDIDATE')
        # Record before ASCIR construction/schedule/codegen so those failures
        # are retained as AF-path failures rather than coverage skips.
        try:
            from autofusex.execution.core_logs import capture_tail
            with capture_tail(forward=True) as native_log:
                result = original_af(graph, *args, **kwargs)
        except Exception as exc:
            from autofusex.execution.module_attribution import boundary_failure
            fact = boundary_failure(exc, sources)
            if fact:
                import os
                flags = os.environ.get('AUTOFUSE_DFX_FLAGS', '')
                modes = [part.split('=', 1)[1] for part in flags.split(';')
                         if part.strip().lstrip('-').startswith('mlir_codegen_config=')]
                if fact['stage'] == 'codegen' and modes == ['none']:
                    fact['stage'] = 'legacy_codegen'
                elif fact['stage'] == 'codegen':
                    from autofusex.execution.reporting.failure_origin import mlir_failed_line
                    import re
                    text = native_log()
                    if mlir_failed_line(text) or re.search(r'(?:ERROR|error:)[^\n]*MLIR codegen:', text):
                        fact['stage'] = 'mlir'
                fact['graph'] = str(getattr(graph, 'name', ''))[:100]
                exc._afx_module_failure = fact
            raise
        finally:
            sources.clear()
        state.codegen_succeeded += 1
        state.event('AF_CODEGEN_OK')
        return result

    depth = 0

    def traced_scheduler(self, *args, **kwargs):
        nonlocal depth
        outermost = depth == 0
        before = state.candidates
        depth += 1
        state.scheduler_codegen_entered += 1
        state.event('SCHEDULER_CODEGEN_ENTER')
        try:
            result = original_scheduler(self, *args, **kwargs)
        except Exception:
            raise
        else:
            state.scheduler_codegen_succeeded += 1
            state.event('SCHEDULER_CODEGEN_OK')
            # This check is intentionally after the real scheduler returns,
            # not on the first fallback node.  A graph may mix AF and fallback
            # kernels, and a normal scheduler exception is not a skip.
            if outermost and state.installed and state.candidates == before == 0:
                state.no_coverage = True
                state.event('NO_AF_COVERAGE')
                raise NoAutofuseCoverage(
                    'scheduler codegen completed without an Autofuse candidate')
            return result
        finally:
            depth -= 1

    import torch._dynamo.config
    with ExitStack() as stack, torch._dynamo.config.patch(suppress_errors=False), patch.object(af_codegen, 'codegen_kernel_def', traced_af), \
         patch.object(Scheduler, 'codegen', traced_scheduler):
        if _asc_codegen is not None:
            stack.enter_context(patch.object(_asc_codegen, 'compile', capture_compile, create=True))
        yield state
