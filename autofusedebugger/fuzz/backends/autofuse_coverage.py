"""Runtime-only Autofuse coverage probe for one target compilation.

The probe observes the real AscendC compiler boundary without changing any
installed torch/torch_npu source.  It is deliberately scoped to a single
worker invocation; candidate failures remain ordinary backend failures while
a normally completed scheduler with no candidate is an un-evaluable case.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest.mock import patch


class NoAutofuseCoverage(RuntimeError):
    """Raised only after scheduler codegen completed with no AF candidate."""


@dataclass
class Coverage:
    candidates: int = 0
    schedule_entered: int = 0
    schedule_succeeded: int = 0
    codegen_succeeded: int = 0
    installed: bool = False
    no_coverage: bool = False
    events: list = field(default_factory=list)

    def event(self, name):
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
        from torch._inductor.scheduler import Scheduler
        from torch_npu._inductor.ascendc import codegen as af_codegen
    except (ImportError, AttributeError):
        yield state
        return

    original_af = getattr(af_codegen, 'codegen_kernel_def', None)
    original_scheduler = getattr(Scheduler, 'codegen', None)
    if not callable(original_af) or not callable(original_scheduler):
        yield state
        return

    state.installed = True

    def traced_af(graph, *args, **kwargs):
        state.candidates += 1
        state.event('AF_CANDIDATE')
        # Record before ASCIR construction/schedule/codegen so those failures
        # are retained as AF-path failures rather than coverage skips.
        result = original_af(graph, *args, **kwargs)
        state.codegen_succeeded += 1
        state.event('AF_CODEGEN_OK')
        return result

    depth = 0

    def traced_scheduler(self, *args, **kwargs):
        nonlocal depth
        outermost = depth == 0
        before = state.candidates
        depth += 1
        state.schedule_entered += 1
        state.event('AF_SCHEDULE_ENTER')
        try:
            result = original_scheduler(self, *args, **kwargs)
        except Exception:
            raise
        else:
            state.schedule_succeeded += 1
            state.event('AF_SCHEDULE_OK')
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

    with patch.object(af_codegen, 'codegen_kernel_def', traced_af), \
         patch.object(Scheduler, 'codegen', traced_scheduler):
        yield state
