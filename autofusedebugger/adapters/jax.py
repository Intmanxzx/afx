from __future__ import annotations

from pathlib import Path

from autofusedebugger.adapters.base import CapturedInvocation, FrameworkAdapter


class JaxAdapter(FrameworkAdapter):
    name = "jax"

    def capture(
        self,
        *,
        script: str | Path,
        script_args: tuple[str, ...] = (),
        capture_call: int = 0,
        module_pattern: str | None = None,
    ) -> CapturedInvocation:
        del script, script_args, capture_call, module_pattern
        raise NotImplementedError("JaxAdapter is reserved but not implemented")
