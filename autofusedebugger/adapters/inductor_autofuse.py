from __future__ import annotations

from typing import Any

from autofusedebugger.adapters.base import BackendAdapter, CapturedInvocation


class InductorAutofuseBackend(BackendAdapter):
    name = "inductor_autofuse"
    supported_frameworks = ("pytorch",)

    def run(self, invocation: CapturedInvocation, **kwargs: Any) -> dict[str, Any]:
        if not self.supports(invocation.framework):
            raise ValueError(f"{self.name} does not support framework {invocation.framework}")

        from autofusedebugger.torch_model import TorchCase, run_torch_case

        case = TorchCase(
            model=invocation.model,
            inputs=invocation.args,
            kwargs=invocation.kwargs,
            name=str(kwargs.pop("name", None) or invocation.name),
            fullgraph=bool(kwargs.pop("fullgraph", False)),
            dynamic=bool(kwargs.pop("dynamic", False)),
            training=invocation.metadata.get("execution", {}).get("training"),
            grad_enabled=bool(invocation.metadata.get("execution", {}).get("grad_enabled", True)),
            autocast=dict(invocation.metadata.get("execution", {}).get("autocast", {})),
            rewrite_device_hints=bool(kwargs.pop("rewrite_device_hints", False)),
        )
        script = invocation.metadata.get("script")
        if not script:
            raise ValueError("captured invocation does not contain its source script")
        return run_torch_case(
            case=case,
            script=script,
            invocation_metadata=invocation.metadata,
            **kwargs,
        )
