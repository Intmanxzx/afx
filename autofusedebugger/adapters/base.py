from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CapturedInvocation:
    framework: str
    model: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FrameworkAdapter(ABC):
    name: str
    implemented: bool = False

    @abstractmethod
    def capture(
        self,
        *,
        script: str | Path,
        script_args: tuple[str, ...] = (),
        capture_call: int = 0,
        module_pattern: str | None = None,
    ) -> CapturedInvocation:
        raise NotImplementedError


class BackendAdapter(ABC):
    name: str
    supported_frameworks: tuple[str, ...]

    def supports(self, framework: str) -> bool:
        return framework in self.supported_frameworks

    @abstractmethod
    def run(self, invocation: CapturedInvocation, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError
