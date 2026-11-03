"""JSON-friendly contracts; no Torch, NPU or NNSmith imports."""
from dataclasses import dataclass, field
import math
from typing import Optional


@dataclass(frozen=True)
class ProfileConfig:
    warmup_steps: int = 3
    steps: int = 10
    timeout_s: float = 600
    keep_artifacts: bool = False

    def __post_init__(self):
        if self.warmup_steps < 1 or self.steps < 1 or not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("warmup_steps, profile steps and timeout must be positive")


@dataclass
class BackendResult:
    backend: str
    status: str = "error"
    correctness: str = "not_checked"
    fingerprint: Optional[str] = None
    mean_kernel_us: Optional[float] = None
    total_kernel_us: Optional[float] = None
    total_kernel_count: Optional[int] = None
    mean_kernel_count: Optional[float] = None
    duration_s: Optional[float] = None
    measured_steps: int = 0
    csv_path: Optional[str] = None
    log_path: Optional[str] = None
    error_stage: Optional[str] = None
    error: Optional[str] = None
    correctness_error: Optional[str] = None
    cleanup_error: Optional[str] = None
    op_statistics: list[dict] = field(default_factory=list)
    resolved_backend: Optional[str] = None
    runtime: dict = field(default_factory=dict)
