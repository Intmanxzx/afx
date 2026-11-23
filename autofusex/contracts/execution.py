"""Backend execution settings, evaluation policy and lightweight results."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str
    device: str
    compile_threads: int = 4
    scheduling: str = "combined"
    static_only: bool = False
    batch_id: str = ""
    save_results: bool = True

    def __post_init__(self):
        if type(self.save_results) is not bool:
            raise ValueError("save_results must be boolean")
        if self.compile_threads < 1 or self.scheduling not in {"combined", "staged"}:
            raise ValueError("positive compile budget required; split execution is not validated")


@dataclass(frozen=True)
class EvaluationConfig:
    correctness: bool = True
    performance: bool = False
    warmup_steps: int = 3
    measured_steps: int = 10

    def __post_init__(self):
        if self.warmup_steps < 0 or self.measured_steps < 1:
            raise ValueError("invalid measurement budget")


@dataclass(frozen=True)
class ShapeResult:
    backend: str
    shape_index: int
    correctness: str
    performance: str = "not_requested"
    duration_ns: int = None
    error_summary: str = ""

    def __post_init__(self):
        statuses = {"pass", "fail", "error", "unsupported", "not_requested"}
        if self.correctness not in statuses or self.performance not in statuses:
            raise ValueError("invalid result status")
        if self.shape_index < 0 or (self.duration_ns is not None and self.duration_ns < 0):
            raise ValueError("negative metric/index")
        if len(self.error_summary) > 500:
            raise ValueError("long logs do not belong in lightweight results")
