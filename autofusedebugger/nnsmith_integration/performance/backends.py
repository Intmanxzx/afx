"""Backend selection is applied before the worker Python interpreter starts."""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional


@dataclass(frozen=True)
class BackendSpec:
    name: str
    npu_backend: Optional[str] = None

    def environment(self, inherited: Mapping[str, str], cache_dir: Path) -> dict[str, str]:
        env = dict(inherited)
        env.pop("TORCHINDUCTOR_NPU_BACKEND", None)
        if self.npu_backend is not None:
            env["TORCHINDUCTOR_NPU_BACKEND"] = self.npu_backend
        env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        env["TRITON_CACHE_DIR"] = str(cache_dir / "triton")
        # Debug hooks/dumps are excluded from measured execution.
        for key in ("TORCH_COMPILE_DEBUG", "AUTOFUSE_DFX_FLAGS"):
            env.pop(key, None)
        return env

    def compile_kwargs(self) -> dict:
        if self.npu_backend is None:
            return {"fullgraph": True}
        return {"fullgraph": True, "backend": "inductor", "options": {"npu_backend": self.npu_backend}}


BACKENDS = {
    "ascendc": BackendSpec("ascendc", "ascendc"),
    # Deliberately use the NPU default: no environment override and no option.
    "triton-ascend": BackendSpec("triton-ascend"),
}
