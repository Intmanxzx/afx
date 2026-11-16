"""Immutable case and execution contracts for versioned seed protocols.

Seed identities select frozen generation semantics without external state.
All collections are tuples and tensor payloads are bytes, not mutable tensors.
"""
from dataclasses import dataclass
import hashlib
import json
import re

CONTRACT_VERSION = 1
RESULT_SCHEMA_VERSION = 2
SEED_PROTOCOL_VERSION = 1


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Seed:
    """One self-contained decimal identity; high bits select immutable semantics."""
    value: int

    def __post_init__(self):
        if type(self.value) is not int or self.value < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.version not in {11, 12}:
            raise ValueError(f"unsupported generation version {self.version}")
        from .generation.sampling import decode
        decode(self)

    @property
    def version(self):
        return self.value // 2**160 + 1

    @property
    def entropy(self):
        return self.value % 2**160

    @property
    def dynamic(self):
        return self.version == 12

    def __str__(self):
        return str(self.value)

    @property
    def short(self):
        alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        number = self.entropy
        digits = ''
        while number:
            number, digit = divmod(number, 36)
            digits = alphabet[digit] + digits
        prefix = 'F' if self.dynamic else 'E'
        return prefix + (digits or '0').zfill(26)

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if type(value) is int:
            return cls(value)
        if isinstance(value, str) and re.fullmatch(r'[EF][0-9A-Z]{26}', value):
            entropy = int(value[1:], 36)
            if entropy >= 2**160:
                raise ValueError('seed entropy overflow')
            return cls((11 if value[0] == 'F' else 10) * 2**160 + entropy)
        if not isinstance(value, str) or not re.fullmatch(r"0|[1-9][0-9]*", value):
            raise ValueError("seed must be canonical decimal digits; no profiles or extra parameters")
        return cls(int(value))


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple
    stride: tuple  # element strides, not bytes
    storage_offset: int = 0
    alias_group: int = 0

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or not isinstance(self.stride, tuple):
            raise TypeError("shape and stride must be immutable tuples")
        if len(self.shape) != len(self.stride) or any(type(v) is not int or v < 0 for v in self.shape + self.stride):
            raise ValueError("invalid tensor dimensions/strides")
        if not isinstance(self.name, str) or not self.name or not isinstance(self.dtype, str) or not self.dtype or any(type(v) is not int or v < 0 for v in (self.storage_offset, self.alias_group)):
            raise ValueError("invalid tensor identity/storage")


@dataclass(frozen=True)
class CaseSpec:
    seed: Seed
    graph_json: str
    inputs: tuple
    weights: tuple
    shape_samples: tuple = ()
    output_samples: tuple = ()

    def __post_init__(self):
        if not isinstance(self.seed, Seed):
            raise TypeError("seed must be a Seed")
        graph = json.loads(self.graph_json)
        if canonical_json(graph) != self.graph_json:
            raise ValueError("graph_json must use canonical encoding")
        for specs in (self.inputs, self.weights, *self.shape_samples, *self.output_samples):
            if not isinstance(specs, tuple) or not all(isinstance(v, TensorSpec) for v in specs):
                raise TypeError("signatures must be tuples of TensorSpec")
            if len({v.name for v in specs}) != len(specs):
                raise ValueError("duplicate tensor name")
        if not isinstance(self.shape_samples, tuple) or not isinstance(self.output_samples, tuple):
            raise TypeError("sample signatures must be tuples")
        if self.output_samples and len(self.output_samples) != len(self.shape_samples):
            raise ValueError("input/output sample counts differ")


@dataclass(frozen=True)
class CaseBundle:
    spec: CaseSpec
    # Canonical little-endian storage bytes; adapters create disposable device tensors.
    storages: tuple

    def __post_init__(self):
        if not isinstance(self.spec, CaseSpec) or not isinstance(self.storages, tuple) or not all(type(v) is bytes for v in self.storages):
            raise TypeError("CaseBundle requires immutable spec and byte storages")
        # Canonical storage mirrors the dtype coverage of the pre-refactor
        # NNSmith generator.  Complex values use their native byte widths;
        # bfloat16 is stored as its 16-bit IEEE high half.
        widths = {
            "bool": 1, "uint8": 1, "int8": 1,
            "uint16": 2, "uint32": 4, "int16": 2, "int32": 4, "int64": 8,
            "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
            "complex64": 8, "complex128": 16,
        }
        for tensor in (*self.spec.weights, *self.spec.inputs, *(t for s in self.spec.shape_samples for t in s)):
            if tensor.dtype not in widths:
                raise ValueError("unsupported canonical storage dtype")
            if tensor.alias_group >= len(self.storages):
                raise ValueError("tensor references missing storage")
            raw = self.storages[tensor.alias_group]
            if len(raw) % widths[tensor.dtype]:
                raise ValueError("misaligned canonical storage")
            last = tensor.storage_offset + sum((dim - 1) * stride for dim, stride in zip(tensor.shape, tensor.stride))
            required = 0 if 0 in tensor.shape else (last + 1) * widths[tensor.dtype]
            if required > len(raw):
                raise ValueError("tensor view exceeds storage bounds")


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
