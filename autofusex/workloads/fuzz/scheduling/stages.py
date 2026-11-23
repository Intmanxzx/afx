"""Common stage contracts. 'combined' is a backend capability, not a scheduler."""
from dataclasses import dataclass, field
from enum import Enum
import uuid


class Kind(str, Enum):
    GENERATE = 'generate'
    REFERENCE = 'reference'
    COMBINED = 'first_call_combined'
    COMPARE = 'compare'
    RECORD = 'record'


@dataclass(frozen=True)
class Requirements:
    cpu_tokens: int = 1
    memory_bytes: int = 0
    cache_bytes: int = 0
    devices: tuple = ()
    device_bytes: int = 0


@dataclass
class Case:
    identity: str
    seed: str
    admitted: dict
    tasks: list
    directory: str
    stages: dict = field(default_factory=dict)
    bundle: dict = None
    reference: dict = None
    skip_reason: str = None
    skip_finalized: bool = False


@dataclass
class BackendSession:
    identity: str
    task: dict
    device: str = None
    uuid: str = None


@dataclass
class StageTask:
    kind: Kind
    case_id: str
    dependencies: tuple = ()
    session_id: str = None
    attempt_id: str = None
    requirements: Requirements = field(default_factory=Requirements)
    identity: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: str = 'waiting_dependencies'
    ready_ns: int = None
    future: object = None
    output: dict = None
    facts: dict = field(default_factory=dict)

    def ready(self, stages, now):
        if self.state == 'waiting_dependencies' and all(stages[d].state == 'succeeded' for d in self.dependencies):
            self.state = 'ready'
            self.ready_ns = now
        return self.state == 'ready'
