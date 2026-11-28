"""Wire models shared by FastAPI validation and its generated OpenAPI schema."""
from typing import Any, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field


class Payload(BaseModel):
    model_config = ConfigDict(extra='allow', strict=True)


class RequestKey(Payload):
    request_key: str = Field(min_length=1, max_length=128, description='Idempotency key: reuse only with the identical request.')


class Claim(RequestKey):
    agent_id: str = Field(min_length=1, max_length=64)
    issue_id: Optional[str] = None
    dimension: Optional[Literal['function', 'accuracy', 'performance']] = None
    backend: Optional[str] = None


class Lease(RequestKey):
    lease_id: str


class Verify(RequestKey):
    lease_id: Optional[str] = None
    fix_version: str = Field(min_length=1, max_length=64)
    fix_sha256: str = Field(pattern='^[0-9a-f]{64}$')
    fix_ref: Optional[str] = Field(default=None, max_length=512)


class IssuePins(RequestKey):
    case_keys: list[str] = Field(min_length=1, max_length=200)
    lease_id: Optional[str] = None
    reason: Optional[str] = Field(default=None, max_length=500)


class Knowledge(Payload):
    kind: Literal['bugfix_function', 'bugfix_accuracy', 'perf_tuning', 'fusion_pattern', 'capability', 'best_practice', 'pitfall']
    title: str = Field(min_length=1, max_length=200)
    symptom: str = Field(min_length=1, max_length=8000)
    root_cause: str = Field(min_length=1, max_length=8000)
    solution: str = Field(min_length=1, max_length=8000)
    graph_pattern: Optional[str] = Field(default=None, max_length=1000)
    applicability: Optional[str] = Field(default=None, max_length=4000)
    version_range: Optional[str] = Field(default=None, max_length=128)
    fix_ref: Optional[str] = Field(default=None, max_length=512)
    keywords: Optional[list[str]] = None
    ops: Optional[list[str]] = None
    dtypes: Optional[list[str]] = None
    backends: Optional[list[str]] = None
    body_md: Optional[str] = None


class SubmitKnowledge(Lease):
    knowledge: Knowledge


class ProposeKnowledge(RequestKey):
    knowledge: Knowledge
    measured: bool = False
    agent_id: Optional[str] = None


class KnowledgeStatus(RequestKey):
    status: Literal['draft', 'published', 'deprecated']


class CloseIssue(RequestKey):
    reason: str


class MergeIssue(RequestKey):
    into: str


class Installed(Payload):
    version: str
    sha256: str = Field(pattern='^[0-9a-f]{64}$')


class ClaimRetests(RequestKey):
    machine_id: str = Field(min_length=1, max_length=128)
    npu_arch: str = Field(min_length=1, max_length=32)
    installed: dict[str, Installed]
    limit: int = Field(default=5, ge=1, le=20)
    coverage_environment: Optional[dict[str, Any]] = None


class CompleteRetest(Lease):
    installed: dict[str, Installed]
    exec_uuids: list[str]
    terminal_reason: Optional[str] = Field(default=None, max_length=500)
    coverage_environment: Optional[dict[str, Any]] = None
    coverage_evidence: Optional[list[dict[str, Any]]] = None


class ManualRetest(RequestKey):
    force: bool = False
    case_keys: list[str] = Field(min_length=1, max_length=200000)
    backends: list[str] = Field(min_length=1, max_length=4)
    npu_arch: Optional[str] = None
    user_ids: Optional[list[str]] = None
    repetitions: int = Field(default=1, ge=1, le=5)


class Pins(RequestKey):
    scope: Literal['global', 'feature']
    scope_value: str = ''
    case_keys: list[str] = Field(min_length=1, max_length=200000)
    reason: Optional[str] = None


class ClientUpdate(RequestKey):
    client_ids: list[int]
    version: Optional[str] = None


class ClientControl(RequestKey):
    client_id: int
    state: Literal['running', 'stopped', 'restart']


class Drafts(RequestKey):
    x: str
    y: str
    npu_arch: Optional[str] = None


class Heartbeat(Payload):
    machine_id: str = Field(min_length=1, max_length=128)
    hostname: Optional[str] = None
    client_home: Optional[str] = None
    npu_arch: Optional[str] = None
    afx_version: Optional[str] = None
    devices: Optional[list[str]] = None
    backend_versions: Optional[dict[str, Optional[str]]] = None
    counters: Optional[dict[str, int]] = None
    runtime: Optional[list[dict[str, Any]]] = None
    windows: Optional[list[dict[str, Any]]] = None
    pending_uploads: Optional[int] = Field(default=None, ge=0)
    busy: bool = False
    daemon_state: Optional[Literal['running', 'stopped']] = None


class Client(Payload):
    machine_id: str
    hostname: Optional[str] = None
    npu_arch: Optional[str] = None
    afx_version: Optional[str] = None


class Run(Payload):
    run_uuid: str
    kind: Literal['fuzz', 'retest', 'verify', 'regress', 'import']


class Case(Payload):
    case_key: str
    case_source: Literal['nnsmith', 'baize_tf', 'baize_torch', 'model']
    shape_count: int = Field(ge=1, le=8)
    graph_fingerprint: Optional[str] = None
    is_dynamic: bool = False
    node_count: Optional[int] = None
    op_types: Optional[list[str]] = None
    dtypes: Optional[list[str]] = None
    max_dim: Optional[int] = None
    max_numel: Optional[int] = None
    external_ref: Optional[str] = None
    features: Optional[dict[str, Any]] = None
    metrics: Optional[dict[str, Any]] = None
    spec: Any = None


class Execution(Payload):
    capture_version: int = Field(default=1, ge=1)
    exec_uuid: str
    case: Case
    backend: str
    env: dict[str, Optional[str]]
    e2e_ns: int = Field(ge=0)
    created_at: int = Field(ge=1)
    npu_arch: Optional[str] = None
    function_status: Optional[str] = None
    accuracy_status: Optional[str] = None
    accuracy_ref: Optional[str] = None
    fail_stage: Optional[str] = None
    max_abs_err: Optional[float] = None
    max_rel_err: Optional[float] = None
    compile_ns: Optional[int] = None
    kernel_ns: Optional[int] = None
    log_summary: Optional[str] = None
    log: Optional[str] = None
    detail: Optional[dict[str, Any]] = None
    shapes: Optional[list[dict[str, Any]]] = None


class Upload(RequestKey):
    protocol_version: int = Field(default=1, ge=1)
    client: Client
    run: Run
    items: list[Execution] = Field(min_length=1, max_length=500)


class Configuration(Payload):
    config: Optional[dict[str, Any]] = Field(default=None, description='Full configuration snapshot; null restores inheritance. Validated by the shared client configuration validator.')


class Settings(Payload):
    changes: dict[str, Any]
    base_revisions: dict[str, int]

class ClassificationRules(Payload):
    rules: list[dict[str, Any]]
    base_revision: Optional[int] = None

class ClassificationRefresh(Payload):
    rules: Optional[list[dict[str, Any]]] = None
    apply: bool = False
    batch_id: Optional[str] = None

class ClassificationJob(RequestKey):
    rules: Optional[list[dict[str, Any]]] = None
    preview_id: Optional[str] = None


class Application(Payload):
    user_id: str


class Receipt(Payload):
    receipt: str


class Review(Payload):
    id: int
    action: Literal['approve', 'reject']
    role: Literal['developer', 'admin'] = 'developer'


class Manage(Payload):
    user_id: str
    action: Literal['enable', 'disable', 'role', 'reset-token']
    role: Optional[Literal['developer', 'admin']] = None


class Success(BaseModel):
    schema_version: Literal[1] = 1
    ok: Literal[True] = True
    data: Any = Field(description='Operation-specific business result, including extensible database projections.')


class ErrorDetail(BaseModel):
    code: str
    message: str


class Error(BaseModel):
    schema_version: Literal[1] = 1
    ok: Literal[False] = False
    error: ErrorDetail


class SourceBundle(Payload):
    case: dict[str, Any]
    files: dict[str, str]


class SourceJob(RequestKey):
    spec: dict[str, Any]


class SourceClaim(Payload):
    machine_id: str = Field(min_length=1, max_length=128)


class SourceReceipt(Payload):
    lease_id: str
    result: dict[str, Any]


class SourceResult(Payload):
    spec: dict[str, Any]
    result: dict[str, Any]


class SourceArtifact(Payload):
    lease_id: str
    path: str
    data: str


class IssueTreatment(RequestKey):
    treatment: Literal['ignore', 'low', 'normal', 'high']
    reason: str = Field(default='', max_length=500)


class IssueStatus(RequestKey):
    status: Literal['open', 'claimed', 'verifying', 'regressing', 'summarizing', 'closed']
    expected_status: Literal['open', 'claimed', 'verifying', 'regressing', 'summarizing', 'resolved', 'closed']
    reason: str = Field(min_length=1, max_length=500)
