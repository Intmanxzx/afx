"""Ordered schema migrations for the three platform databases.

Each migration is applied once and recorded in schema_migrations of the
database it belongs to. New nullable columns are added with ALTER TABLE in a
later migration; there are no placeholder extension columns (design 4.9).
"""
from __future__ import annotations

import sqlite3
import time

CORE = [
    (1, """
CREATE TABLE users(
  user_id TEXT PRIMARY KEY,
  display_name TEXT,
  role TEXT NOT NULL CHECK(role IN ('admin','developer','viewer')),
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','disabled')),
  created_at INTEGER NOT NULL,
  last_seen_at INTEGER,
  attrs TEXT
);
CREATE TABLE user_tokens(
  token_id INTEGER PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  kind TEXT NOT NULL CHECK(kind IN ('client','agent','web')),
  token_prefix TEXT NOT NULL,
  token_hash BLOB NOT NULL UNIQUE,
  scopes TEXT NOT NULL,
  expires_at INTEGER,
  revoked_at INTEGER,
  created_at INTEGER NOT NULL,
  last_used_at INTEGER
);
CREATE INDEX user_tokens_user ON user_tokens(user_id);

CREATE TABLE clients(
  client_id INTEGER PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(user_id),
  machine_id TEXT NOT NULL,
  hostname TEXT,
  npu_arch TEXT,
  afx_version TEXT,
  backend_versions TEXT,
  first_seen_at INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL,
  attrs TEXT,
  UNIQUE(user_id, machine_id)
);

CREATE TABLE runs(
  run_id INTEGER PRIMARY KEY,
  run_uuid TEXT NOT NULL UNIQUE,
  user_id TEXT NOT NULL,
  client_id INTEGER NOT NULL REFERENCES clients(client_id),
  kind TEXT NOT NULL CHECK(kind IN ('fuzz','retest','verify','regress','import')),
  created_at INTEGER NOT NULL,
  attrs TEXT
);

CREATE TABLE environments(
  env_id INTEGER PRIMARY KEY,
  env_hash BLOB NOT NULL UNIQUE,
  cann TEXT, python TEXT, torch TEXT, torch_npu TEXT, backend_version TEXT,
  afx TEXT, nnsmith TEXT, os TEXT, cpu_arch TEXT, gcc TEXT
);

CREATE TABLE cases(
  case_id INTEGER PRIMARY KEY,
  case_key TEXT NOT NULL UNIQUE,
  case_source TEXT NOT NULL CHECK(case_source IN ('nnsmith','baize_tf','baize_torch','model')),
  case_revision INTEGER NOT NULL DEFAULT 1,
  graph_fingerprint BLOB,
  is_dynamic INTEGER NOT NULL,
  shape_count INTEGER NOT NULL,
  node_count INTEGER,
  op_types TEXT,
  dtypes TEXT,
  max_dim INTEGER,
  max_numel INTEGER,
  spec_content_id INTEGER,
  import_source_id INTEGER,
  external_ref TEXT,
  first_seen_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  attrs TEXT
);

CREATE TABLE case_revisions(
  case_id INTEGER NOT NULL,
  revision INTEGER NOT NULL,
  graph_fingerprint BLOB,
  created_at INTEGER NOT NULL,
  PRIMARY KEY(case_id, revision)
) WITHOUT ROWID;
CREATE INDEX case_revisions_fingerprint ON case_revisions(case_id, graph_fingerprint);

CREATE TABLE feature_registry(
  feature_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL CHECK(kind IN ('discrete','numeric')),
  groupable INTEGER NOT NULL DEFAULT 0,
  axis INTEGER NOT NULL DEFAULT 0,
  extractor_version INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL
);
CREATE TABLE feature_values(
  value_id INTEGER PRIMARY KEY,
  feature_id INTEGER NOT NULL REFERENCES feature_registry(feature_id),
  value TEXT NOT NULL,
  UNIQUE(feature_id, value)
);
CREATE TABLE case_features(
  case_id INTEGER NOT NULL,
  feature_id INTEGER NOT NULL,
  value_id INTEGER NOT NULL,
  PRIMARY KEY(case_id, feature_id, value_id)
) WITHOUT ROWID;
CREATE INDEX case_features_value ON case_features(value_id, case_id);
CREATE TABLE case_metrics(
  feature_id INTEGER NOT NULL,
  case_id INTEGER NOT NULL,
  value REAL NOT NULL,
  PRIMARY KEY(feature_id, case_id)
) WITHOUT ROWID;

CREATE TABLE executions(
  exec_id INTEGER PRIMARY KEY,
  exec_uuid TEXT NOT NULL UNIQUE,
  content_hash BLOB NOT NULL,
  case_id INTEGER NOT NULL REFERENCES cases(case_id),
  case_revision INTEGER NOT NULL,
  backend TEXT NOT NULL,
  npu_arch TEXT NOT NULL DEFAULT '',
  user_id TEXT NOT NULL,
  client_id INTEGER NOT NULL,
  run_id INTEGER NOT NULL,
  run_kind TEXT NOT NULL,
  env_id INTEGER NOT NULL REFERENCES environments(env_id),
  function_status TEXT NOT NULL,
  fail_stage TEXT,
  accuracy_status TEXT,
  accuracy_ref TEXT,
  max_abs_err REAL,
  max_rel_err REAL,
  compile_ns INTEGER,
  kernel_ns INTEGER,
  e2e_ns INTEGER NOT NULL,
  log_summary TEXT,
  log_content_id INTEGER,
  issue_signature BLOB,
  created_at INTEGER NOT NULL,
  received_at INTEGER NOT NULL,
  attrs TEXT
);
CREATE INDEX executions_case ON executions(case_id, created_at DESC);
CREATE INDEX executions_run ON executions(run_id);
CREATE INDEX executions_signature ON executions(issue_signature) WHERE issue_signature IS NOT NULL;

CREATE TABLE execution_shapes(
  exec_id INTEGER NOT NULL,
  shape_index INTEGER NOT NULL,
  function_status TEXT NOT NULL,
  accuracy_status TEXT,
  max_abs_err REAL,
  max_rel_err REAL,
  kernel_ns INTEGER,
  e2e_ns INTEGER,
  PRIMARY KEY(exec_id, shape_index)
) WITHOUT ROWID;

CREATE TABLE execution_heads(
  case_id INTEGER NOT NULL,
  backend TEXT NOT NULL,
  npu_arch TEXT NOT NULL,
  exec_id INTEGER NOT NULL,
  PRIMARY KEY(case_id, backend, npu_arch)
) WITHOUT ROWID;
CREATE INDEX execution_heads_backend ON execution_heads(backend, case_id);

CREATE TABLE upload_requests(
  user_id TEXT NOT NULL,
  request_key TEXT NOT NULL,
  body_hash BLOB NOT NULL,
  response TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY(user_id, request_key)
) WITHOUT ROWID;

CREATE TABLE platform_settings(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  revision INTEGER NOT NULL DEFAULT 1,
  updated_by TEXT,
  updated_at INTEGER NOT NULL
);
CREATE TABLE audit_log(
  audit_id INTEGER PRIMARY KEY,
  at INTEGER NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  detail TEXT
);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO meta(key, value) VALUES('ingest_seq', '0');
INSERT INTO meta(key, value) VALUES('case_count', '0');
"""),
    (2, """
CREATE TABLE issues(
  issue_no INTEGER PRIMARY KEY,
  issue_id TEXT NOT NULL UNIQUE,
  dimension TEXT NOT NULL CHECK(dimension IN ('function','accuracy','performance')),
  backend TEXT NOT NULL,
  signature BLOB NOT NULL,
  category TEXT NOT NULL,
  title TEXT NOT NULL,
  log_summary TEXT,
  status TEXT NOT NULL CHECK(status IN ('open','claimed','verifying','regressing','summarizing','resolved','closed')),
  priority INTEGER NOT NULL DEFAULT 0,
  case_count INTEGER NOT NULL DEFAULT 0,
  owner_user TEXT,
  agent_id TEXT,
  lease_id TEXT,
  lease_expires_at INTEGER,
  fix_version TEXT,
  fix_sha256 TEXT,
  fix_ref TEXT,
  knowledge_id TEXT,
  caused_by TEXT,
  duplicate_of TEXT,
  close_reason TEXT,
  round_no INTEGER NOT NULL DEFAULT 0,
  first_seen_at INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL,
  claimed_at INTEGER,
  resolved_at INTEGER,
  updated_at INTEGER NOT NULL,
  attrs TEXT,
  UNIQUE(dimension, backend, signature)
);
CREATE INDEX issues_status ON issues(status, dimension, backend, priority DESC);
CREATE INDEX issues_owner ON issues(owner_user, agent_id) WHERE lease_id IS NOT NULL;
CREATE TABLE issue_signature_alias(
  dimension TEXT NOT NULL,
  backend TEXT NOT NULL,
  signature BLOB NOT NULL,
  issue_id TEXT NOT NULL,
  PRIMARY KEY(dimension, backend, signature)
) WITHOUT ROWID;
CREATE TABLE issue_cases(
  issue_id TEXT NOT NULL,
  case_id INTEGER NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('affected','regression')),
  npu_arch TEXT NOT NULL,
  pick_reason TEXT,
  case_revision INTEGER NOT NULL,
  stale INTEGER NOT NULL DEFAULT 0,
  first_exec_id INTEGER,
  last_exec_id INTEGER,
  round_no INTEGER,
  result TEXT,
  added_at INTEGER NOT NULL,
  PRIMARY KEY(issue_id, role, case_id, npu_arch)
) WITHOUT ROWID;
CREATE INDEX issue_cases_case ON issue_cases(case_id);
CREATE TABLE issue_rounds(
  issue_id TEXT NOT NULL,
  round_no INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN ('verify','regress')),
  fix_version TEXT NOT NULL,
  fix_sha256 TEXT NOT NULL,
  selection_seed INTEGER,
  started_at INTEGER NOT NULL,
  finished_at INTEGER,
  outcome TEXT,
  pass_count INTEGER NOT NULL DEFAULT 0,
  fail_count INTEGER NOT NULL DEFAULT 0,
  invalid_count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(issue_id, round_no)
) WITHOUT ROWID;
CREATE TABLE issue_events(
  event_id INTEGER PRIMARY KEY,
  issue_id TEXT NOT NULL,
  at INTEGER NOT NULL,
  actor TEXT NOT NULL,
  kind TEXT NOT NULL,
  from_status TEXT,
  to_status TEXT,
  detail TEXT
);
CREATE INDEX issue_events_issue ON issue_events(issue_id, event_id);

CREATE TABLE regression_pins(
  pin_id INTEGER PRIMARY KEY,
  scope TEXT NOT NULL CHECK(scope IN ('global','feature','issue')),
  scope_value TEXT NOT NULL,
  case_id INTEGER NOT NULL,
  added_by TEXT NOT NULL,
  reason TEXT,
  created_at INTEGER NOT NULL,
  revoked_at INTEGER
);
CREATE INDEX regression_pins_scope ON regression_pins(scope, scope_value) WHERE revoked_at IS NULL;

CREATE TABLE retest_tasks(
  task_id INTEGER PRIMARY KEY,
  issue_id TEXT NOT NULL,
  round_no INTEGER NOT NULL,
  case_id INTEGER NOT NULL,
  npu_arch TEXT NOT NULL,
  backends TEXT NOT NULL,
  required_backend TEXT NOT NULL,
  required_version TEXT NOT NULL,
  required_sha256 TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('pending','leased','done','invalid','cancelled')),
  client_id INTEGER,
  lease_id TEXT,
  lease_expires_at INTEGER,
  attempts INTEGER NOT NULL DEFAULT 0,
  result TEXT,
  detail TEXT,
  created_at INTEGER NOT NULL,
  finished_at INTEGER
);
CREATE INDEX retest_tasks_queue ON retest_tasks(status, npu_arch, required_backend, required_version);
CREATE INDEX retest_tasks_round ON retest_tasks(issue_id, round_no);
CREATE TABLE retest_task_execs(
  task_id INTEGER NOT NULL,
  exec_id INTEGER NOT NULL,
  PRIMARY KEY(task_id, exec_id)
) WITHOUT ROWID;

CREATE TABLE knowledge(
  knowledge_no INTEGER PRIMARY KEY,
  knowledge_id TEXT NOT NULL UNIQUE,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  keywords TEXT NOT NULL,
  ops TEXT, dtypes TEXT, backends TEXT,
  graph_pattern TEXT,
  symptom TEXT, root_cause TEXT, solution TEXT, applicability TEXT,
  evidence TEXT,
  version_range TEXT,
  confidence TEXT NOT NULL CHECK(confidence IN ('verified','measured','proposed')),
  status TEXT NOT NULL CHECK(status IN ('draft','published','deprecated')),
  author_user TEXT NOT NULL,
  agent_id TEXT,
  body_md TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  attrs TEXT
);
CREATE TABLE knowledge_links(
  knowledge_id TEXT NOT NULL,
  target_kind TEXT NOT NULL CHECK(target_kind IN ('issue','case','knowledge')),
  target_id TEXT NOT NULL,
  PRIMARY KEY(knowledge_id, target_kind, target_id)
) WITHOUT ROWID;
CREATE VIRTUAL TABLE knowledge_fts USING fts5(title, keywords, body, tokenize='trigram');
CREATE TABLE agent_requests(
  user_id TEXT NOT NULL,
  request_key TEXT NOT NULL,
  route TEXT NOT NULL,
  body_hash BLOB NOT NULL,
  status INTEGER NOT NULL,
  response TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY(user_id, request_key)
) WITHOUT ROWID;
"""),
    (3, """
ALTER TABLE clients ADD COLUMN target_afx_version TEXT;
ALTER TABLE clients ADD COLUMN last_heartbeat_at INTEGER;
ALTER TABLE clients ADD COLUMN busy INTEGER;
ALTER TABLE clients ADD COLUMN pending_uploads INTEGER;
CREATE TABLE metric_rollups(
  client_id INTEGER NOT NULL,
  hour INTEGER NOT NULL,
  executions INTEGER NOT NULL DEFAULT 0,
  uploads INTEGER NOT NULL DEFAULT 0,
  upload_errors INTEGER NOT NULL DEFAULT 0,
  retests INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(client_id, hour)
) WITHOUT ROWID;
CREATE INDEX metric_rollups_hour ON metric_rollups(hour);
ALTER TABLE retest_tasks ADD COLUMN task_kind TEXT;
ALTER TABLE retest_tasks ADD COLUMN requested_by TEXT;
ALTER TABLE retest_tasks ADD COLUMN batch TEXT;
CREATE INDEX retest_tasks_batch ON retest_tasks(requested_by, batch) WHERE batch IS NOT NULL;
CREATE TABLE import_sources(
  import_id INTEGER PRIMARY KEY,
  framework TEXT NOT NULL CHECK(framework IN ('baize_tf','baize_torch')),
  package_version TEXT,
  user_id TEXT NOT NULL,
  case_count INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  attrs TEXT
);
CREATE TABLE model_jobs(
  job_id TEXT PRIMARY KEY,
  model_name TEXT NOT NULL,
  model_kind TEXT NOT NULL CHECK(model_kind IN ('rec','llm')),
  config_content_id INTEGER,
  backend TEXT NOT NULL,
  npu_arch TEXT,
  env_id INTEGER,
  user_id TEXT NOT NULL,
  client_id INTEGER,
  status TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  finished_at INTEGER,
  attrs TEXT
);
CREATE TABLE model_reports(
  job_id TEXT NOT NULL,
  metric TEXT NOT NULL,
  value REAL,
  content_id INTEGER,
  PRIMARY KEY(job_id, metric)
) WITHOUT ROWID;
"""),
]

CONTENT = [
    (1, """
CREATE TABLE contents(
  content_id INTEGER PRIMARY KEY,
  sha256 BLOB NOT NULL UNIQUE,
  codec TEXT NOT NULL,
  raw_size INTEGER NOT NULL,
  data BLOB NOT NULL,
  created_at INTEGER NOT NULL
);
"""),
]

STATS = [
    (1, """
CREATE TABLE stat_buckets(
  backend TEXT NOT NULL,
  npu_arch TEXT NOT NULL,
  dim TEXT NOT NULL,
  dim_value TEXT NOT NULL,
  function_status TEXT NOT NULL,
  fail_stage TEXT NOT NULL,
  accuracy_status TEXT NOT NULL,
  count INTEGER NOT NULL,
  PRIMARY KEY(backend, npu_arch, dim, dim_value, function_status, fail_stage, accuracy_status)
) WITHOUT ROWID;
CREATE TABLE perf_pair_buckets(
  backend_x TEXT NOT NULL,
  backend_y TEXT NOT NULL,
  npu_arch TEXT NOT NULL,
  dim TEXT NOT NULL,
  dim_value TEXT NOT NULL,
  verdict TEXT NOT NULL,
  ratio_bin INTEGER NOT NULL,
  count INTEGER NOT NULL,
  sum_log REAL NOT NULL,
  PRIMARY KEY(backend_x, backend_y, npu_arch, dim, dim_value, verdict, ratio_bin)
) WITHOUT ROWID;
CREATE TABLE stats_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO stats_meta(key, value) VALUES('applied_seq', '0');
"""),
]


def apply(conn: sqlite3.Connection, schema: str, migrations) -> None:
    """Apply pending migrations for one attached schema ('main', 'content', 'stats')."""
    conn.execute(f"CREATE TABLE IF NOT EXISTS {schema}.schema_migrations("
                 "version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL)")
    done = {row[0] for row in conn.execute(f"SELECT version FROM {schema}.schema_migrations")}
    for version, sql in migrations:
        if version in done:
            continue
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _statements(sql):
                conn.execute(_qualify(statement, schema))
            conn.execute(f"INSERT INTO {schema}.schema_migrations VALUES(?, ?)",
                         (version, int(time.time())))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


def _statements(sql: str):
    for chunk in sql.split(";"):
        chunk = chunk.strip()
        if chunk:
            yield chunk


def _qualify(statement: str, schema: str) -> str:
    """Prefix CREATE TABLE/INDEX and INSERT targets with the attached schema name."""
    if schema == "main":
        return statement
    for keyword in ("CREATE TABLE ", "CREATE INDEX ", "CREATE UNIQUE INDEX ", "INSERT INTO "):
        if statement.startswith(keyword):
            rest = statement[len(keyword):]
            return f"{keyword}{schema}.{rest}"
    return statement
