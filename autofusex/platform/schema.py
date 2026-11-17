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
