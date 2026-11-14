"""Owned result schema; unknown versions are rejected, never silently migrated."""
SCHEMA_VERSION = 1
APPLICATION_ID = 0x4146465A
DDL = """
CREATE TABLE contexts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('execution','evaluation','environment')),
    payload TEXT NOT NULL
);
CREATE TABLE attempts (
    id TEXT PRIMARY KEY,
    seed TEXT NOT NULL,
    workload_id TEXT NOT NULL,
    oracle_id TEXT NOT NULL,
    backend TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    execution_id TEXT NOT NULL REFERENCES contexts(id),
    evaluation_id TEXT NOT NULL REFERENCES contexts(id),
    environment_id TEXT NOT NULL REFERENCES contexts(id),
    prepare_ns INTEGER,
    error_stage TEXT,
    error_summary TEXT,
    result_hash TEXT NOT NULL
);
CREATE INDEX attempts_seed ON attempts(seed, created_at);
CREATE TABLE shapes (
    attempt_id TEXT NOT NULL REFERENCES attempts(id),
    shape_index INTEGER NOT NULL,
    correctness TEXT NOT NULL,
    performance TEXT NOT NULL,
    first_call_combined_ns INTEGER,
    mean_wall_ns INTEGER,
    measured_steps INTEGER,
    metric TEXT,
    error_stage TEXT,
    error_summary TEXT,
    PRIMARY KEY(attempt_id, shape_index)
);
"""
TABLE_COLUMNS = {
    'contexts': ('id', 'kind', 'payload'),
    'attempts': ('id','seed','workload_id','oracle_id','backend','status','created_at',
                 'execution_id','evaluation_id','environment_id','prepare_ns','error_stage','error_summary','result_hash'),
    'shapes': ('attempt_id','shape_index','correctness','performance','first_call_combined_ns',
               'mean_wall_ns','measured_steps','metric','error_stage','error_summary'),
}
