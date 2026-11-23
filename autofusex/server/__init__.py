"""AutofuseX one-stop development platform server (v3).

Design: docs/fuzz/platform_v3_design.md. Data lives in three SQLite files:
core.db (authoritative facts), content.db (deduplicated compressed blobs) and
stats.db (incremental aggregates that can always be rebuilt from core.db).
"""
