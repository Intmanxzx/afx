#!/bin/sh
# Run as the service owner; private, consistent SQLite backup, seven-day retention.
set -eu
umask 077
ROOT=${AF_FUZZ_ROOT:-/home/ubuntu/af-fuzz}
PYTHON=${AF_FUZZ_PYTHON:-$ROOT/venv/bin/afx}
mkdir -p "$ROOT/private/backups"
"$PYTHON" fuzz admin backup "$ROOT/data/results.sqlite" \
  "$ROOT/private/backups/results-$(date -u +%Y%m%dT%H%M%S)-$$.sqlite"
# Scope only this script's backup naming pattern, never deployment rollback files.
find "$ROOT/private/backups" -maxdepth 1 -type f -name 'results-*.sqlite' -mtime +7 -delete
