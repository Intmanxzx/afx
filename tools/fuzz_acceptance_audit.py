"""Audit actual persisted ETR acceptance, never count generated-only cases.

Usage: python tools/fuzz_acceptance_audit.py results.sqlite --count 1000
Historical failed attempts remain visible. A passing repeat does not erase them.
"""
import argparse
from collections import Counter
import hashlib
import json
import sqlite3

from autofusedebugger.fuzz.generation import materialize, workload_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('database')
    parser.add_argument('--count', type=int, default=1000)
    args = parser.parse_args()
    conn = sqlite3.connect('file:' + args.database + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    missing = []; hashes = []; statuses = Counter(); failed = []; verified_shapes = 0
    for i in range(args.count):
        seed = str(2**128+i); expected = workload_id(materialize(seed)); hashes.append(expected)
        for backend in ('cpu-eager', 'ascendc'):
            rows = conn.execute('SELECT * FROM attempts WHERE seed=? AND backend=? ORDER BY rowid',
                                (seed, backend)).fetchall()
            passed = False
            for row in rows:
                statuses[backend+'/'+row['status']] += 1
                shapes = conn.execute('SELECT * FROM shapes WHERE attempt_id=? ORDER BY shape_index',
                                      (row['id'],)).fetchall()
                assert row['workload_id'] == expected, (seed, 'workload drift')
                if row['status'] != 'ok':
                    failed.append({'seed': seed, 'backend': backend, 'attempt_id': row['id'],
                                   'errors': [s['error_summary'] for s in shapes if s['error_summary']]})
                if row['status'] == 'ok' and [s['shape_index'] for s in shapes] == [0,1,2] and all(s['correctness']=='pass' for s in shapes):
                    passed = True
            if passed:
                verified_shapes += 3
            else:
                missing.append([seed, backend])
    sync = dict(conn.execute('SELECT source_id,ack FROM sync_identity WHERE id=1').fetchone())
    sync['max_revision'] = conn.execute('SELECT max(revision) FROM sync_outbox').fetchone()[0]
    print(json.dumps({'requested_seeds': args.count, 'verified_backend_cases': verified_shapes//3,
                      'verified_shape_checks': verified_shapes, 'status_counts': statuses,
                      'missing_passed_pairs': missing, 'historical_errors': failed,
                      'workloads_sha256': hashlib.sha256('\n'.join(hashes).encode()).hexdigest(),
                      'sync': sync, 'passed': not missing}, indent=2))
    conn.close()
    return bool(missing)


if __name__ == '__main__':
    raise SystemExit(main())
