"""Backfill current membership without resetting issue or verification history."""
from collections import Counter
import json

from autofusex.server import issues
from autofusex.server.error_logs import issue_summary
from autofusex.server.store import get_content, now_ms


def backfill(store, *, apply=False, progress=None):
    """Explicit offline repair; preview rolls back all derived changes.

    Frozen issue members and signatures are excluded. Existing current links,
    historical affected evidence, tasks and rounds are never overwritten.
    """
    with store.write() as conn:
        conn.execute('SAVEPOINT issue_memberships')
        cfg = issues.config(conn)
        protected = {r[0] for r in conn.execute(
            "SELECT issue_id FROM issues WHERE status IN ('verifying','regressing','summarizing')")}
        conn.execute('CREATE TEMP TABLE frozen_members(case_id,backend,npu_arch,PRIMARY KEY(case_id,backend,npu_arch))')
        conn.execute("INSERT OR IGNORE INTO frozen_members SELECT ic.case_id,ic.backend,ic.npu_arch "
                     "FROM issue_cases ic JOIN issues i USING(issue_id) "
                     "WHERE i.status IN ('verifying','regressing','summarizing')")
        frozen_signatures = {issues._digest('function', r['category'],
            issues.normalize(r['log_summary']) or 'No diagnostic evidence') for r in conn.execute(
            "SELECT category,log_summary FROM issues WHERE dimension='function' AND cluster_generation=2 "
            "AND status IN ('verifying','regressing','summarizing')")}
        placeholders = ','.join('?' for _ in cfg['backends'])
        rows = conn.execute(
            "SELECT e.* FROM execution_heads h JOIN executions e ON e.exec_id=h.exec_id "
            "JOIN cases c ON c.case_id=h.case_id LEFT JOIN issue_current ic "
            "ON ic.case_id=h.case_id AND ic.backend=h.backend AND ic.npu_arch=h.npu_arch "
            f"WHERE ic.issue_id IS NULL AND e.backend IN ({placeholders}) AND e.case_revision=c.case_revision "
            "AND (e.function_status NOT IN ('pass','unsupported','invalid') "
            "OR (e.function_status='pass' AND e.accuracy_status='fail')) ORDER BY e.exec_id",
            cfg['backends']).fetchall()
        counts = Counter()
        old_count = conn.execute('SELECT COUNT(*) FROM issues').fetchone()[0]
        for index, row in enumerate(rows, 1):
            key = (row['case_id'], row['backend'], row['npu_arch'])
            if conn.execute('SELECT 1 FROM frozen_members WHERE case_id=? AND backend=? AND npu_arch=?', key).fetchone():
                counts['frozen_members'] += 1
                continue
            traits = issues.case_traits(conn, row['case_id'])
            if issues.failure_dimension(row) == 'function':
                summary = issue_summary(
                    get_content(conn, row['detail_content_id']) if row['detail_content_id'] else None,
                    get_content(conn, row['log_content_id']) if row['log_content_id'] else None,
                    row['log_summary'], row['function_status'])
                signature = issues.function_signature(dict(row, log_summary=summary), traits)[0]
                dimension = 'function'
            else:
                signature = issues.accuracy_signature(row, traits)[0]
                dimension = 'accuracy'
            target = issues.signature_target(conn, dimension, signature)
            if signature in frozen_signatures or target is not None and target['issue_id'] in protected:
                counts['frozen_signatures'] += 1
                continue
            issues.reconcile(conn, row, cfg=cfg, preserve_evidence=True)
            counts['linked'] += 1
            if progress and index % 1000 == 0:
                progress(index, len(rows), dict(counts))
        report = dict(applied=apply, candidates=len(rows), counts=dict(counts),
                      new_issues=conn.execute('SELECT COUNT(*) FROM issues').fetchone()[0]-old_count,
                      protected_issues=sorted(protected))
        conn.execute('DROP TABLE frozen_members')
        if apply:
            conn.execute('INSERT INTO audit_log(at,actor,action,detail) VALUES(?,?,?,?)',
                         (now_ms(), 'offline', 'issues.memberships', json.dumps(report)))
        else:
            conn.execute('ROLLBACK TO issue_memberships')
        conn.execute('RELEASE issue_memberships')
        return report
