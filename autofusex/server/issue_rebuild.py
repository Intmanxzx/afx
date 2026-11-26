"""Explicit offline issue reset. Never called as a schema migration."""
from autofusex.server import issues
from autofusex.server.store import now_ms


def rebuild(store, *, apply=False):
    with store.write() as conn:
        conn.execute('SAVEPOINT issue_rebuild')
        old = conn.execute('SELECT COUNT(*) FROM issues').fetchone()[0]
        # Never reuse identifiers: stale clients and idempotency responses must not target new issues.
        next_no = conn.execute('SELECT COALESCE(MAX(issue_no),0)+1 FROM issues').fetchone()[0]
        conn.execute("DELETE FROM retest_task_execs WHERE task_id IN (SELECT task_id FROM retest_tasks WHERE issue_id<>'')")
        conn.execute("DELETE FROM retest_tasks WHERE issue_id<>''")
        conn.execute("DELETE FROM knowledge_links WHERE target_kind='issue'")
        conn.execute("DELETE FROM regression_pins WHERE scope='issue'")
        for table in ('issue_current','issue_events','issue_rounds','issue_cases','issue_signature_alias','issues'):
            conn.execute(f'DELETE FROM {table}')
        # Reserve the old ID range until new clusters have been allocated.
        conn.execute("INSERT INTO issues(issue_no,issue_id,dimension,backend,signature,category,title,status,"
                     "first_seen_at,last_seen_at,updated_at) VALUES(?,'rebuild-reservation','function','',X'00','','','closed',0,0,0)", (next_no,))
        cfg = issues.config(conn)
        for row in conn.execute('SELECT case_id,exec_id FROM execution_heads ORDER BY exec_id').fetchall():
            issues.on_head(conn, row['case_id'], row['exec_id'], cfg)
        conn.execute("DELETE FROM issues WHERE issue_id='rebuild-reservation'")
        counts = dict(conn.execute('SELECT dimension,COUNT(*) FROM issues GROUP BY dimension').fetchall())
        report = dict(old_issues=old,new_issues=counts,applied=apply)
        if apply:
            conn.execute('INSERT INTO audit_log(at,actor,action,detail) VALUES(?,?,?,?)',
                         (now_ms(),'offline','issues.rebuild',str(report)))
        else:
            conn.execute('ROLLBACK TO issue_rebuild')
        conn.execute('RELEASE issue_rebuild')
        return report
