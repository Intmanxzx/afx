"""Explicit administrative result deletion.

Keep identities, receipts and ACKs. Tombstones prevent restored local archives
from uploading the same attempts under a new replication source.
"""
from contextlib import closing
from ..storage.repository import _connect


def purge_all_results(path, *, expected_count):
    """Explicit reviewed reset; caller must take a private central backup first."""
    return _purge(path, expected_count=expected_count)


def purge_attempts(path, identities):
    """Reviewed exact-ID cleanup, retaining the same anti-resurrection guards."""
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('nonempty unique attempt IDs required')
    return _purge(path, expected_count=len(identities), identities=identities)


def _purge(path, *, expected_count, identities=None):
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        query="""SELECT a.id,a.seed,a.result_hash,o.user_id FROM attempts a
            JOIN owners o ON o.attempt_id=a.id JOIN contexts c ON c.id=a.environment_id
            """
        if identities:
            query += ' WHERE a.id IN (' + ','.join('?' for _ in identities) + ')'
        rows=conn.execute(query, identities or ()).fetchall()
        seeds={row[1] for row in rows}
        if len(rows)!=expected_count:
            raise ValueError('record count changed; review before deleting')
        conn.execute('CREATE TABLE IF NOT EXISTS deleted_attempts (id TEXT PRIMARY KEY, result_hash TEXT NOT NULL, user_id TEXT NOT NULL)')
        conn.executemany('INSERT OR IGNORE INTO deleted_attempts VALUES (?,?,?)',[(row[0],row[2],row[3]) for row in rows])
        for identity, _, _, _ in rows:
            for table,column in [('report_search','attempt_id'),('stage_metrics','attempt_id'),('attempt_facts','attempt_id'),('shapes','attempt_id'),('owners','attempt_id'),('attempts','id')]:
                conn.execute(f'DELETE FROM {table} WHERE {column}=?',(identity,))
        for table in ('lifecycle_v1','stage_attempts_v1'):
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name=?",(table,)).fetchone():
                conn.execute('DELETE FROM '+table+' WHERE attempt_id NOT IN (SELECT id FROM attempts)')
        if conn.execute("SELECT 1 FROM sqlite_master WHERE name='stage_facts_v1'").fetchone():
            conn.execute('DELETE FROM stage_facts_v1 WHERE case_id NOT IN (SELECT case_id FROM stage_attempts_v1) OR (attempt_id IS NOT NULL AND attempt_id NOT IN (SELECT id FROM attempts))')
            # Aggregates including removed attempts cannot remain public.
            conn.execute('DELETE FROM stage_window_aggregates_v1')
            conn.execute('DELETE FROM stage_windows_v1 WHERE window_id NOT IN (SELECT window_id FROM stage_attempts_v1)')
        conn.execute('DELETE FROM case_dimensions WHERE workload_id NOT IN (SELECT workload_id FROM attempts)')
        conn.execute('DELETE FROM case_facts WHERE workload_id NOT IN (SELECT workload_id FROM attempts)')
        conn.execute('DELETE FROM report_devices WHERE source_id NOT IN (SELECT source_id FROM attempt_facts)')
        conn.execute('DELETE FROM contexts WHERE id NOT IN (SELECT environment_id FROM attempts UNION SELECT execution_id FROM attempts UNION SELECT evaluation_id FROM attempts)')
        from ..reporting.materialized import refresh_if_installed
        refresh_if_installed(conn)
        from .issue_groups import refresh_index
        refresh_index(conn, seeds)
        if conn.execute('PRAGMA foreign_key_check').fetchone():raise ValueError('cleanup broke foreign keys')
        return {'deleted':len(rows),'remaining':conn.execute('SELECT count(*) FROM attempts').fetchone()[0]}
