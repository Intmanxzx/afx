"""Explicit administrative deletion of incomplete legacy result records.

Keep identities, receipts and ACKs. Tombstones prevent restored local archives
from uploading the same attempts under a new replication source.
"""
from contextlib import closing
from ..storage.repository import _connect


def purge_incomplete(path, *, expected_count):
    return _purge(path, expected_count=expected_count, incomplete=True)


def purge_all_results(path, *, expected_count):
    """Explicit reviewed reset; caller must take a private central backup first."""
    return _purge(path, expected_count=expected_count, incomplete=False)


def purge_attempts(path, identities):
    """Reviewed exact-ID cleanup, retaining the same anti-resurrection guards."""
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('nonempty unique attempt IDs required')
    return _purge(path, expected_count=len(identities), incomplete=False, identities=identities)


def _purge(path, *, expected_count, incomplete, identities=None):
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        query="""SELECT a.id,a.result_hash,o.user_id FROM attempts a
            JOIN owners o ON o.attempt_id=a.id JOIN contexts c ON c.id=a.environment_id
            """
        if incomplete:
            query += """WHERE json_extract(c.payload,'$.cann') IS NULL
            AND json_extract(c.payload,'$.autofuse_binary') IS NULL
            AND json_extract(c.payload,'$.autofuse_commit') IS NULL"""
        if identities:
            query += ' WHERE a.id IN (' + ','.join('?' for _ in identities) + ')'
        rows=conn.execute(query, identities or ()).fetchall()
        if len(rows)!=expected_count:
            raise ValueError('record count changed; review before deleting')
        conn.execute('CREATE TABLE IF NOT EXISTS deleted_attempts (id TEXT PRIMARY KEY, result_hash TEXT NOT NULL, user_id TEXT NOT NULL)')
        conn.executemany('INSERT OR IGNORE INTO deleted_attempts VALUES (?,?,?)',rows)
        for identity, _, _ in rows:
            for table,column in [('report_search','attempt_id'),('stage_metrics','attempt_id'),('attempt_facts','attempt_id'),('shapes','attempt_id'),('owners','attempt_id'),('attempts','id')]:
                conn.execute(f'DELETE FROM {table} WHERE {column}=?',(identity,))
        conn.execute('DELETE FROM case_dimensions WHERE workload_id NOT IN (SELECT workload_id FROM attempts)')
        conn.execute('DELETE FROM case_facts WHERE workload_id NOT IN (SELECT workload_id FROM attempts)')
        conn.execute('DELETE FROM report_devices WHERE source_id NOT IN (SELECT source_id FROM attempt_facts)')
        conn.execute('DELETE FROM contexts WHERE id NOT IN (SELECT environment_id FROM attempts UNION SELECT execution_id FROM attempts UNION SELECT evaluation_id FROM attempts)')
        if conn.execute('PRAGMA foreign_key_check').fetchone():raise ValueError('cleanup broke foreign keys')
        return {'deleted':len(rows),'remaining':conn.execute('SELECT count(*) FROM attempts').fetchone()[0]}
