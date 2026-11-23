"""Explicit maintenance for unreferenced derived content; never on a GET path."""
from contextlib import closing
from autofusex.execution.storage.repository import _connect


def collect(database):
    with closing(_connect(database, writable=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        counts = {}
        for table, predicate in (
            ('report_versions', 'id NOT IN (SELECT version_id FROM attempt_provenance)'),
            ('case_preview_v2', 'id NOT IN (SELECT preview_id FROM case_attempt_refs_v2 WHERE preview_id IS NOT NULL)'),
            ('display_text_v2', '''id NOT IN (
                SELECT evidence_id FROM case_display_refs_v2 WHERE evidence_id IS NOT NULL
                UNION SELECT core_error_id FROM case_display_refs_v2 WHERE core_error_id IS NOT NULL
                UNION SELECT run_log_id FROM case_display_refs_v2 WHERE run_log_id IS NOT NULL)'''),
        ):
            if conn.execute('SELECT 1 FROM sqlite_master WHERE name=?', (table,)).fetchone():
                counts[table] = conn.execute(f'DELETE FROM {table} WHERE {predicate}').rowcount
        return counts
