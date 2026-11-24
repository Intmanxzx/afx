"""Space invariant and fail-closed collision checks for shared display text."""
import sqlite3
import pytest
from autofusex.execution.reporting.shared_content import intern_text


def test_long_unique_content_is_not_copied_into_index():
    with sqlite3.connect(':memory:') as conn:
        conn.execute('CREATE TABLE display_text_v2(id INTEGER PRIMARY KEY, digest BLOB UNIQUE, content TEXT)')
        value = 'diagnostic line\n' * 10000
        first = intern_text(conn, 'display_text_v2', 'content', value)
        assert intern_text(conn, 'display_text_v2', 'content', value) == first
        assert conn.execute('SELECT count(*),length(digest) FROM display_text_v2').fetchone() == (1, 32)
        index = conn.execute('PRAGMA index_list(display_text_v2)').fetchone()[1]
        assert [r[2] for r in conn.execute(f'PRAGMA index_info({index})')] == ['digest']
        conn.execute('UPDATE display_text_v2 SET content=?', ('corrupted',))
        with pytest.raises(ValueError, match='collision or corruption'):
            intern_text(conn, 'display_text_v2', 'content', value)


def test_collection_preserves_live_display_and_rejects_silent_context_edits(tmp_path):
    from contextlib import closing
    from autofusex.execution.storage.repository import ResultRepository, _connect
    from autofusex.execution.reporting.materialized import rebuild
    from autofusex.execution.reporting.compact import collect
    from autofusex.execution.storage.queries import ResultQueries
    from test_fuzz_global_rounds import save
    repo = ResultRepository(tmp_path / 'results')
    save(repo, 'one', 'user', 'cpu')
    save(repo, 'one', 'user', 'af-mlir')
    rebuild(repo.path)
    before = ResultQueries(repo.path).case_page()
    with closing(_connect(repo.path, writable=True)) as conn, conn:
        intern_text(conn, 'display_text_v2', 'content', 'orphan text')
        intern_text(conn, 'case_preview_v2', 'payload', 'orphan preview')
        with pytest.raises(sqlite3.IntegrityError, match='immutable facts'):
            conn.execute("UPDATE attempt_facts SET execution='{}'")
    counts = collect(repo.path)
    assert counts['display_text_v2'] == counts['case_preview_v2'] == 1
    assert ResultQueries(repo.path).case_page() == before
    with closing(_connect(repo.path)) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert not conn.execute('PRAGMA foreign_key_check').fetchall()


def test_statistics_labels_are_shared_between_owners():
    from autofusex.execution.reporting.incremental import DDL, replace_group, COUNT_FIELDS
    result = {kind: [] for kind in COUNT_FIELDS}
    result.update(overall=[dict(label='all', status='ok', count=1)],
                  attempts=1, case_groups=1, stages=[], performance=[])
    with sqlite3.connect(':memory:') as conn:
        conn.executescript(DDL)
        for n in range(100):
            replace_group(conn, str(n), result)
        assert conn.execute('SELECT count(*) FROM statistic_keys').fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM contributions WHERE typeof(key)!='integer'").fetchone()[0] == 0
        assert conn.execute("SELECT n FROM totals WHERE kind='attempts'").fetchone()[0] == 100
        replace_group(conn, '0', result)
        assert conn.execute("SELECT n FROM totals WHERE kind='attempts'").fetchone()[0] == 100


def test_current_heads_store_references_not_duplicate_history(tmp_path):
    from contextlib import closing
    from autofusex.execution.storage.repository import ResultRepository, _connect
    from autofusex.execution.reporting.materialized import rebuild
    from test_fuzz_global_rounds import save
    repo = ResultRepository(tmp_path / 'db')
    for batch in ('first', 'second'):
        save(repo, batch, 'user', 'cpu')
        save(repo, batch, 'user', 'af-mlir')
    rebuild(repo.path)
    with closing(_connect(repo.path)) as conn:
        assert conn.execute("SELECT type FROM sqlite_master WHERE name='case_heads_v1'").fetchone()[0] == 'view'
        assert len(conn.execute('PRAGMA table_info(case_head_refs_v2)').fetchall()) == 7
        assert conn.execute('SELECT count(*) FROM case_heads_v1').fetchone()[0] == 3
        assert conn.execute('SELECT count(*) FROM case_history_v1').fetchone()[0] == 6
        assert conn.execute('SELECT min(history_count) FROM case_heads_v1').fetchone()[0] == 2
        plan = [r[3] for r in conn.execute('EXPLAIN QUERY PLAN SELECT * FROM case_heads_v1 WHERE cursor<? ORDER BY cursor DESC LIMIT 51',(2**63-1,))]
        assert any('case_history_identity_v2 (cursor=?)' in step for step in plan)
        assert not any('SCAN r' in step for step in plan)
