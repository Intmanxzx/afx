import json
import sqlite3

from autofusex.execution.storage.stage_facts import initialize, timeline


def test_timeline_uses_case_index_and_preserves_shared_and_attempt_stages():
    with sqlite3.connect(':memory:') as conn:
        initialize(conn)
        conn.execute('CREATE INDEX public_stage_attempt ON stage_facts_v1(attempt_id)')
        conn.execute("INSERT INTO stage_attempts_v1 VALUES ('target','case','window')")
        rows = [('shared', 'window', 'case', None, None, json.dumps({'kind':'reference'})),
                ('own', 'window', 'case', None, 'target', json.dumps({'kind':'compare'})),
                ('peer', 'window', 'case', None, 'peer', '{}')]
        rows += [(str(i), 'window', 'other'+str(i), None, None, '{}') for i in range(1000)]
        conn.executemany('INSERT INTO stage_facts_v1 VALUES (?,?,?,?,?,?)', rows)
        plan = conn.execute('EXPLAIN QUERY PLAN SELECT stage_id,payload FROM stage_facts_v1 '
            'WHERE case_id=? AND (attempt_id=? OR attempt_id IS NULL)', ('case','target')).fetchall()
        assert any('stage_case_attempt_v1' in row[3] and 'case_id=?' in row[3] for row in plan)
        assert {r['stage_id'] for r in timeline(conn,'target')} == {'shared','own'}
        assert timeline(conn,'missing') == []
        conn.execute('DROP INDEX stage_case_attempt_v1')
        initialize(conn)  # Existing deployments acquire the additive index too.
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='stage_case_attempt_v1'").fetchone()
