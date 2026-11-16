"""Durable public statistics: transactional change journal and group contributions.

The source stores only small invalidations, in the writer's own transaction.
Heavy reads run on a read snapshot; derived data/checkpoint commit together in a
separate, disposable SQLite file. No source write lock is held while calculating.
"""
from contextlib import closing
import json
import logging
from pathlib import Path
import sqlite3
import time

from ...client.storage.repository import _connect
from ...client.storage.queries import reading
from .statistics import statistics
from .efficiency import statistics as efficiency

from .failure_origin import VERSION as CLASSIFICATION_VERSION

VERSION = 7
BATCH = 64
COUNT_FIELDS = ('overall', 'backend', 'category', 'dtype', 'op', 'dimension', 'elements', 'mode', 'errors', 'failure_path')


def initialize(database):
    """Install tracking before any snapshot. All supported writes hit triggers."""
    from ...client.storage.stage_facts import initialize as stages
    from ...client.storage.lifecycle import initialize as lifecycle
    with closing(_connect(database, writable=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        stages(conn); lifecycle(conn)
        conn.execute('CREATE TABLE IF NOT EXISTS public_changes_v1 (seq INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,identity TEXT NOT NULL)')
        # These indexes also eliminate per-attempt scans in finite-window metrics.
        conn.execute('CREATE INDEX IF NOT EXISTS public_stage_attempt ON stage_facts_v1(attempt_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS public_window_attempt ON stage_attempts_v1(window_id,attempt_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS public_attempt_workload ON attempts(workload_id,id)')
        # v6 changes contribution ownership from provenance group to global seed.
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'public_%'").fetchall():
            conn.execute('DROP TRIGGER "'+name+'"')
        def emit(kind, select):
            return "INSERT INTO public_changes_v1(kind,identity) SELECT '"+kind+"',identity FROM ("+select+");"
        def groups(attempt):
            return emit('group', 'SELECT seed AS identity FROM attempts WHERE id='+attempt)
        def windows(attempt):
            return emit('window', 'SELECT window_id AS identity FROM stage_attempts_v1 WHERE attempt_id='+attempt)
        specs = {}
        for table in ('attempts', 'shapes', 'stage_metrics', 'attempt_provenance', 'lifecycle_v1'):
            column = 'id' if table == 'attempts' else 'attempt_id'
            specs[table] = lambda row, column=column, table=table: (
                (groups(row+'.'+column) if table != 'lifecycle_v1' else '') + windows(row+'.'+column))
        for table in ('stage_facts_v1', 'stage_attempts_v1', 'stage_windows_v1', 'stage_window_aggregates_v1'):
            specs[table] = lambda row: emit('window', 'SELECT '+row+'.window_id AS identity')
        for table in ('case_facts', 'case_dimensions'):
            specs[table] = lambda row: emit('group', 'SELECT DISTINCT a.seed AS identity FROM attempts a JOIN attempt_facts f ON f.attempt_id=a.id WHERE a.workload_id='+row+'.workload_id')
        for table, body in specs.items():
            for event, rows, timing in [('INSERT', ('NEW',), 'AFTER'), ('UPDATE', ('OLD','NEW'), 'AFTER'), ('DELETE', ('OLD',), 'BEFORE')]:
                sql = ''.join(body(row) for row in rows)
                # REPLACE may not fire DELETE triggers: capture previous group
                # before replacement, as well as the new group after insertion.
                if table == 'attempt_provenance' and event != 'DELETE':
                    sql += ''.join(emit('group', 'SELECT seed AS identity FROM attempts WHERE id='+row+'.attempt_id') for row in rows)
                conn.execute(f'CREATE TRIGGER IF NOT EXISTS public_{table}_{event.lower()}_v1 {timing} {event} ON {table} BEGIN {sql} END')
        conn.execute('''CREATE TRIGGER IF NOT EXISTS public_facts_replace_v1 BEFORE INSERT ON attempt_provenance BEGIN
            INSERT INTO public_changes_v1(kind,identity) SELECT 'group',seed FROM attempts WHERE id=NEW.attempt_id;
        END''')
        conn.execute('''CREATE TRIGGER IF NOT EXISTS public_attempt_window_replace_v1 BEFORE INSERT ON stage_attempts_v1 BEGIN
            INSERT INTO public_changes_v1(kind,identity) SELECT 'window',window_id FROM stage_attempts_v1 WHERE attempt_id=NEW.attempt_id;
        END''')
        conn.execute("""CREATE TRIGGER IF NOT EXISTS public_completion_windows_v1
            AFTER INSERT ON attempt_provenance BEGIN
            INSERT INTO public_changes_v1(kind,identity)
            SELECT DISTINCT 'window',s.window_id FROM attempt_facts f
            JOIN stage_attempts_v1 s ON s.attempt_id=f.attempt_id
            WHERE f.group_id=NEW.group_id;
        END""")
        # Contexts normally immutable, but explicit administrative edits must not
        # silently leave derived mode/window metrics stale.
        for event, row in [('UPDATE','NEW'), ('DELETE','OLD')]:
            conn.execute(f'''CREATE TRIGGER IF NOT EXISTS public_contexts_{event.lower()}_v1 BEFORE {event} ON contexts BEGIN
                INSERT INTO public_changes_v1(kind,identity) SELECT 'group',a.seed FROM attempts a JOIN attempt_facts f ON f.attempt_id=a.id WHERE {row}.id IN (a.execution_id,a.environment_id,a.evaluation_id);
                INSERT INTO public_changes_v1(kind,identity) SELECT 'window',s.window_id FROM attempts a JOIN stage_attempts_v1 s ON s.attempt_id=a.id WHERE {row}.id IN (a.execution_id,a.environment_id,a.evaluation_id);
            END''')


DDL = '''
CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY, source TEXT NOT NULL, checkpoint INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS pending (kind TEXT, identity TEXT, PRIMARY KEY(kind,identity));
CREATE TABLE IF NOT EXISTS statistic_keys (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(kind,payload));
CREATE TABLE IF NOT EXISTS contributions (
 kind TEXT, key INTEGER, owner TEXT, n INTEGER NOT NULL, total INTEGER, measured INTEGER NOT NULL,
 lo INTEGER, hi INTEGER, PRIMARY KEY(owner,kind,key));
CREATE INDEX IF NOT EXISTS contribution_min ON contributions(kind,key,lo);
CREATE INDEX IF NOT EXISTS contribution_max ON contributions(kind,key,hi);
CREATE TABLE IF NOT EXISTS totals (kind TEXT,key INTEGER,n INTEGER,total INTEGER,measured INTEGER,PRIMARY KEY(kind,key));
CREATE TABLE IF NOT EXISTS windows (identity TEXT PRIMARY KEY,payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS template (payload TEXT NOT NULL);
'''


def _key(row, fields):
    return json.dumps({k: row[k] for k in fields}, sort_keys=True, ensure_ascii=True)


def contributions(result):
    for kind in COUNT_FIELDS:
        fields = ('backend','error_stage') if kind == 'errors' else ('label','status')
        for row in result[kind]:
            yield kind, _key(row,fields), row['count'], 0, 0, None, None
    for kind in ('attempts','case_groups'):
        yield kind, '{}', result[kind], 0, 0, None, None
    for row in result['stages']:
        yield 'stages', _key(row,('backend','stage')), row['samples'], row['total_ns'], row['samples'], None, row['max_ns']
    for row in result['performance']:
        yield 'performance', _key(row,('backend','mode','correctness','performance','metric')), row['samples'], row['_sum_ns'] or 0, row['_measured'], row['min_ns'], row['max_ns']


def replace_group(conn, group, result):
    # Remove the group's old contribution before adding its recomputed value.
    # Extremes live in indexed per-group rows: deleting a former max/min is safe.
    for kind,key,n,total,measured in conn.execute('SELECT kind,key,n,total,measured FROM contributions WHERE owner=?',(group,)).fetchall():
        conn.execute('UPDATE totals SET n=n-?,total=total-?,measured=measured-? WHERE kind=? AND key=?',(n,total,measured,kind,key))
    conn.execute('DELETE FROM contributions WHERE owner=?',(group,))
    for kind,key,n,total,measured,lo,hi in contributions(result):
        if not n: continue
        conn.execute('INSERT OR IGNORE INTO statistic_keys(kind,payload) VALUES (?,?)',(kind,key))
        key=conn.execute('SELECT id FROM statistic_keys WHERE kind=? AND payload=?',(kind,key)).fetchone()[0]
        conn.execute('INSERT INTO contributions VALUES (?,?,?,?,?,?,?,?)',(kind,key,group,n,total,measured,lo,hi))
        conn.execute('''INSERT INTO totals VALUES (?,?,?,?,?) ON CONFLICT(kind,key) DO UPDATE SET
            n=n+excluded.n,total=total+excluded.total,measured=measured+excluded.measured''',(kind,key,n,total,measured))
    conn.execute('DELETE FROM totals WHERE n=0')


class IncrementalStatistics:
    def __init__(self, database, *, stopped=lambda: False):
        self.database = Path(database).resolve()
        self.path = Path(str(self.database)+'.public.sqlite')
        self.installed = None
        self.last_groups = 0
        self.stopped = stopped

    def _identity(self):
        s = self.database.stat()
        return json.dumps([VERSION,CLASSIFICATION_VERSION,s.st_dev,s.st_ino])

    def prepare(self):
        identity = self._identity()
        if identity != self.installed:
            initialize(self.database)
            self.installed = identity
        return identity

    def refresh(self):
        identity = self.prepare()
        self.path.touch(mode=0o600, exist_ok=True)
        self.last_groups = 0
        started = time.monotonic()
        with closing(sqlite3.connect(self.path, timeout=5)) as cache:
            cache.executescript(DDL)
            cache.execute('PRAGMA journal_mode=WAL')
            while True:
                if self.stopped():
                    raise RuntimeError('incremental refresh stopped')
                with reading(self.database, timeout=30) as source:
                    source.execute('BEGIN')
                    with cache:
                        cache.execute('BEGIN IMMEDIATE')
                        high = source.execute("SELECT coalesce((SELECT seq FROM sqlite_sequence WHERE name='public_changes_v1'),0)").fetchone()[0]
                        meta = cache.execute('SELECT source,checkpoint FROM meta WHERE id=1').fetchone()
                        if meta is None or meta[0] != identity or meta[1] > high:
                            for table in ('meta','pending','contributions','totals','windows','template','statistic_keys'):
                                cache.execute('DELETE FROM '+table)
                            # One bootstrap pass. Pending work is durable and bounded
                            # per batch, so restart/failure does not restart bootstrap.
                            cache.executemany("INSERT OR IGNORE INTO pending VALUES ('group',?)", source.execute('SELECT DISTINCT seed FROM attempts'))
                            cache.executemany("INSERT OR IGNORE INTO pending VALUES ('window',?)", source.execute('SELECT window_id FROM stage_windows_v1 ORDER BY rowid DESC LIMIT 100'))
                            empty = statistics(self.database, _conn=source, _group='')
                            cache.execute('INSERT INTO template VALUES (?)',(json.dumps(empty),))
                            cache.execute('INSERT INTO meta VALUES (1,?,?)',(identity,high))
                            low = high
                        else:
                            low = meta[1]
                        cache.executemany('INSERT OR IGNORE INTO pending VALUES (?,?)', source.execute('SELECT DISTINCT kind,identity FROM public_changes_v1 WHERE seq>? AND seq<=?',(low,high)))
                        for key, in source.execute('SELECT window_id FROM stage_windows_v1 ORDER BY rowid DESC LIMIT 100'):
                            if not cache.execute('SELECT 1 FROM windows WHERE identity=?',(key,)).fetchone():
                                cache.execute("INSERT OR IGNORE INTO pending VALUES ('window',?)",(key,))
                        work = cache.execute('SELECT kind,identity FROM pending ORDER BY kind,identity LIMIT ?', (BATCH,)).fetchall()
                        batch_started = time.monotonic()
                        for kind, key in work:
                            if kind == 'group':
                                replace_group(cache,key,statistics(self.database,_conn=source,_group=key))
                                self.last_groups += 1
                            else:
                                rows = efficiency(source,use_cached=True,window_id=key)
                                cache.execute('INSERT OR REPLACE INTO windows VALUES (?,?)',(key,json.dumps(rows)))
                            cache.execute('DELETE FROM pending WHERE kind=? AND identity=?',(kind,key))
                            if self.stopped() or time.monotonic()-batch_started >= 2:
                                break
                        cache.execute('UPDATE meta SET checkpoint=? WHERE id=1',(high,))
                        more = cache.execute('SELECT 1 FROM pending LIMIT 1').fetchone()
                        if not more:
                            result = self._render(cache,source)
                        # Commit the completed batch explicitly before touching the
                        # source journal. The surrounding connection is deliberately
                        # not allowed to roll back a saved checkpoint on timeout.
                        cache.commit()
                # Commit sidecar first, then prune journal. A crash in between
                # replays nothing twice (checkpoint); new commits have larger seq.
                with closing(_connect(self.database,writable=True)) as writer, writer:
                    writer.execute('DELETE FROM public_changes_v1 WHERE seq<=?',(high,))
                if not more:
                    logging.getLogger(__name__).info('Public incremental refresh: groups=%d elapsed=%.3fs',self.last_groups,time.monotonic()-started)
                    return result
                if time.monotonic()-started >= 20:
                    raise TimeoutError('incremental backlog checkpointed; continue next refresh')
                # Bootstrap/backlog yields between batches on small shared VMs.
                time.sleep(.05)

    @staticmethod
    def _render(cache,source):
        result = json.loads(cache.execute('SELECT payload FROM template').fetchone()[0])
        for kind in (*COUNT_FIELDS,'stages','performance'):
            result[kind] = []
        result['attempts'] = result['case_groups'] = 0
        for kind,key,n,total,measured in cache.execute('SELECT kind,key,n,total,measured FROM totals'):
            row = json.loads(cache.execute('SELECT payload FROM statistic_keys WHERE id=?',(key,)).fetchone()[0])
            if kind in ('attempts','case_groups'):
                result[kind] = n; continue
            if kind in COUNT_FIELDS:
                row['count'] = n
            else:
                hi = cache.execute('SELECT hi FROM contributions WHERE kind=? AND key=? AND hi IS NOT NULL ORDER BY hi DESC LIMIT 1',(kind,key)).fetchone()
                row.update(samples=n,mean_ns=total/measured if measured else None,max_ns=hi[0] if hi else None)
                if kind == 'stages': row['total_ns'] = total
                else:
                    lo = cache.execute('SELECT lo FROM contributions WHERE kind=? AND key=? AND lo IS NOT NULL ORDER BY lo LIMIT 1',(kind,key)).fetchone()
                    row['min_ns'] = lo[0] if lo else None
            result[kind].append(row)
        for kind in (*COUNT_FIELDS,'stages','performance'):
            fields = ('backend','stage') if kind == 'stages' else ('backend','mode','correctness','performance','metric') if kind == 'performance' else ('backend','error_stage') if kind == 'errors' else ('label','status')
            result[kind].sort(key=lambda row: tuple((row[k] is not None,row[k] or '') for k in fields))
        from ...client.backend_names import display_backend
        for row in result['backend']:row['label']=display_backend(row['label'])
        windows = []
        for key, in source.execute('SELECT window_id FROM stage_windows_v1 ORDER BY rowid DESC LIMIT 100'):
            row = cache.execute('SELECT payload FROM windows WHERE identity=?',(key,)).fetchone()
            if row: windows.extend(json.loads(row[0]))
        result['tool_efficiency']['windows'] = windows
        result['tool_efficiency']['processing'] = [r for r in result['stages'] if r['stage']=='shape_processing']
        result['precise_metrics'] = [r for r in result['stages'] if r['stage'] in {'compile','kernel_mean'} and not (r['stage']=='kernel_mean' and r['backend'] in {'cpu'})]
        return result
