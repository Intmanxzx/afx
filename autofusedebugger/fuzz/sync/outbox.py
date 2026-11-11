"""Append-only attempt revision 1, atomic outbox and cumulative ACK progress.

Results are immutable: new experiments get new attempt IDs, never overwrite an
acknowledged experiment. Revision is the local monotonically increasing outbox
sequence. Source ID identifies this database's replication epoch.
"""
from contextlib import closing
import json
import uuid

from ..storage.repository import _connect
from ..storage.records import canonical, digest, normalize


def initialize(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS sync_identity (id INTEGER PRIMARY KEY CHECK(id=1), source_id TEXT NOT NULL, ack INTEGER NOT NULL DEFAULT 0)')
    conn.execute('INSERT OR IGNORE INTO sync_identity(id,source_id) VALUES (1,?)', (str(uuid.uuid4()),))
    conn.execute('CREATE TABLE IF NOT EXISTS sync_outbox (revision INTEGER PRIMARY KEY AUTOINCREMENT, attempt_id TEXT NOT NULL UNIQUE REFERENCES attempts(id))')


def enqueue(conn, attempt_id):
    initialize(conn)
    conn.execute('INSERT OR IGNORE INTO sync_outbox(attempt_id) VALUES (?)', (attempt_id,))


def pending(path, limit=50):
    if not 1 <= limit <= 100:
        raise ValueError('batch limit must be 1..100')
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        initialize(conn)
        # Backfill results created before sync support or restored from snapshots.
        conn.execute('INSERT INTO sync_outbox(attempt_id) SELECT id FROM attempts WHERE NOT EXISTS (SELECT 1 FROM sync_outbox WHERE attempt_id=attempts.id) ORDER BY rowid')
        source, ack = conn.execute('SELECT source_id,ack FROM sync_identity WHERE id=1').fetchone()
        rows = conn.execute('SELECT revision,attempt_id FROM sync_outbox WHERE revision>? ORDER BY revision LIMIT ?', (ack,limit)).fetchall()
        conn.row_factory = __import__('sqlite3').Row
        batch=[]
        for revision, identity in rows:
            row = dict(conn.execute('SELECT * FROM attempts WHERE id=?',(identity,)).fetchone())
            contexts={kind:json.loads(conn.execute('SELECT payload FROM contexts WHERE id=?',(row[kind+'_id'],)).fetchone()[0]) for kind in ('execution','evaluation','environment')}
            shapes=[dict(r) for r in conn.execute('SELECT * FROM shapes WHERE attempt_id=? ORDER BY rowid',(identity,))]
            record=normalize(dict(row, execution=contexts['execution'],evaluation=contexts['evaluation'],shape_results=shapes),contexts['environment'])
            if digest(record)!=row['result_hash']:
                raise ValueError('local result corruption; refusing upload')
            batch.append({'revision':revision,'attempt_id':identity,'created_at':row['created_at'],'record':record})
        return {'protocol':1,'source_id':source,'after':ack,'items':batch}


def acknowledge(path, source, revision):
    with closing(_connect(path,writable=True)) as conn,conn:
        conn.execute('BEGIN IMMEDIATE')
        identity=conn.execute('SELECT source_id,ack FROM sync_identity WHERE id=1').fetchone()
        maximum=conn.execute('SELECT coalesce(max(revision),0) FROM sync_outbox').fetchone()[0]
        if identity is None or source!=identity[0] or type(revision) is not int or not identity[1]<=revision<=maximum:
            raise ValueError('invalid sync ACK')
        completed = [r[0] for r in conn.execute('SELECT attempt_id FROM sync_outbox WHERE revision>? AND revision<=?', (identity[1], revision))]
        conn.execute('UPDATE sync_identity SET ack=? WHERE id=1',(revision,))
    from ..storage.lifecycle import append, stamp
    observed = stamp()
    for attempt_id in completed:
        append(path, attempt_id, 'uploaded_ack', observed)


def reset_epoch(path):
    """Explicit recovery after restoring an older central backup; retain all results.

    A new source replays immutable attempt IDs. The central owner/content checks
    deduplicate already present records; missing records are restored normally.
    Caller must hold the upload lock so no old-epoch request can race this reset.
    """
    with closing(_connect(path, writable=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        initialize(conn)
        source = str(uuid.uuid4())
        conn.execute('UPDATE sync_identity SET source_id=?,ack=0 WHERE id=1', (source,))
        from .facts import rebuild_outbox
        from ..storage.stage_facts import initialize as init_facts
        init_facts(conn)
        conn.execute('DELETE FROM fact_published_v1')
        conn.execute('DELETE FROM fact_outbox_v1')
        conn.execute('UPDATE fact_ack_v1 SET ack=0')
        rebuild_outbox(conn)
        return {'source_id': source, 'ack': 0, 'recovery': 'full_idempotent_resync'}
