"""Offline, transactional reclassification with reversible audit records.

Call only on a stopped server or an isolated consistent snapshot.
"""
import json
from autofusex.server import aggregates
from autofusex.server.store import get_content
from autofusex.server.failure_attribution import classify, VERSION

FIELDS = ('function_status','fail_stage','failure_owner','classifier_version')

def refresh(store, batch_id, *, apply=False):
    changes=[]
    with store.write() as conn:
        if conn.execute('SELECT 1 FROM classification_changes WHERE batch_id=?',(batch_id,)).fetchone():
            raise ValueError('batch already exists; use a new batch or rollback')
        for row in conn.execute('SELECT * FROM executions').fetchall():
            if row['classifier_version']==VERSION: continue
            raw=get_content(conn,row['raw_upload_content_id']) if row['raw_upload_content_id'] else None
            original=json.loads(raw) if raw else {}
            # Old records lack original declared labels: preserve the pre-refresh
            # snapshot in the audit; do not pretend these are original client facts.
            result=classify(backend=row['backend'],
                declared_status=original.get('function_status') or row['reported_status'] or row['function_status'],
                declared_stage=original.get('fail_stage',row['reported_stage'] if row['reported_status'] else row['fail_stage']),
                summary=original.get('log_summary',row['log_summary']),
                log=original.get('log') if raw else get_content(conn,row['log_content_id']) if row['log_content_id'] else None,
                detail=original.get('detail') if raw else get_content(conn,row['detail_content_id']) if row['detail_content_id'] else None)
            before=[row[k] for k in FIELDS];after=list(result)
            if before==after:continue
            changes.append({'exec_id':row['exec_id'],'before':before,'after':after})
        if apply:
            for change in changes:
                conn.execute('INSERT INTO classification_changes VALUES(?,?,?,?)',
                    (batch_id,change['exec_id'],json.dumps(change['before']),json.dumps(change['after'])))
                conn.execute('UPDATE executions SET function_status=?,fail_stage=?,failure_owner=?,classifier_version=? WHERE exec_id=?',
                    (*change['after'],change['exec_id']))
            aggregates.rebuild(conn,store.setting(conn,'perf.tie_threshold'))
            aggregates.advance_seq(conn)
    return changes

def rollback(store,batch_id):
    with store.write() as conn:
        rows=conn.execute('SELECT * FROM classification_changes WHERE batch_id=?',(batch_id,)).fetchall()
        if not rows:raise ValueError('unknown batch')
        for row in rows:
            current=conn.execute('SELECT function_status,fail_stage,failure_owner,classifier_version FROM executions WHERE exec_id=?',(row['exec_id'],)).fetchone()
            if list(current)!=json.loads(row['after_json']):raise ValueError('classification changed after batch; refusing rollback')
        for row in rows:
            conn.execute('UPDATE executions SET function_status=?,fail_stage=?,failure_owner=?,classifier_version=? WHERE exec_id=?',
                         (*json.loads(row['before_json']),row['exec_id']))
        conn.execute('DELETE FROM classification_changes WHERE batch_id=?',(batch_id,))
        aggregates.rebuild(conn,store.setting(conn,'perf.tie_threshold'));aggregates.advance_seq(conn)
    return len(rows)
