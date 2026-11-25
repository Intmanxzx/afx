"""Offline, transactional reclassification with reversible audit records.

Call only on a stopped server or an isolated consistent snapshot.
"""
import json
from autofusex.server import aggregates
from autofusex.server.store import get_content
from autofusex.server.failure_attribution import classify, local_aclnn_failure, frontend_failure, nnsmith_failure, VERSION

FIELDS = ('function_status','fail_stage','failure_owner','classifier_version')


def aclnn_plan(conn):
    """Preview only records with complete shape-local aclNN evidence."""
    changes = []
    for row in conn.execute(
            "SELECT e.*,c.case_key,(h.exec_id IS NOT NULL) AS is_head FROM executions e "
            "JOIN cases c ON c.case_id=e.case_id LEFT JOIN execution_heads h "
            "ON h.case_id=e.case_id AND h.backend=e.backend AND h.npu_arch=e.npu_arch AND h.exec_id=e.exec_id "
            "WHERE e.function_status IN ('build_fail','gen_fail','ascgraph_fail','compile_fail','runtime_fail') "
            "AND COALESCE(e.fail_stage,'')<>'aclnn' AND e.detail_content_id IS NOT NULL ORDER BY e.exec_id"):
        result = local_aclnn_failure(get_content(conn, row['detail_content_id']), row['backend'])
        if result is not None:
            changes.append({'exec_id': row['exec_id'], 'exec_uuid': row['exec_uuid'],
                            'case_key': row['case_key'], 'backend': row['backend'], 'is_head': bool(row['is_head']),
                            'before': [row[k] for k in FIELDS], 'after': [*result, VERSION]})
    return changes


def frontend_plan(conn):
    """Preview PyAscIR inference failures with matching terminal evidence."""
    changes = []
    for row in conn.execute(
            "SELECT e.*,c.case_key,(h.exec_id IS NOT NULL) AS is_head FROM executions e "
            "JOIN cases c ON c.case_id=e.case_id LEFT JOIN execution_heads h "
            "ON h.case_id=e.case_id AND h.backend=e.backend AND h.npu_arch=e.npu_arch AND h.exec_id=e.exec_id "
            "WHERE e.backend IN ('af-legacy','af-mlir') "
            "AND e.function_status IN ('build_fail','gen_fail','ascgraph_fail','compile_fail','runtime_fail') "
            "AND COALESCE(e.fail_stage,'') NOT IN ('af_fronted','aclnn') ORDER BY e.exec_id"):
        detail = get_content(conn, row['detail_content_id']) if row['detail_content_id'] else None
        # Canonical native ERROR lines are stored in log_summary and detail.plog_errors.
        result = frontend_failure(row['backend'], row['log_summary'], detail)
        if result is not None:
            changes.append({'exec_id': row['exec_id'], 'exec_uuid': row['exec_uuid'],
                            'case_key': row['case_key'], 'backend': row['backend'], 'is_head': bool(row['is_head']),
                            'before': [row[k] for k in FIELDS], 'after': [*result, VERSION]})
    return changes


def apply_aclnn(store, batch_id, changes):
    """Apply a reviewed, scoped manifest with stale-evidence checks."""
    return _apply_scoped(store, batch_id, changes, 'aclnn')


def apply_frontend(store, batch_id, changes):
    return _apply_scoped(store, batch_id, changes, 'frontend')


def nnsmith_plan(conn):
    changes = []
    for row in conn.execute(
            "SELECT e.*,c.case_key,(h.exec_id IS NOT NULL) AS is_head FROM executions e "
            "JOIN cases c ON c.case_id=e.case_id LEFT JOIN execution_heads h "
            "ON h.case_id=e.case_id AND h.backend=e.backend AND h.npu_arch=e.npu_arch AND h.exec_id=e.exec_id "
            "WHERE e.function_status IN ('build_fail','gen_fail','ascgraph_fail','compile_fail','runtime_fail') "
            "AND COALESCE(e.fail_stage,'')<>'nnsmith' ORDER BY e.exec_id"):
        detail = get_content(conn, row['detail_content_id']) if row['detail_content_id'] else None
        result = nnsmith_failure(row['log_summary'], detail)
        if result:
            changes.append({'exec_id': row['exec_id'], 'exec_uuid': row['exec_uuid'],
                            'case_key': row['case_key'], 'backend': row['backend'], 'is_head': bool(row['is_head']),
                            'before': [row[k] for k in FIELDS], 'after': [*result, VERSION]})
    return changes


def apply_nnsmith(store, batch_id, changes):
    return _apply_scoped(store, batch_id, changes, 'nnsmith')


def _apply_scoped(store, batch_id, changes, rule):
    with store.write() as conn:
        if conn.execute('SELECT 1 FROM classification_changes WHERE batch_id=?', (batch_id,)).fetchone():
            raise ValueError('batch already exists')
        for change in changes:
            row = conn.execute('SELECT * FROM executions WHERE exec_id=?', (change['exec_id'],)).fetchone()
            if (row is None or row['exec_uuid'] != change['exec_uuid']
                    or [row[k] for k in FIELDS] != change['before']):
                raise ValueError('record changed after preview')
            detail = get_content(conn, row['detail_content_id']) if row['detail_content_id'] else None
            if rule == 'aclnn':
                result = local_aclnn_failure(detail, row['backend'])
            elif rule == 'frontend':
                result = frontend_failure(row['backend'], row['log_summary'], detail)
            elif rule == 'nnsmith':
                result = nnsmith_failure(row['log_summary'], detail)
            else:
                raise ValueError('unknown classification rule')
            if result is None or change['after'] != [*result, VERSION]:
                raise ValueError('classification evidence changed after preview')
            conn.execute('INSERT INTO classification_changes VALUES(?,?,?,?)',
                         (batch_id, row['exec_id'], json.dumps(change['before']), json.dumps(change['after'])))
            conn.execute('UPDATE executions SET function_status=?,fail_stage=?,failure_owner=?,classifier_version=? WHERE exec_id=?',
                         (*change['after'], row['exec_id']))
        if changes:
            aggregates.rebuild(conn, store.setting(conn, 'perf.tie_threshold'))
            aggregates.advance_seq(conn)
    return len(changes)

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
