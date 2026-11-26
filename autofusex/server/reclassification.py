"""Offline, transactional reclassification with reversible audit records.

Call only on a stopped server or an isolated consistent snapshot.
"""
import json
from autofusex.server import aggregates
from autofusex.server.issues import refresh_classified_executions
from autofusex.server.store import get_content
from autofusex.server.triton_attribution import triton_failure
from autofusex.server.native_runtime_attribution import native_runtime_failure
from autofusex.server.local_failure_attribution import local_failure
from autofusex.server.mlir_attribution import mlir_failure
from autofusex.server.failure_attribution import classify, local_aclnn_failure, frontend_failure, nnsmith_failure, VERSION

FIELDS = ('function_status','fail_stage','failure_owner','classifier_version')

def mlir_plan(conn):
    """Preview AFIR diagnostics only on currently unowned AF failures."""
    changes = []
    for row in conn.execute(
            "SELECT e.*,c.case_key,(h.exec_id IS NOT NULL) AS is_head FROM executions e "
            "JOIN cases c ON c.case_id=e.case_id LEFT JOIN execution_heads h ON h.exec_id=e.exec_id "
            "WHERE e.backend IN ('af-legacy','af-mlir') "
            "AND e.function_status IN ('gen_fail','ascgraph_fail','compile_fail','runtime_fail','build_fail') "
            "AND COALESCE(e.fail_stage,'') IN ('','unknown','unknown_retest') ORDER BY e.exec_id"):
        detail = get_content(conn, row['detail_content_id']) if row['detail_content_id'] else None
        result = mlir_failure(row['backend'], row['log_summary'], detail)
        if result:
            changes.append({'exec_id': row['exec_id'], 'exec_uuid': row['exec_uuid'],
                            'case_key': row['case_key'], 'backend': row['backend'], 'is_head': bool(row['is_head']),
                            'before': [row[k] for k in FIELDS], 'after': [*result, VERSION]})
    return changes


def apply_mlir(store, batch_id, changes):
    return _apply_scoped(store, batch_id, changes, 'mlir')


def local_failure_plan(conn):
    """Preview only approved local-evidence rules on unknown-retest failures."""
    changes = []
    for row in conn.execute(
            "SELECT e.*,c.case_key,(h.exec_id=e.exec_id) AS is_head FROM executions e "
            "JOIN cases c ON c.case_id=e.case_id LEFT JOIN execution_heads h "
            "ON h.case_id=e.case_id AND h.backend=e.backend AND h.npu_arch=e.npu_arch "
            "WHERE e.fail_stage='unknown_retest' AND e.function_status IN "
            "('build_fail','gen_fail','ascgraph_fail','compile_fail','runtime_fail') ORDER BY e.exec_id"):
        detail = get_content(conn, row['detail_content_id']) if row['detail_content_id'] else None
        result = local_failure(detail, row['backend'])
        if result:
            changes.append({'exec_id': row['exec_id'], 'exec_uuid': row['exec_uuid'],
                            'case_key': row['case_key'], 'backend': row['backend'],
                            'npu_arch': row['npu_arch'], 'is_head': bool(row['is_head']),
                            'before': [row[k] for k in FIELDS], 'after': [*result, VERSION]})
    return changes


def apply_local_failure(store, batch_id, changes):
    return _apply_scoped(store, batch_id, changes, 'local_failure')


def triton_plan(conn):
    changes = []
    for row in conn.execute(
            "SELECT e.*,c.case_key,(h.exec_id IS NOT NULL) AS is_head FROM executions e "
            "JOIN cases c ON c.case_id=e.case_id LEFT JOIN execution_heads h ON h.exec_id=e.exec_id "
            "WHERE e.backend='triton-npu' AND e.function_status IN ('gen_fail','compile_fail','runtime_fail') "
            "AND COALESCE(e.fail_stage,'') IN ('','unknown','unknown_retest','inductor') ORDER BY e.exec_id"):
        detail = get_content(conn, row['detail_content_id']) if row['detail_content_id'] else None
        result = triton_failure(row['backend'], row['log_summary'], detail)
        if result:
            changes.append({'exec_id': row['exec_id'], 'exec_uuid': row['exec_uuid'],
                            'case_key': row['case_key'], 'backend': row['backend'], 'is_head': bool(row['is_head']),
                            'before': [row[k] for k in FIELDS], 'after': [*result, VERSION]})
    return changes


def apply_triton(store, batch_id, changes):
    return _apply_scoped(store, batch_id, changes, 'triton')


def native_runtime_plan(conn):
    """Correct only unowned failures with attempt-local native runtime evidence."""
    changes = []
    for row in conn.execute(
            "SELECT e.*,c.case_key,(h.exec_id IS NOT NULL) AS is_head FROM executions e "
            "JOIN cases c ON c.case_id=e.case_id LEFT JOIN execution_heads h ON h.exec_id=e.exec_id "
            "WHERE e.function_status IN ('gen_fail','compile_fail','runtime_fail','timeout') "
            "AND COALESCE(e.fail_stage,'') IN ('','unknown','unknown_retest','environment','af_common') ORDER BY e.exec_id"):
        detail = get_content(conn, row['detail_content_id']) if row['detail_content_id'] else None
        result = native_runtime_failure(detail, row['backend'])
        if result:
            changes.append({'exec_id': row['exec_id'], 'exec_uuid': row['exec_uuid'],
                            'case_key': row['case_key'], 'backend': row['backend'], 'is_head': bool(row['is_head']),
                            'before': [row[k] for k in FIELDS], 'after': [*result, VERSION]})
    return changes


def apply_native_runtime(store, batch_id, changes):
    return _apply_scoped(store, batch_id, changes, 'native_runtime')


def aclnn_plan(conn):
    """Preview only records with complete shape-local aclNN evidence."""
    changes = []
    for row in conn.execute(
            "SELECT e.*,c.case_key,(h.exec_id IS NOT NULL) AS is_head FROM executions e "
            "JOIN cases c ON c.case_id=e.case_id LEFT JOIN execution_heads h "
            "ON h.case_id=e.case_id AND h.backend=e.backend AND h.npu_arch=e.npu_arch AND h.exec_id=e.exec_id "
            "WHERE e.function_status IN ('build_fail','gen_fail','ascgraph_fail','compile_fail','runtime_fail','timeout') "
            "AND COALESCE(e.fail_stage,'')<>'aclnn' AND e.detail_content_id IS NOT NULL ORDER BY e.exec_id"):
        detail = get_content(conn, row['detail_content_id'])
        result = native_runtime_failure(detail, row['backend'])
        if result and result[1] != 'aclnn':
            result = None
        if result is None and row['function_status'] != 'timeout':
            result = local_aclnn_failure(detail, row['backend'])
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

def apply_rule_changes(store, batch_id, changes):
    """Apply administrator-reviewed log-rule matches in one audited transaction."""
    with store.write() as conn:
        if conn.execute('SELECT 1 FROM classification_changes WHERE batch_id=?', (batch_id,)).fetchone():
            raise ValueError('batch already exists; use a new batch or rollback')
        for change in changes:
            row = conn.execute('SELECT * FROM executions WHERE exec_id=?', (change['exec_id'],)).fetchone()
            if row is None or row['exec_uuid'] != change['exec_uuid'] or [row[k] for k in FIELDS] != change['before']:
                raise ValueError('record changed after planning')
            conn.execute('INSERT INTO classification_changes VALUES(?,?,?,?)',
                         (batch_id, row['exec_id'], json.dumps(change['before']), json.dumps(change['after'])))
            conn.execute('UPDATE executions SET function_status=?,fail_stage=?,failure_owner=?,classifier_version=? WHERE exec_id=?',
                         (*change['after'], row['exec_id']))
        if changes:
            refresh_classified_executions(conn, [c['exec_id'] for c in changes])
            aggregates.rebuild(conn, store.setting(conn, 'perf.tie_threshold'))
            aggregates.advance_seq(conn)
    return len(changes)


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
            if rule == 'mlir':
                if (row['backend'] not in ('af-legacy', 'af-mlir')
                        or row['function_status'] not in
                        ('gen_fail', 'ascgraph_fail', 'compile_fail', 'runtime_fail', 'build_fail')
                        or row['fail_stage'] not in (None, '', 'unknown', 'unknown_retest')):
                    raise ValueError('record is outside MLIR correction scope')
                result = mlir_failure(row['backend'], row['log_summary'], detail)
            elif rule == 'local_failure':
                if (row['fail_stage'] != 'unknown_retest' or row['function_status'] not in
                        ('build_fail', 'gen_fail', 'ascgraph_fail', 'compile_fail', 'runtime_fail')):
                    raise ValueError('record is outside local failure correction scope')
                result = local_failure(detail, row['backend'])
            elif rule == 'aclnn':
                result = native_runtime_failure(detail, row['backend'])
                if result and result[1] != 'aclnn':
                    result = None
                if result is None and row['function_status'] != 'timeout':
                    result = local_aclnn_failure(detail, row['backend'])
            elif rule == 'frontend':
                result = frontend_failure(row['backend'], row['log_summary'], detail)
            elif rule == 'nnsmith':
                result = nnsmith_failure(row['log_summary'], detail)
            elif rule == 'triton':
                if (row['backend'] != 'triton-npu'
                        or row['function_status'] not in ('gen_fail', 'compile_fail', 'runtime_fail')
                        or row['fail_stage'] not in (None, '', 'unknown', 'unknown_retest', 'inductor')):
                    raise ValueError('record is outside Triton correction scope')
                result = triton_failure(row['backend'], row['log_summary'], detail)
            elif rule == 'native_runtime':
                if (row['function_status'] not in ('gen_fail', 'compile_fail', 'runtime_fail', 'timeout')
                        or row['fail_stage'] not in (None, '', 'unknown', 'unknown_retest', 'environment', 'af_common')):
                    raise ValueError('record is outside native runtime correction scope')
                result = native_runtime_failure(detail, row['backend'])
            else:
                raise ValueError('unknown classification rule')
            if result is None or change['after'] != [*result, VERSION]:
                raise ValueError('classification evidence changed after preview')
            conn.execute('INSERT INTO classification_changes VALUES(?,?,?,?)',
                         (batch_id, row['exec_id'], json.dumps(change['before']), json.dumps(change['after'])))
            conn.execute('UPDATE executions SET function_status=?,fail_stage=?,failure_owner=?,classifier_version=? WHERE exec_id=?',
                         (*change['after'], row['exec_id']))
        if changes:
            refresh_classified_executions(conn, [c['exec_id'] for c in changes])
            aggregates.rebuild(conn, store.setting(conn, 'perf.tie_threshold'))
            aggregates.advance_seq(conn)
    return len(changes)

def refresh(store, batch_id, *, apply=False):
    # Planning is read-only and can be expensive because legacy uploads may
    # require loading compressed detail/raw blobs.  Do not hold SQLite's
    # IMMEDIATE writer lock during this scan; only the short apply phase needs
    # exclusive write access.
    changes=[]
    with store.connect(readonly=True) as conn:
        for row in conn.execute('SELECT * FROM executions'):
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
        with store.write() as conn:
            if conn.execute('SELECT 1 FROM classification_changes WHERE batch_id=?',(batch_id,)).fetchone():
                raise ValueError('batch already exists; use a new batch or rollback')
            for change in changes:
                row=conn.execute('SELECT * FROM executions WHERE exec_id=?',(change['exec_id'],)).fetchone()
                if row is None or [row[k] for k in FIELDS] != change['before']:
                    raise ValueError('record changed after refresh planning')
                conn.execute('INSERT INTO classification_changes VALUES(?,?,?,?)',
                    (batch_id,change['exec_id'],json.dumps(change['before']),json.dumps(change['after'])))
                conn.execute('UPDATE executions SET function_status=?,fail_stage=?,failure_owner=?,classifier_version=? WHERE exec_id=?',
                    (*change['after'],change['exec_id']))
            if changes:
                refresh_classified_executions(conn, [c['exec_id'] for c in changes])
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
        refresh_classified_executions(conn, [r['exec_id'] for r in rows])
        conn.execute('DELETE FROM classification_changes WHERE batch_id=?',(batch_id,))
        aggregates.rebuild(conn,store.setting(conn,'perf.tie_threshold'));aggregates.advance_seq(conn)
    return len(rows)
