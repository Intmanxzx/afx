from autofusex.server.plog_summary import extract
from test_platform import env, body, item, upload
from autofusex.server import queries


def test_summary_requires_explicit_plog_evidence():
    assert extract(log='RuntimeError: bad') is None
    assert extract({'error':'[ERROR] Python failure'}) is None
    assert extract({'plog_errors':['[INFO] ignored','[ERROR] CANN first','[ERROR] second']})=='[ERROR] CANN first'
    assert extract(log=chr(10).join(['RuntimeError: wrapper','','# CANN plog (ERROR)','[ERROR] historical']))=='[ERROR] historical'


def test_discovery_and_history_restore_plog_not_generic_summary(env):
    store, ingestor, principal=env
    upload(ingestor,principal,body([
        item('summary-one',status='compile_fail',log_summary='Python wrapper',detail={'plog_errors':['[ERROR] original CANN line']}),
        item('summary-two',status='runtime_fail',log_summary='RuntimeError: not plog'),
    ]))
    group=queries.case_page(store)['groups'][0]
    rows={r['exec_uuid']:r for r in group['executions']}
    assert rows['summary-one']['log_summary']=='[ERROR] original CANN line'
    assert rows['summary-two']['log_summary'] is None
    history=queries.case_history(store,group['case']['case_key'])
    assert {r['exec_uuid']:r['log_summary'] for r in history['executions']}=={'summary-one':'[ERROR] original CANN line','summary-two':None}
