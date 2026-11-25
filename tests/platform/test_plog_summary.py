from autofusex.server.plog_summary import extract
from test_platform import env, body, item, upload
from autofusex.server import queries


def test_summary_includes_python_and_native_errors():
    assert extract(log='RuntimeError: bad') == 'RuntimeError: bad'
    assert extract({'error':'[ERROR] Python failure'}) == '[ERROR] Python failure'
    assert extract({'plog_errors':['[INFO] ignored','[ERROR] CANN first','[ERROR] second']})=='[ERROR] CANN first'
    assert extract(log=chr(10).join(['InductorError: wrapper','','# CANN plog (ERROR)','[ERROR] historical']))=='[ERROR] historical'


def test_discovery_and_history_include_all_error_sources(env):
    store, ingestor, principal=env
    upload(ingestor,principal,body([
        item('summary-one',status='compile_fail',log_summary='Python wrapper',detail={'plog_errors':['[ERROR] original CANN line']}),
        item('summary-two',status='runtime_fail',log_summary='RuntimeError: not plog'),
    ]))
    group=queries.case_page(store)['groups'][0]
    rows={r['exec_uuid']:r for r in group['executions']}
    assert rows['summary-one']['log_summary']=='[ERROR] original CANN line'
    assert rows['summary-two']['log_summary'] == 'RuntimeError: not plog'
    history=queries.case_history(store,group['case']['case_key'])
    assert {r['exec_uuid']:r['log_summary'] for r in history['executions']}=={'summary-one':'[ERROR] original CANN line','summary-two':'RuntimeError: not plog'}
