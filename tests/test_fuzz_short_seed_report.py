import json
import pytest
from autofusedebugger.fuzz.contracts import Seed, ExecutionConfig, EvaluationConfig
from autofusedebugger.fuzz.generation import workload_id
from fuzz_fixtures import bundle as materialize
from autofusedebugger.fuzz.storage import ResultRepository
from autofusedebugger.fuzz.storage.queries import ResultQueries


def test_short_seed_is_lossless_and_protocol_hashes_unchanged():
    assert Seed.parse(str(2*2**128+42)).short == 'C000000016'
    assert str(Seed.parse('C16')) == str(2*2**128+42)
    from autofusedebugger.fuzz.generation import materialize as generate
    with pytest.raises(ValueError): generate('A16')
    assert len(Seed(2*2**128 + 36**9-1).short) == 10
    for value in (2*2**128, 2*2**128+42, 3*2**128-1):
        assert Seed.parse(Seed(value).short).value == value
    for invalid in ('B-1','b16','D0','B'+'Z'*25):
        with pytest.raises(ValueError): Seed.parse(invalid)


def test_report_short_query_detail_and_no_heavy_import(tmp_path):
    repo=ResultRepository(tmp_path/'results')
    identity=repo.save({'seed':str(2*2**128+42),'workload_id':workload_id(materialize('C16')),
        'oracle_id':'ref/1','backend':'cpu','status':'ok',
        'shape_results':[{'shape_index':0,'correctness':'pass','performance':'not_requested'}]})
    query=ResultQueries(repo.path)
    assert query.page(seed='C16')['items'][0]['seed_short']=='C000000016'
    assert query.detail(identity)['replay_command']=='autofusedbg fuzz replay C000000016'


def test_mlir_fails_closed_without_runtime_evidence():
    pytest.importorskip("torch")
    from autofusedebugger.fuzz.backends import create_backend
    from autofusedebugger.fuzz.execution import execute
    result=execute(materialize('C16'),create_backend(ExecutionConfig('af-mlir','npu:0')),EvaluationConfig())
    assert result['status']=='unsupported'
    assert 'runtime verification failed' in result['error_summary']


def test_report_endpoint_csp_and_html_escapes_errors(tmp_path):
    import threading
    from urllib.request import urlopen
    from autofusedebugger.fuzz.central.server import create_server
    repo=ResultRepository(tmp_path/'results')
    repo.save({'seed':str(2*2**128+42),'workload_id':workload_id(materialize('C16')),
        'oracle_id':'ref/1','backend':'<img src=x onerror=alert(1)>','status':'error',
        'error_summary':'<script>alert(1)</script>','shape_results':[]})
    with create_server(repo.path,port=0) as server:
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            with urlopen(f'http://127.0.0.1:{server.server_port}/report?seed=B16') as response:
                page=response.read()
                assert response.status==200 and b'charts.js' in page
                assert "script-src 'self'" in response.headers['Content-Security-Policy']
                assert b'<script>alert(1)</script>' not in page
        finally: server.shutdown();thread.join()
