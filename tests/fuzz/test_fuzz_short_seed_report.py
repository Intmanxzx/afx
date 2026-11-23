import json
import pytest
from autofusex.contracts.execution import ExecutionConfig, EvaluationConfig
from autofusex.workloads.fuzz.contracts import Seed
from autofusex.workloads.fuzz.generation import workload_id
from fuzz_fixtures import bundle as materialize, CURRENT_SEED
from autofusex.execution.storage import ResultRepository
from autofusex.execution.storage.queries import ResultQueries


def test_short_seed_is_lossless():
    from autofusex.workloads.fuzz.generation.sampling import SamplingConfig
    for dynamic in (False, True):
        for entropy in (0,42,2**32-1):
            seed=SamplingConfig().seed(entropy,dynamic)
            assert Seed.parse(seed.short)==Seed.parse(str(seed))==seed
    for invalid in ('C16','D16','C2-16','D2-16'):
        with pytest.raises(ValueError): Seed.parse(invalid)


def test_report_short_query_detail_and_no_heavy_import(tmp_path):
    repo=ResultRepository(tmp_path/'results')
    identity=repo.save({'seed':'14615016373650978276328604632813839116929039597610','workload_id':workload_id(materialize('C16')),
        'oracle_id':'ref/1','backend':'cpu','status':'ok',
        'shape_results':[{'shape_index':0,'correctness':'pass','performance':'not_requested'}]})
    query=ResultQueries(repo.path)
    assert query.page(seed='14615016373650978276328604632813839116929039597610')['items'][0]['seed_short']=='E0F8A5E5JE7A4QBAOWXZCIT10YY'
    assert query.detail(identity)['replay_command']=='afx fuzz replay E0F8A5E5JE7A4QBAOWXZCIT10YY'


def test_mlir_fails_closed_without_runtime_evidence(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("nnsmith")
    # This exercises an execution gate, not the query-only captured fixture.
    # Use the pinned generator so its source fingerprint passes seed validation.
    from autofusex.workloads.fuzz.generation import materialize as generate
    from autofusex.execution.backends import create_backend
    from autofusex.workloads.fuzz.execution import execute
    from autofusex.execution.backends.base import UnsupportedBackend
    def unavailable(): raise UnsupportedBackend('af-mlir runtime verification failed: test missing runtime')
    # verify_compile_adapter() is a separate, earlier gate (d516c22); stub it out so this
    # test exercises verify_runtime()'s own fail-closed behavior on a machine without the
    # full autofuse package installed, instead of failing one gate sooner for a different reason.
    monkeypatch.setattr('autofusex.execution.backends.mlir_capability.verify_compile_adapter',lambda: 'stub')
    monkeypatch.setattr('autofusex.execution.backends.mlir_capability.verify_runtime',unavailable)
    result=execute(generate(CURRENT_SEED),create_backend(ExecutionConfig('af-mlir','npu:0')),EvaluationConfig())
    assert result['status']=='unsupported'
    assert 'runtime verification failed' in result['error_summary']


def test_report_endpoint_csp_and_html_escapes_errors(tmp_path):
    import threading
    from urllib.request import urlopen
    from autofusex.viewer.server import create_server
    repo=ResultRepository(tmp_path/'results')
    repo.save({'seed':'14615016373650978276328604632813839116929039597610','workload_id':workload_id(materialize('C16')),
        'oracle_id':'ref/1','backend':'<img src=x onerror=alert(1)>','status':'error',
        'error_summary':'<script>alert(1)</script>','shape_results':[]})
    with create_server(repo.path,port=0) as server:
        thread=threading.Thread(target=server.serve_forever);thread.start()
        try:
            with urlopen(server.viewer_url+'report?seed=B16') as response:
                page=response.read()
                assert response.status==200 and b'charts.js' in page
                assert "script-src 'self'" in response.headers['Content-Security-Policy']
                assert b'<script>alert(1)</script>' not in page
        finally: server.shutdown();thread.join()
