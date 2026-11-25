import pytest

from autofusex.server.failure_attribution import classify


@pytest.mark.parametrize('stage', [None, 'schedule', 'mlir', 'inductor', 'aclnn',
                                  'environment', 'af_fronted', 'torch', 'unknown_retest'])
@pytest.mark.parametrize('status', ['compile_fail', 'runtime_fail', 'gen_fail', 'ascgraph_fail', 'build_fail'])
def test_client_stage_is_never_failure_evidence(stage, status):
    result = classify(backend='af-mlir', declared_status=status, declared_stage=stage,
                      summary='unresolved failure')
    assert result[:3] == ('gen_fail' if status == 'build_fail' else status,
                          'unknown_retest', 'unknown')


@pytest.mark.parametrize('error', [
    'ValueError: missing compile APIs: build_tiling_obj, build_kernel_obj, build_kern',
    "ModuleNotFoundError: No module named 'autofuse.compile_adapter'",
    "ModuleNotFoundError: No module named 'asc_op_compile_base'",
    "ModuleNotFoundError: No module named 'tbe'",
    "InductorError: Failed to execute graph code: No module named 'autofuse.pyautofuse'",
    'UnsupportedBackend: Autofuse compile runtime is incomplete (selected package: None)',
])
@pytest.mark.parametrize('source', ['summary', 'execution', 'shape'])
def test_compile_environment_is_proven_on_server(error, source):
    summary = error if source == 'summary' else None
    detail = ({'error_summary': error} if source == 'execution' else
              {'shape_results': [{'shape_index': 0, 'correctness': 'error',
                                  'error_summary': error}]} if source == 'shape' else None)
    assert classify(backend='af-legacy', declared_status='compile_fail',
                    declared_stage='schedule', summary=summary, detail=detail)[:3] == (
                        'runtime_fail', 'environment', 'infrastructure')


def test_shared_environment_error_does_not_override_local_unknown():
    assert classify(backend='af-mlir', declared_status='compile_fail', declared_stage='schedule',
                    summary='unresolved failure',
                    log="ModuleNotFoundError: No module named 'autofuse.compile_adapter'")[:2] == (
                        'compile_fail', 'unknown_retest')


def test_server_evidence_ignores_client_retest_label():
    assert classify(backend='af-mlir', declared_status='runtime_fail',
                    declared_stage='unknown_retest', summary='aclnnSign failed')[:2] == ('runtime_fail', 'aclnn')
