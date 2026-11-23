import pytest

from autofusex.execution.failure_details import diagnostic_view, error_summary


@pytest.mark.parametrize('cause', [
    'Check dtype failed for graph_hint/cast Cast; input_dtypes: [DT_INT64], output_dtypes: [DT_FLOAT]',
    'Infer dtype failed for graph_hint/floor Floor; input_dtypes: [DT_FLOAT] is not supported now',
    'Check shape failed for graph_hint/add',
    "error: 'arith.index_cast' op result must be signless-integer, but got 'ui8'",
])
def test_generated_source_does_not_hide_cause(cause):
    text = ('Failed to execute graph code:from autofuse.pyautofuse import ascir\n'
            + 'graph = generated_source\n' * 300 + cause + '\n'
            + 'Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace')
    assert error_summary(RuntimeError(text)) == cause
    # Historical rows retain a truncated summary but have the diagnostic tail.
    view = diagnostic_view({'shapes': {'0': {'error': text[-1600:]}}}, 0, text[:500])
    assert view['summary'] == cause


def test_concrete_mlir_error_precedes_generic_codegen_failure():
    cause = 'error: UnsupportedOperation: scalar inf is not a numeric literal'
    assert error_summary(RuntimeError('Codegen generate kernel failed or abort\n' + cause)) == cause


def test_summary_redacts_and_bounds_plain_exceptions():
    summary = error_summary(ValueError('token=private-value ' + 'x' * 900))
    assert summary.startswith('ValueError: token=[REDACTED]')
    assert 'private-value' not in summary
    assert len(summary) == 500
    assert error_summary(ValueError('bad input')) == 'ValueError: bad input'


def test_shared_log_does_not_override_shape_cause():
    view = diagnostic_view({'error': 'error: another shape failed',
                            'shapes': {'0': {'error': 'Check dtype failed for Cast'}}}, 0)
    assert view['summary'] == 'Check dtype failed for Cast'
    assert view['shared_lines'] == ['error: another shape failed']
