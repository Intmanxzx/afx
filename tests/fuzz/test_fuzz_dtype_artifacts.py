import pytest


def test_bfloat16_stage_artifact_roundtrip(tmp_path):
    torch = pytest.importorskip('torch')
    from autofusex.workloads.fuzz.scheduling.artifacts import write_arrays, read_arrays
    samples = [(torch.tensor([1., -2., 0.], dtype=torch.bfloat16),
                torch.tensor([1 + 2j], dtype=torch.complex64)),
               (torch.tensor(3, dtype=torch.int64),)]
    ref = write_arrays(tmp_path/'outputs.npz', samples)
    actual = read_arrays(tmp_path, ref)
    for xs, ys in zip(samples, actual):
        for x, y in zip(xs, ys):
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)


@pytest.fixture
def torch_adapter():
    from autofusex.workloads.fuzz.torch_backend import TorchAdapter
    from autofusex.contracts.execution import ExecutionConfig
    adapter = TorchAdapter(ExecutionConfig('cpu', 'cpu'))
    adapter.torch = pytest.importorskip('torch')
    return adapter


@pytest.mark.parametrize('dtype,tolerance', [
    ('float64', 1e-7), ('float32', 1e-5), ('float16', 1e-3), ('bfloat16', 1e-2),
])
@pytest.mark.parametrize('serialized', [False, True])
def test_fuzz_dtype_tolerance(torch_adapter, dtype, tolerance, serialized):
    from types import SimpleNamespace
    adapter = torch_adapter
    torch = adapter.torch
    reference = torch.tensor([0., 100.], dtype=getattr(torch, dtype))
    expected = ((SimpleNamespace(shape=(2,), dtype=dtype),
                 reference.view(torch.uint8).numpy().tobytes()),) if serialized else (reference,)
    for scale, passed in [(0.5, True), (4., False)]:
        actual = (torch.tensor([scale * tolerance, 100. + scale * tolerance * 100.],
                               dtype=reference.dtype),)
        assert adapter.compare(actual, expected) is passed
        assert adapter.diagnose_accuracy(actual, expected)['mismatches'] == (0 if passed else 2)


@pytest.mark.parametrize('dtype,values', [('int64', [2**60, 2**60 + 1]), ('bool', [False, True])])
def test_fuzz_discrete_outputs_remain_exact(torch_adapter, dtype, values):
    torch = torch_adapter.torch
    left, right = [(torch.tensor([value], dtype=getattr(torch, dtype)),) for value in values]
    assert torch_adapter.compare(left, left)
    assert not torch_adapter.compare(left, right)
    assert torch_adapter.diagnose_accuracy(left, right)['mismatches'] == 1


@pytest.mark.parametrize('actual,reference,passed', [
    (float('inf'), float('inf'), True), (float('-inf'), float('-inf'), True),
    (float('inf'), float('-inf'), False), (float('nan'), float('nan'), True),
    (float('nan'), 0., False), (0., float('nan'), False),
    (float('nan'), float('inf'), False),
    (float('inf'), 0., False),
])
def test_fuzz_nonfinite_outputs(torch_adapter, actual, reference, passed):
    torch = torch_adapter.torch
    actual, reference = (torch.tensor([actual]),), (torch.tensor([reference]),)
    assert torch_adapter.compare(actual, reference) is passed
    assert torch_adapter.diagnose_accuracy(actual, reference)['mismatches'] == int(not passed)


@pytest.mark.parametrize('actual,reference,passed', [
    (complex(float('nan'), 2), complex(float('nan'), 2), True),
    (complex(float('nan'), 3), complex(float('nan'), 2), False),
    (complex(2, float('nan')), complex(2, float('nan')), True),
    (complex(float('inf'), 2), complex(float('inf'), 2), True),
    (complex(float('inf'), 3), complex(float('inf'), 2), False),
    (complex(float('-inf'), 2), complex(float('inf'), 2), False),
    (complex(float('nan'), 2), complex(2, float('nan')), False),
])
def test_complex_nonfinite_alignment(torch_adapter, actual, reference, passed):
    torch = torch_adapter.torch
    actual, reference = (torch.tensor([actual], dtype=torch.complex64),), (torch.tensor([reference], dtype=torch.complex64),)
    assert torch_adapter.compare(actual, reference) is passed
    metrics = torch_adapter.diagnose_accuracy(actual, reference)
    assert metrics['mismatches'] == int(not passed)
    assert metrics['outputs'][0]['nan_mismatches'] == int(not passed and (torch.isnan(actual[0]) | torch.isnan(reference[0])).any())


def test_nonfinite_rules_match_execution_compare(torch_adapter):
    import numpy as np
    from autofusex.execution.compare import _metrics_with_close, _elementwise_payload
    torch = torch_adapter.torch
    values = [float('nan'), float('inf'), float('-inf'), 0., 1.]
    for actual_value in values:
        for reference_value in values:
            actual = torch.tensor([actual_value], dtype=torch.float64)
            reference = torch.tensor([reference_value], dtype=torch.float64)
            expected = np.array([reference_value])
            observed = np.array([actual_value])
            metrics, close = _metrics_with_close(expected, observed, rtol=1e-7, atol=1e-7)
            assert bool(torch_adapter._close(actual, reference).all()) == metrics['strict_passed']
            assert close.all() == metrics['strict_passed']
            payload = _elementwise_payload(expected, observed, rtol=1e-7, atol=1e-7, logical_dtype='float64')
            import base64
            assert bool(base64.b64decode(payload['strict_mask']['data'])[0]) == metrics['strict_passed']


def test_fuzz_output_signature_remains_strict(torch_adapter):
    torch = torch_adapter.torch
    expected = (torch.ones(2),)
    assert not torch_adapter.compare((torch.ones(1),), expected)
    assert not torch_adapter.compare((torch.ones(2, dtype=torch.float64),), expected)
    assert not torch_adapter.compare((), expected)


def test_fuzz_tolerance_boundary(torch_adapter):
    torch = torch_adapter.torch
    reference = (torch.zeros(1, dtype=torch.float64),)
    boundary = torch.tensor([1e-7], dtype=torch.float64)
    outside = torch.nextafter(boundary, torch.full_like(boundary, float('inf')))
    assert torch_adapter.compare((boundary,), reference)
    assert not torch_adapter.compare((outside,), reference)
    assert torch_adapter.diagnose_accuracy((outside,), reference)['mismatches'] == 1
