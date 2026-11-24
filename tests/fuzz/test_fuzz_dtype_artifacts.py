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
    (float('inf'), float('-inf'), False), (float('nan'), float('nan'), False),
    (float('inf'), 0., False),
])
def test_fuzz_nonfinite_outputs(torch_adapter, actual, reference, passed):
    torch = torch_adapter.torch
    actual, reference = (torch.tensor([actual]),), (torch.tensor([reference]),)
    assert torch_adapter.compare(actual, reference) is passed
    assert torch_adapter.diagnose_accuracy(actual, reference)['mismatches'] == int(not passed)


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
