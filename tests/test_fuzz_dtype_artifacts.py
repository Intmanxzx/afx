import pytest


def test_bfloat16_stage_artifact_roundtrip(tmp_path):
    torch = pytest.importorskip('torch')
    from autofusedebugger.fuzz.scheduling.artifacts import write_arrays, read_arrays
    samples = [(torch.tensor([1., -2., 0.], dtype=torch.bfloat16),
                torch.tensor([1 + 2j], dtype=torch.complex64)),
               (torch.tensor(3, dtype=torch.int64),)]
    ref = write_arrays(tmp_path/'outputs.npz', samples)
    actual = read_arrays(tmp_path, ref)
    for xs, ys in zip(samples, actual):
        for x, y in zip(xs, ys):
            assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
