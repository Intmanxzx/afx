import pytest
from autofusex.server.classification_rules import merge_builtins, match


@pytest.mark.parametrize('text,backend,expected', [
    ("torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_function aten.view_copy.default: got ValueError('Cannot view a tensor with shape ...')", 'af-legacy', ('gen_fail','nnsmith','nnsmith')),
    ('RuntimeError: RuntimeError: TimeoutError: timed out\nAF_MLIR_STRICT gate=none shortfall=no graph=autofused_minimum', 'af-mlir', ('timeout','timeout','npu_runtime')),
    ("/tmp/autofused_mean_tiling.cpp:8042:204: error: no member named 'get_ks3'\nAF_MLIR_STRICT gate=none shortfall=no graph=autofused_mean", 'af-mlir', ('compile_fail','mlir','autofuse')),
    ("error: failed to legalize operation 'afir.broadcast'", 'af-mlir', ('compile_fail','mlir','autofuse')),
    ("torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:\nTypeError: int_noop() got an unexpected keyword argument 'decimals'", 'triton-npu', ('gen_fail','inductor','torch')),
    ("torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:\nAttributeError: 'SymInt' object has no attribute 'size'", 'af-legacy', ('gen_fail','inductor','torch')),
    ('torch._inductor.exc.InductorError: UnsupportedIndexExpr: Unsupported index_expr (torch.float32): ToFloat(ks1); unknown SymPy node ToFloat', 'af-mlir', ('gen_fail','inductor','torch')),
])
def test_reported_signatures(text, backend, expected):
    assert match(merge_builtins([]), backend, text)[:3] == expected


@pytest.mark.parametrize('text,backend', [
    ('torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: different failure', 'af-mlir'),
    ("torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:\nRuntimeError: different failure", 'af-mlir'),
    ("/tmp/autofused_mean_tiling.cpp:1:2: error: no member named 'get_ks3'", 'af-mlir'),
    ("error: failed to legalize operation 'afir.broadcast'", 'triton-npu'),
    ('TimeoutError: timed out', 'cpu'),
])
def test_unrelated_errors_do_not_match(text, backend):
    assert match(merge_builtins([]), backend, text) is None
