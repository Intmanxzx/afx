import pytest
from test_platform import env
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


MLIR_MARKER = 'AF_MLIR_STRICT gate=none shortfall=no graph=autofused_mul_mean_clamp_min_clamp_max_e19390bf3f6e3ae4f1b13934fdedaed5'


def test_mlir_runtime_fallback_and_saved_rule_order():
    rules = merge_builtins([])
    rules = rules[-1:] + rules[:-1]
    text = 'RuntimeError: permute(sparse_coo): number of dimensions does not match\n' + MLIR_MARKER
    assert match(rules, 'af-mlir', text, current=('runtime_fail', 'unknown_retest'))[:3] == ('runtime_fail', 'fused_kernel', 'npu_runtime')
    assert match(rules, 'af-legacy', text) is None
    assert match(rules, 'af-mlir', MLIR_MARKER.replace('shortfall=no', 'shortfall=yes')) is None
    assert match(rules, 'af-mlir', MLIR_MARKER + '\nTimeoutError: timed out')[:3] == ('timeout', 'timeout', 'npu_runtime')
    assert match(rules, 'af-mlir', MLIR_MARKER + "\nerror: failed to legalize operation 'afir.broadcast'")[:3] == ('compile_fail', 'mlir', 'autofuse')
    assert match(rules, 'af-mlir', MLIR_MARKER + '\ntorch._inductor.exc.InductorError: UnsupportedIndexExpr: Unsupported index_expr')[:3] == ('gen_fail', 'inductor', 'torch')


@pytest.mark.parametrize('current', [
    ('timeout', 'unknown'), ('runtime_fail', 'environment'),
    ('runtime_fail', 'aclnn'), ('compile_fail', 'unknown_retest'),
    ('gen_fail', 'inductor'), ('pass', None),
])
def test_mlir_fallback_preserves_established_attribution(current):
    assert match(merge_builtins([]), 'af-mlir', MLIR_MARKER, current=current) is None


def test_mlir_fallback_ingest_and_historical_preview(env):
    from test_platform import body, item, upload
    from autofusex.server.classification_rules import plan
    store, ingestor, principal = env
    record = item('mlir-fallback-upload', backend='af-mlir', status='runtime_fail',
                  log_summary=MLIR_MARKER)
    upload(ingestor, principal, body([record]))
    row = store.reader().execute('SELECT exec_id,function_status,fail_stage FROM executions').fetchone()
    assert (row['function_status'], row['fail_stage']) == ('runtime_fail', 'fused_kernel')
    with store.write() as conn:
        conn.execute("UPDATE executions SET fail_stage='unknown_retest' WHERE exec_id=?", (row['exec_id'],))
    changes = plan(store, [])
    assert len(changes) == 1
    assert changes[0]['after'][:3] == ['runtime_fail', 'fused_kernel', 'npu_runtime']
