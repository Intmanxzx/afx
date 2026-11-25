"""Mutually exclusive overview buckets; stages never imply a second count."""
def category(status, stage, accuracy):
    if status == 'pass':
        return 'accuracy_fail' if accuracy == 'fail' else 'pass' if accuracy == 'pass' else 'unchecked'
    if status in {'build_fail', 'gen_fail'}: return 'gen_fail'
    if status == 'ascgraph_fail': return 'ascgraph_fail'
    if status == 'unsupported': return 'other'
    if status == 'timeout': return 'timeout'
    if stage == 'environment': return 'environment'
    if status == 'compile_fail':
        return stage if stage in {'schedule','legacy_codegen','mlir','triton_ascend','af_common'} else 'compile_unknown'
    if status == 'runtime_fail':
        return stage if stage in {'aclnn','fused_kernel'} else 'runtime_unknown'
    return 'unknown'
