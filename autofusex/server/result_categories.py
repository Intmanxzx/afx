"""Mutually exclusive overview buckets; stages never imply a second count."""
def category(status, stage, accuracy):
    if stage == 'unknown_retest' and status not in {'pass', 'unsupported', 'timeout'}:
        return 'unknown_retest'
    if status == 'pass':
        return 'accuracy_fail' if accuracy == 'fail' else 'pass' if accuracy == 'pass' else 'unchecked'
    if status in {'build_fail', 'gen_fail'}:
        return 'nnsmith' if stage == 'nnsmith' else 'gen_fail'
    if status == 'ascgraph_fail': return 'ascgraph_fail'
    if status == 'unsupported': return 'other'
    if status == 'timeout': return 'timeout'
    if stage == 'environment': return 'environment'
    if status == 'compile_fail':
        return stage if stage in {'schedule','legacy_codegen','mlir','triton_ascend','af_common'} else 'compile_unknown'
    if status == 'runtime_fail':
        return stage if stage in {'aclnn','aicpu','fused_kernel'} else 'runtime_unknown'
    return 'unknown'


RESULT_KEYS = {'nnsmith', 'gen_fail', 'ascgraph_fail', 'schedule', 'legacy_codegen', 'mlir',
               'triton_ascend', 'af_common', 'compile_unknown', 'aclnn', 'aicpu', 'fused_kernel',
               'runtime_unknown', 'pass', 'accuracy_fail', 'unchecked', 'other', 'environment',
               'timeout', 'unknown', 'unknown_retest'}


def category_sql():
    """SQL equivalent of category(), including its precedence and NULL handling."""
    return """CASE
      WHEN e.fail_stage='unknown_retest' AND e.function_status NOT IN ('pass','unsupported','timeout') THEN 'unknown_retest'
      WHEN e.function_status='pass' THEN CASE e.accuracy_status WHEN 'fail' THEN 'accuracy_fail' WHEN 'pass' THEN 'pass' ELSE 'unchecked' END
      WHEN e.function_status IN ('build_fail','gen_fail') THEN CASE WHEN e.fail_stage='nnsmith' THEN 'nnsmith' ELSE 'gen_fail' END
      WHEN e.function_status='ascgraph_fail' THEN 'ascgraph_fail'
      WHEN e.function_status='unsupported' THEN 'other'
      WHEN e.function_status='timeout' THEN 'timeout'
      WHEN e.fail_stage='environment' THEN 'environment'
      WHEN e.function_status='compile_fail' THEN CASE WHEN e.fail_stage IN ('schedule','legacy_codegen','mlir','triton_ascend','af_common') THEN e.fail_stage ELSE 'compile_unknown' END
      WHEN e.function_status='runtime_fail' THEN CASE WHEN e.fail_stage IN ('aclnn','aicpu','fused_kernel') THEN e.fail_stage ELSE 'runtime_unknown' END
      ELSE 'unknown' END"""
