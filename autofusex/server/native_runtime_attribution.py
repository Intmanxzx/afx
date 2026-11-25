"""Native runtime ownership from failing shapes or attempt-scoped worker plog."""
import re

from autofusex.execution.module_attribution import classify_module

_AICPU = re.compile(r'aicpu execute failed|exception occurred during AICPU execution|'
                    r'AI CPU kernel (?:task )?execution failed|\baicpu exception\b|'
                    r'The aicpu execution is abnormal', re.I)


def native_runtime_failure(detail, backend):
    from autofusex.server.failure_attribution import _object, local_aclnn_failure
    d = _object(detail)
    execution = _object(_object(d.get('contexts')).get('execution') or d.get('execution'))
    diag = _object(execution.get('diagnostics'))
    shapes = _object(diag.get('shapes'))
    summaries = {str(s.get('shape_index')): s.get('error_summary') or ''
                 for s in d.get('shape_results', []) if isinstance(s, dict)
                 and s.get('correctness') == 'error'}
    indices = set(summaries) | {str(k) for k, v in shapes.items() if _object(v).get('error')}
    if indices:
        aclnn = local_aclnn_failure(d, backend)
        if aclnn:
            return aclnn
        for index in indices:
            text = summaries.get(index, '') + '\n' + str(_object(shapes.get(index)).get('error') or '')
            if classify_module(text, backend) or not _AICPU.search(text):
                return None
        return 'runtime_fail', 'aicpu', 'npu_runtime'
    if d.get('error_stage') != 'worker' or d.get('shape_results'):
        return None
    if classify_module(str(diag.get('error') or ''), backend):
        return None
    failed = [f for f in d.get('stage_timeline', []) if isinstance(f, dict)
              and f.get('state') == 'failed' and (f.get('lease') or {}).get('device_uuid')]
    if len(failed) != 1 or not failed[0].get('worker_pid'):
        return None
    pid = failed[0]['worker_pid']
    # The collector limits plog_errors to this attempt's PID/time window.
    # Match the failed worker again; never use arbitrary shared log strings.
    aicpu = False
    for line in d.get('plog_errors', []):
        if not isinstance(line, str):
            continue
        native = re.match(r'\[ERROR\] (OP|RUNTIME)\((\d+),', line)
        if not native or int(native[2]) != pid:
            continue
        if (native[1] == 'OP' and re.search(r'errno\[(?!0\])\d+\]', line)
                and re.search(r'OpName:\[aclnn\w+\]', line)):
            return 'runtime_fail', 'aclnn', 'npu_runtime'
        if native[1] == 'RUNTIME' and _AICPU.search(line):
            aicpu = True
    return ('runtime_fail', 'aicpu', 'npu_runtime') if aicpu else None
