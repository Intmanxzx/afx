"""Failure ownership proven by terminal exceptions and failing-shape frames."""
import re

from autofusex.execution.module_attribution import classify_module


_INDUCTOR = ('gen_fail', 'inductor', 'torch')
_ENVIRONMENT = ('runtime_fail', 'environment', 'infrastructure')
_FUSED_KERNEL = ('runtime_fail', 'fused_kernel', 'npu_runtime')
_AF_COMPILE = ('compile_fail', 'af_common', 'autofuse')
_AF_BACKENDS = {'af-mlir', 'af-legacy'}
_ATT_GE = re.compile(r'^\[ERROR\] GE\([^\n]*\).*\[[^\n]*autofuse/att/[^\n]*\].*\bErrorNo:\s*\d+\([^\n]*\)', re.M)
_VECTOR_CORE = re.compile(r'^\[Error\]: The vector core execution is abnormal\.\s*$', re.M)
_MLIR_KERNEL_BUILD = re.compile(r'AF_MLIR_STRICT\s+gate=none\s+shortfall=no\s+graph=autofused_[^\s]+.*RuntimeError:\s*Failed to build ascend kernel, trigger bug by following command:', re.S)
_ASCEND_KERNEL_BUILD = re.compile(
    r'(?:torch\._inductor\.exc\.)?InductorError:\s*RuntimeError:\s*'
    r'Failed to build ascend kernel, trigger bug by following command:', re.M)

# Match real traceback frame lines, not filenames mentioned in source strings.
_FRAME = re.compile(r'^\s*File "([^"\n]+)", line \d+, in ([^\n]+)$', re.M)
_CONFLICT = re.compile(
    r'AFX_MODULE_V1=|'
    r'^(?:[\w.]*Error|[\w.]*Exception):[^\n]*(?:'
    r'out of memory|no space left|timed out|device busy|'
    r'aicpu execute failed|AICPU execution|AI CPU kernel|'
    r'materialized Torch graph differs from declared NNSmith output signature)|'
    r'^triton\.compiler\.errors\.', re.M | re.I)


def _exception(text, pattern):
    return re.search(r'^(?:[\w.]+: )?' + pattern, text, re.M) is not None


def _frame(frames, path, functions=None):
    return any(filename.endswith(path) and (functions is None or function in functions)
               for filename, function in frames)


def _shape_failure(text, backend):
    from autofusex.execution.reporting.failure_origin import mlir_failed_line

    if classify_module(text, backend) or _CONFLICT.search(text) or mlir_failed_line(text):
        return None
    frames = _FRAME.findall(text)
    post_grad = '/torch/_inductor/fx_passes/post_grad.py'
    if (_exception(text, r"TypeError: int_noop\(\) got an unexpected keyword argument 'decimals'\s*$")
            and _frame(frames, post_grad, {'remove_noop_ops'})):
        return _INDUCTOR
    if (_exception(text, r"AttributeError: 'SymInt' object has no attribute 'size'\s*$")
            and _frame(frames, post_grad, {'same_meta'})):
        return _INDUCTOR
    if (_exception(text, r'AssertionError: aot_autograd expected to have an entirely functional graph\b')
            and _frame(frames, '/torch/_functorch/_aot_autograd/functional_utils.py',
                       {'assert_functional_graph'})):
        return _INDUCTOR
    if (_exception(text, r'(?:InductorError: )?UnsupportedIndexExpr: Unsupported index_expr\b')
            and _frame(frames, '/torch_npu/_inductor/ascendc/lowering/index_expr_lowering.py',
                       {'_validate_expression', '_reject_non_integer_atoms'})):
        return _INDUCTOR
    if _exception(text, r"(?:InductorError: )?LoweringException: RuntimeError: aten::pow\(\) "
                   r"Expected a value of type 'Tensor' for argument 'exponent' but instead found type 'SymInt'\."):
        return _INDUCTOR
    if (_exception(text, r'(?:InductorError: )?PermissionError: [^\n]+ '
                   r'must not be writable by group or others, but got 0o[0-7]+\s*$')
            and _frame(frames, '/torch_npu/_inductor/ascendc/common/utils.py', {'validate_lib'})):
        return _ENVIRONMENT
    if backend in _AF_BACKENDS:
        if _ATT_GE.search(text):
            return _AF_COMPILE
        if backend == 'af-mlir' and (_VECTOR_CORE.search(text) or _MLIR_KERNEL_BUILD.search(text)
                                      or _ASCEND_KERNEL_BUILD.search(text)):
            return _FUSED_KERNEL
        if (_exception(text, r'RuntimeError: NPU kernel [^\n]*autofused_[^\n]+ '
                       r'(?:init failed|execution failed\(-?\d+\))\s*$')
                and _frame(frames, '/torch_npu/_inductor/ascendc/compiler/_kernel.py',
                           {'load', '__call__'})):
            return _FUSED_KERNEL
        if (_exception(text, r'autofuse\.ascendc_compile\.CompileError: '
                       r'(?:Device|Host) compile failed\b')
                and _frame(frames, '/autofuse/ascendc_compile.py')):
            return _AF_COMPILE
    return None


def local_failure(detail, backend):
    """Require every failing shape to prove the same module; ignore shared logs."""
    from autofusex.server.failure_attribution import _object

    d = _object(detail)
    execution = _object(_object(d.get('contexts')).get('execution') or d.get('execution'))
    shapes = _object(_object(execution.get('diagnostics')).get('shapes'))
    summaries = {str(s.get('shape_index')): s.get('error_summary') or ''
                 for s in d.get('shape_results', []) if isinstance(s, dict)
                 and s.get('correctness') == 'error'}
    indices = set(summaries) | {str(k) for k, v in shapes.items() if _object(v).get('error')}
    result = None
    for index in indices:
        text = summaries.get(index, '') + '\n' + str(_object(shapes.get(index)).get('error') or '')
        current = _shape_failure(text, backend)
        if current is None or (result is not None and current != result):
            return None
        result = current
    return result
