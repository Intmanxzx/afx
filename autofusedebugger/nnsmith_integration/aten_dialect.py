"""NNSmith plugin for functional ATen operators; no installed files are patched.

Every class participates in the usual NNSmith symbolic solver and type transfer.
The registry is private; no installed NNSmith files or core registry are patched.
"""
import random
from functools import lru_cache

from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import (AbsOpBase, broadcast_shapes, broadcast_cons_binary,
                                 rank_all, rank_from, weighted_axis_choice,
                                 weighted_axis_sample)
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.abstract.arith import nnsmith_eq, nnsmith_ge, nnsmith_neq, nnsmith_div
from nnsmith.error import ConstraintCheck

from .aten_catalog import RULES

FLOAT = (DType.float16, DType.bfloat16, DType.float32, DType.float64)
INT = (DType.uint8, DType.int8, DType.int16, DType.int32, DType.int64)
COMPLEX = (DType.complex64, DType.complex128)
TYPE_SETS = {'float': FLOAT, 'integral': INT, 'integral_bool': INT + (DType.bool,),
             'complex': COMPLEX, 'float_complex_view': (DType.float32, DType.float64), 'inexact': FLOAT + COMPLEX,
             'real': FLOAT + INT, 'real_bool': FLOAT + INT + (DType.bool,),
             'number': FLOAT + INT + COMPLEX, 'all': FLOAT + INT + COMPLEX + (DType.bool,)}


def result_dtype(rule, dtype):
    if rule.result == 'complex': return DType.complex128 if dtype == DType.float64 else DType.complex64
    if rule.result == 'float_power':
        return DType.complex128 if dtype in COMPLEX else DType.float64
    if rule.result == 'bool': return DType.bool
    if rule.result == 'int64': return DType.int64
    if rule.result == 'logical_reduce': return DType.uint8 if dtype == DType.uint8 else DType.bool
    if rule.result == 'promote_int' and dtype in INT + (DType.bool,): return DType.int64
    if rule.result == 'division' and dtype in INT: return DType.float32
    if rule.result in {'real', 'abs'}:
        return {DType.complex64: DType.float32, DType.complex128: DType.float64}.get(dtype, dtype)
    return dtype


class AtenOp(AbsOpBase):
    dialect = 'aten_v1'
    num_var_param = None

    def __init__(self):
        super().__init__()
        rule = self.rule
        ranks = rank_all()
        if rule.family in {'reduce', 'reduce_one', 'scan', 'topk'}: ranks = rank_from(1)
        if rule.family == 'view':
            if rule.mode in {'unflatten', 'transpose', 'permute', 'movedim', 'squeeze', 'squeeze_all', 'slice', 'select', 'narrow', 'split', 'unbind', 'repeat', 'diagonal', 'unfold', 'real_complex'}:
                ranks = rank_from(1)
            if rule.mode in {'unflatten', 'unsqueeze', 'expand', 'complex_real', 'unfold'}: ranks = tuple(range(0 if rule.mode != 'unflatten' else 1, 5))
            if rule.mode == 't': ranks = (0, 1, 2)
            if rule.mode == 'diagonal': ranks = rank_from(2)
        self.inp_ranks = [ranks] * rule.arity
        outputs = rank_all()
        if rule.family in {'reduce_all', 'dist'}: outputs = (0,)
        if rule.family == 'view' and rule.mode == 'flatten': outputs = (1,)
        if rule.family == 'view' and rule.mode == 't': outputs = (0, 1, 2)
        self.out_ranks = [outputs] * rule.outputs

    def _configure(self, tensors):
        rule = self.rule
        attrs = self.extra_attrs
        rank = tensors[0].ndims
        def lazy(name, make):
            if name not in attrs: attrs[name] = make()
            return attrs[name]
        if rule.family.startswith('reduce'):
            lazy('keepdim', lambda: random.choice([False, True]) if rule.family != 'reduce_all' and rule.target != 'aten.count_nonzero.default' and rule.target != 'aten.count_nonzero.dim_IntList' else False)
            if 'dims' not in attrs:
                if rule.family == 'reduce_all': dims = list(range(rank))
                elif rule.family == 'reduce_optional' and random.choice([False, True]): dims = list(range(rank)); attrs['all'] = True
                elif rank == 0: dims = []
                else:
                    count = lazy('axis_count', lambda: random.randint(1, min(rank, 3)) if rule.family == 'reduce' else 1)
                    dims = weighted_axis_sample(rank, min(count, rank))
                attrs['dims'] = dims
            lazy('correction', lambda: random.choice([0, 1]))
            lazy('p', lambda: random.choice([1, 2, 3]))
        elif rule.family in {'scan', 'topk'}:
            lazy('dim', lambda: weighted_axis_choice(rank))
        elif rule.family == 'view':
            mode = rule.mode
            if mode == 'permute':
                if 'dims' not in attrs:
                    dims = list(range(rank)); random.shuffle(dims); attrs['dims'] = dims
            elif mode in {'transpose', 'movedim'}:
                lazy('dim', lambda: weighted_axis_choice(rank))
                attrs.setdefault('other', (attrs['dim'] + 1) % max(rank, 1))
            elif mode == 'unsqueeze': lazy('dim', lambda: random.randrange(rank + 1))
            elif mode in {'unflatten', 'squeeze', 'squeeze_all', 'slice', 'select', 'narrow', 'split', 'unbind', 'repeat', 'diagonal', 'unfold', 'real_complex'}:
                lazy('dim', lambda: weighted_axis_choice(rank))
            if mode == 'slice': lazy('step', lambda: random.choice([1, 2, 3]))
        if rule.family in {'scalar', 'scalar_left', 'unary_param', 'binary', 'ternary', 'masked'}:
            # Values are data in the saved graph, not re-sampled by materializers.
            lazy('scalar', lambda: random.choice([1, 2, 3]))
            lazy('alpha', lambda: random.choice([1, 2]))
            lazy('negative_slope', lambda: random.choice([0.01, 0.1, 0.2]))
            lazy('approximate', lambda: random.choice(['none', 'tanh']))
            lazy('decimals', lambda: random.choice([-1, 0, 1, 2]))

    def requires(self, tensors):
        self._configure(tensors)
        rule, attrs = self.rule, self.extra_attrs
        cons = []
        if rule.family in {'binary', 'ternary', 'where', 'masked', 'dist'}:
            if len(tensors) == 2:
                cons += broadcast_cons_binary(tensors[0].shape, tensors[1].shape)
            else:
                cons += broadcast_cons_binary(tensors[0].shape, tensors[1].shape)
                cons += broadcast_cons_binary(broadcast_shapes(tensors[0].shape, tensors[1].shape), tensors[2].shape)
        if rule.family == 'view_pair':
            if rule.mode == 'reshape': cons += [nnsmith_eq(tensors[0].nelement(), tensors[1].nelement())]
            elif rule.mode == 'expand':
                ConstraintCheck.true(tensors[0].ndims <= tensors[1].ndims)
                from nnsmith.abstract.arith import nnsmith_or
                a = [1] * (tensors[1].ndims - tensors[0].ndims) + list(tensors[0].shape)
                cons += [nnsmith_or(nnsmith_eq(x, 1), nnsmith_eq(x, y)) for x, y in zip(a, tensors[1].shape)]
            else: cons += broadcast_cons_binary(tensors[0].shape, tensors[1].shape)
        if rule.family == 'view':
            mode = rule.mode
            if mode in {'squeeze', 'squeeze_all'}:
                cons += [nnsmith_eq(tensors[0].shape[attrs['dim']], 1)]
                if mode == 'squeeze_all':
                    cons += [nnsmith_neq(size, 1) for i, size in enumerate(tensors[0].shape) if i != attrs['dim']]
            elif mode in {'select', 'slice', 'narrow'}: cons += [nnsmith_ge(tensors[0].shape[attrs['dim']], 1)]
            elif mode == 'unbind': cons += [nnsmith_eq(tensors[0].shape[attrs['dim']], 2)]
            elif mode == 'real_complex': cons += [nnsmith_eq(tensors[0].shape[-1], 2)]
            elif mode == 'split':
                size = tensors[0].shape[attrs['dim']]
                cons += [nnsmith_ge(size, 2), nnsmith_eq(size % 2, 0)]
        if rule.family.startswith('reduce'):
            cons += [nnsmith_ge(tensors[0].shape[d], 1) for d in attrs['dims']]
        return cons

    def type_transfer(self, tensors):
        self._configure(tensors)
        rule, attrs = self.rule, self.extra_attrs
        shape = list(tensors[0].shape)
        dtype = tensors[1].dtype if rule.family == 'where' else tensors[0].dtype
        if rule.family in {'binary', 'ternary', 'where', 'masked', 'dist'}:
            shape = broadcast_shapes(*(t.shape for t in tensors))
        elif rule.family.startswith('reduce'):
            dims = set(attrs['dims'])
            shape = [1 if i in dims else v for i, v in enumerate(shape)] if attrs['keepdim'] else [v for i, v in enumerate(shape) if i not in dims]
        elif rule.family == 'view_pair':
            shape = broadcast_shapes(*(t.shape for t in tensors)) if rule.mode == 'broadcast' else list(tensors[1].shape)
        elif rule.family == 'view':
            mode = rule.mode
            if mode == 'flatten': shape = [tensors[0].nelement()]
            elif mode == 'unflatten':
                d = attrs['dim']; shape = shape[:d] + [1, shape[d]] + shape[d+1:]
            elif mode in {'transpose', 'movedim'}:
                a, b = attrs['dim'], attrs['other']
                if mode == 'transpose': shape[a], shape[b] = shape[b], shape[a]
                else: shape.insert(b, shape.pop(a))
            elif mode == 'permute': shape = [shape[i] for i in attrs['dims']]
            elif mode == 't': shape = shape[::-1] if len(shape) == 2 else shape
            elif mode == 'unsqueeze': shape.insert(attrs['dim'], 1)
            elif mode in {'squeeze', 'squeeze_all', 'select', 'unbind'}: shape.pop(attrs['dim'])
            elif mode == 'expand': shape = [2] + shape
            elif mode == 'repeat': shape = [v * 2 if i == attrs['dim'] else v for i, v in enumerate(shape)]
            elif mode == 'diagonal':
                from nnsmith.abstract.arith import nnsmith_min
                shape = shape[:-2] + [nnsmith_min(shape[-2], shape[-1])]
            elif mode == 'unfold': shape = shape + [1]
            elif mode == 'complex_real': shape = shape + [2]
            elif mode == 'real_complex': shape = shape[:-1]
            elif mode == 'slice':
                d = attrs['dim']; shape[d] = nnsmith_div(shape[d] + attrs['step'] - 1, attrs['step'])
            elif mode == 'narrow': shape[attrs['dim']] = 1
            elif mode == 'split': shape[attrs['dim']] = nnsmith_div(shape[attrs['dim']], 2)
        if rule.family == 'dist': shape = []
        if rule.family == 'topk': shape[attrs['dim']] = 1
        dtype = result_dtype(rule, dtype)
        result = [AbsTensor(shape, dtype)]
        if rule.outputs == 2:
            result.append(AbsTensor(list(shape), DType.int64 if rule.mode == 'indices' else DType.int32 if rule.mode == 'exponent' else dtype))
        return result

    def deduct_inp_ranks_and_dtype(self, outputs):
        rule, attrs = self.rule, self.extra_attrs
        signatures = [(ins, outs) for ins, outs in zip(self.in_dtypes, self.out_dtypes)
                      if tuple(t.dtype for t in outputs) == outs]
        ConstraintCheck.true(bool(signatures))
        ins, _ = random.choice(signatures)
        rank = outputs[0].ndims
        ranks = [rank] * rule.arity
        if rule.family.startswith('reduce'):
            keep = attrs.setdefault('keepdim', random.choice([False, True]) if rule.family != 'reduce_all' and not rule.target.startswith('aten.count_nonzero.') else False)
            if rule.family == 'reduce_all' or rule.family == 'reduce_optional' and rank == 0:
                ranks = [random.randrange(1, 6)]; attrs['all'] = True
                attrs['dims'] = list(range(ranks[0]))
            else:
                count = attrs.setdefault('axis_count', random.randint(1, min(3, max(1, 5-rank))) if rule.family == 'reduce' and not keep else 1)
                ranks = [rank if keep else rank + count]
        elif rule.family == 'view':
            mode = rule.mode
            if mode == 'flatten': ranks = [random.randrange(0, 6)]
            elif mode in {'unsqueeze', 'expand', 'unflatten', 'complex_real', 'unfold'}: ranks = [rank - 1]
            elif mode in {'squeeze', 'squeeze_all', 'select', 'unbind', 'diagonal', 'real_complex'}: ranks = [rank + 1]
        elif rule.family == 'dist': ranks = [random.randrange(0, 6)] * 2
        elif rule.family == 'view_pair' and rule.mode == 'reshape': ranks[0] = random.randrange(0, 6)
        ConstraintCheck.true(all(r in allowed for r, allowed in zip(ranks, self.inp_ranks)))
        return list(zip(ranks, ins))


def _class_name(rule):
    return 'Aten_' + rule.target.removeprefix('aten.').replace('.', '_')


@lru_cache(maxsize=1)
def classes():
    result = []
    for rule in RULES:
        signatures = []
        outputs = []
        for dtype in TYPE_SETS[rule.types]:
            ins = (DType.bool, dtype, dtype) if rule.family == 'where' else (dtype, DType.bool) if rule.family == 'masked' else (dtype,) * rule.arity
            out = (result_dtype(rule, dtype),)
            if rule.outputs == 2: out += (DType.int64 if rule.mode == 'indices' else DType.int32 if rule.mode == 'exponent' else out[0],)
            signatures.append(ins); outputs.append(out)
        cls = type(_class_name(rule), (AtenOp,), dict(rule=rule, in_dtypes=signatures, out_dtypes=outputs,
                    __module__=__name__))
        globals()[cls.__name__] = cls
        result.append(cls)
    return tuple(result)


def resolve_target(target):
    from .aten_catalog import BY_TARGET
    if target not in BY_TARGET: raise ValueError('unregistered ATen target')
    import torch
    _, name, overload = target.split('.')
    return getattr(getattr(torch.ops.aten, name), overload)


def validate_runtime():
    import torch
    from ..fuzz.generation.runtime import source_evidence
    evidence = source_evidence()
    if str(torch.__version__) != '2.10.0+cpu':
        raise RuntimeError('C2/D2 requires pinned torch 2.10.0+cpu; no catalog fallback')
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError('C2/D2 requires default float32 dtype')
    for rule in RULES: resolve_target(rule.target)
    return evidence


def materialize(op):
    rule, attrs = op.rule, op.extra_attrs
    fn = resolve_target(rule.target)
    target = rule.target
    if rule.family in {'unary', 'dist'}: return fn
    if rule.family == 'topk': return lambda x: fn(x, 1, attrs['dim'])
    if rule.family == 'scalar':
        if target == 'aten.div.Scalar_mode': return lambda x: fn(x, attrs['scalar'], rounding_mode='floor')
        return lambda x: fn(x, attrs['scalar'])
    if rule.family == 'scalar_left': return lambda x: fn(attrs['scalar'], x)
    if rule.family == 'masked': return lambda x, mask: fn(x, mask, 0 if target == 'aten.masked_fill.Scalar' else 1.0)
    if rule.family == 'unary_param':
        kwargs = {}
        if target in {'aten.clamp.default', 'aten.clip.default', 'aten.hardtanh.default'}:
            kwargs = {'min': -2, 'max': 2} if target != 'aten.hardtanh.default' else {'min_val': -2, 'max_val': 2}
        elif 'clamp_min' in target: kwargs = {'min': -2}
        elif 'clamp_max' in target: kwargs = {'max': 2}
        elif 'leaky_relu' in target: kwargs = {'negative_slope': attrs['negative_slope']}
        elif 'gelu' in target: kwargs = {'approximate': attrs['approximate']}
        elif 'round.decimals' in target: kwargs = {'decimals': attrs['decimals']}
        elif 'logit' in target: kwargs = {'eps': 0.01}
        elif 'mvlgamma' in target: kwargs = {'p':1}
        elif 'rrelu' in target: kwargs = {'training':False}
        elif 'threshold' in target: kwargs = {'threshold': 0.5, 'value': 0}
        return lambda x: fn(x, **kwargs)
    if rule.family == 'binary':
        if target in {'aten.add.Tensor', 'aten.sub.Tensor'}: return lambda x, y: fn(x, y, alpha=attrs['alpha'])
        if target == 'aten.div.Tensor_mode': return lambda x, y: fn(x, y, rounding_mode='floor')
        if target == 'aten.threshold_backward.default': return lambda x, y: fn(x, y, 0.5)
        if target == 'aten.lerp.Scalar': return lambda x, y: fn(x, y, 0.5)
        return fn
    if rule.family in {'ternary', 'where'}: return fn
    if rule.family.startswith('reduce'):
        def reduce(x):
            kwargs = {}
            if rule.family != 'reduce_all':
                dims = attrs['dims']
                dim = None if attrs.get('all') else (dims[0] if rule.family in {'reduce_one', 'reduce_optional'} and dims else None)
                kwargs['dim'] = dims if rule.family == 'reduce' else dim
                if not target.startswith('aten.count_nonzero.'): kwargs['keepdim'] = attrs['keepdim']
            if rule.mode == 'norm':
                kwargs.update({'p':attrs['p']} if target.startswith('aten.norm.') else {'ord':attrs['p']})
            if '.correction' in target: kwargs['correction'] = attrs['correction']
            if target == 'aten.kthvalue.default': kwargs['k'] = 1
            return fn(x, **kwargs)
        return reduce
    if rule.family == 'scan':
        if target in {'aten._softmax.default', 'aten._log_softmax.default'}:
            return lambda x: fn(x, attrs['dim'], False)
        return lambda x: fn(x, attrs['dim'])
    if rule.family == 'view_pair':
        if rule.mode == 'broadcast': return lambda x, y: fn([x, y])
        return fn
    mode = rule.mode
    if mode == 'complex_real' or mode == 'real_complex': return fn
    if mode == 'repeat':
        return lambda x: fn(x, [2 if i == attrs['dim'] else 1 for i in range(x.ndim)])
    if mode == 'diagonal': return lambda x: fn(x, 0, -2, -1)
    if mode == 'unfold': return lambda x: fn(x, attrs['dim'], 1, 1)
    if mode == 'flatten':
        if target == 'aten.ravel.default': return fn
        if target == 'aten.flatten.using_ints': return lambda x: fn(x, 0, -1)
        return lambda x: fn(x, [-1])
    if mode == 'unflatten': return lambda x: fn(x, attrs['dim'], [1, x.shape[attrs['dim']]])
    if mode == 'transpose': return lambda x: fn(x, attrs['dim'], attrs['other'])
    if mode == 'permute': return lambda x: fn(x, attrs['dims'])
    if mode == 'movedim':
        if target.endswith('.intlist'): return lambda x: fn(x, [attrs['dim']], [attrs['other']])
        return lambda x: fn(x, attrs['dim'], attrs['other'])
    if mode == 'unsqueeze': return lambda x: fn(x, attrs['dim'])
    if mode == 'squeeze':
        return lambda x: fn(x, [attrs['dim']] if target.endswith('.dims') else attrs['dim'])
    if mode == 'squeeze_all': return fn
    if mode == 'expand': return lambda x: fn(x, [2, *x.shape])
    if mode == 'slice': return lambda x: fn(x, attrs['dim'], 0, 9223372036854775807, attrs['step'])
    if mode == 'select': return lambda x: fn(x, attrs['dim'], 0)
    if mode == 'narrow': return lambda x: fn(x, attrs['dim'], 0, 1)
    if mode == 'split':
        if target == 'aten.chunk.default': return lambda x: fn(x, 2, attrs['dim'])
        if target == 'aten.split_with_sizes.default': return lambda x: fn(x, [x.shape[attrs['dim']] // 2] * 2, attrs['dim'])
        return lambda x: fn(x, x.shape[attrs['dim']] // 2, attrs['dim'])
    if mode == 'unbind': return lambda x: fn(x, attrs['dim'])
    return fn
