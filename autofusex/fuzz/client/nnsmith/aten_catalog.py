"""Current E/F catalog: functional ATen elementwise, comparison, reduction and views.

Import-light, ordered specifications; never discover a different random catalog
from the local device or silently omit unavailable overloads. New rules require
another generation protocol. Matrix/convolution operators are deliberately absent.
"""
from dataclasses import dataclass


_DISPLAY_OVERRIDES = {
    'absolute': 'abs', 'negative': 'neg', 'multiply': 'mul',
    'true_divide': 'div', 'log_sigmoid': 'logsigmoid',
    'logical_and': 'logical_and', 'logical_or': 'logical_or',
    'logical_xor': 'logical_xor', 'bitwise_left_shift': 'bitwise_lshift',
    'bitwise_right_shift': 'bitwise_rshift', 'count_nonzero': 'count_nonzero',
    'linalg_vector_norm': 'vector_norm', 'log_softmax': 'logsoftmax',
    '_log_softmax': 'logsoftmax', '_softmax': 'softmax',
}


def _display_name(packet, family='', mode=''):
    """Return a short stable operator name independent of overload spelling."""
    if packet in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[packet]
    if family.startswith('reduce'):
        return {'sum': 'reducesum', 'mean': 'reducemean', 'amax': 'reducemax',
                'amin': 'reducemin', 'argmax': 'argmax', 'argmin': 'argmin'}.get(packet, packet)
    if mode in {'transpose', 'permute', 'movedim', 't'}:
        return 'transpose' if mode != 'permute' else 'permute'
    if mode == 'split':
        return 'split'
    if mode == 'flatten':
        return 'reshape'
    return packet


@dataclass(frozen=True)
class Rule:
    target: str
    family: str
    types: str = 'float'
    result: str = 'same'
    arity: int = 1
    outputs: int = 1
    mode: str = ''
    # Short, stable name used by reports and operator allow/deny lists.
    # ``target`` remains the executable ATen overload identity.
    display_name: str = ''


def _build():
    rows = []
    def add(names, family='unary', types='float', result='same', arity=1, outputs=1, mode='', display_name=None):
        for name in names.split():
            packet = name.split('.', 1)[0]
            short = (display_name(packet) if callable(display_name) else display_name) or _display_name(packet, family, mode)
            rows.append(Rule('aten.' + name, family, types, result, arity, outputs, mode, short))

    add('abs.default absolute.default', types='number', result='abs')
    add('neg.default negative.default square.default', types='number')
    add('exp.default exp2.default expm1.default log.default log2.default log10.default log1p.default '
        'sqrt.default rsqrt.default reciprocal.default sin.default cos.default tan.default '
        'asin.default acos.default atan.default sinh.default cosh.default tanh.default '
        'asinh.default acosh.default atanh.default sinc.default', types='inexact')
    add('ceil.default floor.default round.default trunc.default frac.default '
        'erf.default erfc.default erfinv.default lgamma.default digamma.default '
        'sigmoid.default relu.default relu6.default hardsigmoid.default hardswish.default '
        'silu.default mish.default log_sigmoid.default selu.default')
    add('sign.default', types='real')
    add('sgn.default conj.default resolve_conj.default resolve_neg.default', types='number')
    add('real.default imag.default', types='complex', result='real')
    add('angle.default', types='inexact', result='real')
    add('isfinite.default isinf.default isnan.default isreal.default logical_not.default', types='all', result='bool')
    add('isposinf.default isneginf.default signbit.default', types='real', result='bool')
    add('bitwise_not.default', types='integral_bool')
    add('nan_to_num.default')
    add('clamp.default clip.default clamp_min.default clamp_max.default '
        'hardtanh.default leaky_relu.default elu.default celu.default gelu.default '
        'hardshrink.default softshrink.default softplus.default threshold.default '
        'round.decimals logit.default', family='unary_param')
    add('add.Tensor sub.Tensor mul.Tensor multiply.Tensor pow.Tensor_Tensor', family='binary', types='number', arity=2)
    add('div.Tensor divide.Tensor true_divide.Tensor', family='binary', types='number', result='division', arity=2)
    add('maximum.default minimum.default fmax.default fmin.default '
        'remainder.Tensor fmod.Tensor floor_divide.default', family='binary', types='real', arity=2)
    add('atan2.default hypot.default copysign.Tensor nextafter.default '
        'logaddexp.default logaddexp2.default xlogy.Tensor', family='binary', arity=2)
    add('eq.Tensor ne.Tensor', family='binary', types='all', result='bool', arity=2)
    add('gt.Tensor ge.Tensor lt.Tensor le.Tensor', family='binary', types='real_bool', result='bool', arity=2)
    add('logical_and.default logical_or.default logical_xor.default', family='binary', types='all', result='bool', arity=2)
    add('bitwise_and.Tensor bitwise_or.Tensor bitwise_xor.Tensor', family='binary', types='integral_bool', arity=2)
    add('bitwise_left_shift.Tensor bitwise_right_shift.Tensor', family='binary', types='integral', arity=2)
    add('isclose.default', family='binary', result='bool', arity=2)
    add('add.Scalar sub.Scalar mul.Scalar multiply.Scalar pow.Tensor_Scalar '
        'remainder.Scalar fmod.Scalar floor_divide.Scalar rsub.Scalar', family='scalar', types='real')
    add('div.Scalar divide.Scalar true_divide.Scalar', family='scalar', types='real', result='division')
    add('eq.Scalar ne.Scalar gt.Scalar ge.Scalar lt.Scalar le.Scalar', family='scalar', types='real_bool', result='bool')
    add('bitwise_and.Scalar bitwise_or.Scalar bitwise_xor.Scalar '
        'bitwise_left_shift.Tensor_Scalar bitwise_right_shift.Tensor_Scalar', family='scalar', types='integral')
    add('lerp.Scalar', family='binary', arity=2)
    add('lerp.Tensor addcmul.default addcdiv.default', family='ternary', arity=3)
    add('where.self', family='where', types='all', arity=3)

    add('sum.default', family='reduce_all', types='real_bool', result='promote_int')
    add('nansum.default', family='reduce', types='real_bool', result='promote_int')
    add('sum.dim_IntList', family='reduce', types='real_bool', result='promote_int')
    add('mean.default', family='reduce_all', types='inexact')
    add('mean.dim', family='reduce', types='inexact')
    add('nanmean.default', family='reduce')
    add('prod.default', family='reduce_all', types='real_bool', result='promote_int')
    add('prod.dim_int', family='reduce_one', types='real_bool', result='promote_int')
    add('amax.default amin.default', family='reduce', types='real_bool')
    add('max.default min.default', family='reduce_all', types='real_bool')
    add('max.dim min.dim', family='reduce_one', types='real_bool', outputs=2, mode='indices')
    add('argmax.default argmin.default', family='reduce_optional', types='real', result='int64')
    add('all.default any.default', family='reduce_all', types='all', result='logical_reduce')
    add('all.dim any.dim', family='reduce_one', types='all', result='logical_reduce')
    add('all.dims any.dims', family='reduce', types='all', result='logical_reduce')
    add('count_nonzero.default', family='reduce_optional', types='all', result='int64')
    add('count_nonzero.dim_IntList', family='reduce', types='all', result='int64')
    add('logsumexp.default', family='reduce')
    add('var.correction std.correction', family='reduce')
    add('var_mean.correction std_mean.correction', family='reduce', outputs=2)
    add('aminmax.default', family='reduce_optional', types='real_bool', outputs=2)
    add('norm.ScalarOpt_dim linalg_vector_norm.default', family='reduce', types='inexact', result='real', mode='norm')
    add('cumsum.default cumprod.default', family='scan', types='real_bool', result='promote_int')
    add('cummax.default cummin.default', family='scan', types='real_bool', outputs=2, mode='indices')
    add('softmax.int log_softmax.int _softmax.default _log_softmax.default', family='scan')

    add('view.default reshape.default _unsafe_view.default view_copy.default '
        'flatten.using_ints', family='view', types='all', mode='flatten')
    add('unflatten.int', family='view', types='all', mode='unflatten')
    add('transpose.int transpose_copy.int', family='view', types='all', mode='transpose')
    add('permute.default permute_copy.default', family='view', types='all', mode='permute')
    add('t.default t_copy.default', family='view', types='all', mode='t')
    add('movedim.int movedim.intlist', family='view', types='all', mode='movedim')
    add('unsqueeze.default unsqueeze_copy.default', family='view', types='all', mode='unsqueeze')
    add('squeeze.default squeeze_copy.default', family='view', types='all', mode='squeeze_all')
    add('squeeze.dim squeeze_copy.dim squeeze.dims squeeze_copy.dims', family='view', types='all', mode='squeeze')
    add('expand.default expand_copy.default broadcast_to.default', family='view', types='all', mode='expand')
    add('slice.Tensor slice_copy.Tensor', family='view', types='all', mode='slice')
    add('select.int select_copy.int', family='view', types='all', mode='select')
    add('narrow.default', family='view', types='all', mode='narrow')
    add('clone.default detach.default detach_copy.default alias.default contiguous.default', family='view', types='all', mode='identity')
    add('view_as.default reshape_as.default', family='view_pair', types='all', arity=2, mode='reshape')
    add('expand_as.default', family='view_pair', types='all', arity=2, mode='expand')
    add('broadcast_tensors.default', family='view_pair', types='all', arity=2, outputs=2, mode='broadcast')
    add('chunk.default split.Tensor split_with_sizes.default', family='view', types='all', outputs=2, mode='split')
    add('unbind.int', family='view', types='all', outputs=2, mode='unbind')

    add('median.default nanmedian.default', family='reduce_all', types='real')
    add('median.dim nanmedian.dim mode.default kthvalue.default', family='reduce_one', types='real', outputs=2, mode='indices')
    add('topk.default', family='topk', types='real', outputs=2, mode='indices')
    add('var.default std.default', family='reduce_all')
    add('var.dim std.dim', family='reduce')
    add('var_mean.default std_mean.default', family='reduce_all', outputs=2)
    add('var_mean.dim std_mean.dim', family='reduce', outputs=2)
    add('logcumsumexp.default', family='scan')
    add('norm.Scalar', family='reduce_all', types='inexact', result='real', mode='norm')
    add('dist.default', family='dist', arity=2)
    add('ravel.default', family='view', types='all', mode='flatten')
    add('swapaxes.default swapdims.default', family='view', types='all', mode='transpose')
    add('tile.default repeat.default', family='view', types='all', mode='repeat')
    add('diagonal.default diagonal_copy.default', family='view', types='all', mode='diagonal')
    add('unfold.default unfold_copy.default', family='view', types='all', mode='unfold')
    add('view_as_real.default', family='view', types='complex', result='real', mode='complex_real')
    add('view_as_complex.default', family='view', types='float_complex_view', result='complex', mode='real_complex')
    # Additional pointwise tags, including functional backward formulas. These
    # are ordinary forward calls in the generated graph, not training mode.
    add('conj_physical.default positive.default', types='number')
    add('deg2rad.default rad2deg.default i0.default special_airy_ai.default '
        'special_bessel_j0.default special_bessel_j1.default special_bessel_y0.default '
        'special_bessel_y1.default special_entr.default special_erfcx.default '
        'special_i0e.default special_i1.default special_i1e.default '
        'special_log_ndtr.default special_modified_bessel_i0.default '
        'special_modified_bessel_i1.default special_modified_bessel_k0.default '
        'special_modified_bessel_k1.default special_ndtr.default special_ndtri.default '
        'special_scaled_modified_bessel_k0.default special_scaled_modified_bessel_k1.default '
        'special_spherical_bessel_j0.default')
    add('max.other min.other', family='binary', types='real', arity=2)
    add('clamp_min.Tensor clamp_max.Tensor', family='binary', types='real', arity=2)
    add('clamp.Tensor clip.Tensor', family='ternary', types='real', arity=3)
    add('gcd.default lcm.default __lshift__.Tensor __rshift__.Tensor __xor__.Tensor', family='binary', types='integral', arity=2)
    add('__lshift__.Scalar __rshift__.Scalar __xor__.Scalar', family='scalar', types='integral')
    add('bitwise_and.Scalar_Tensor bitwise_or.Scalar_Tensor bitwise_xor.Scalar_Tensor '
        'bitwise_left_shift.Scalar_Tensor bitwise_right_shift.Scalar_Tensor', family='scalar_left', types='integral')
    add('pow.Scalar', family='scalar_left', types='float')
    add('remainder.Scalar_Tensor', family='scalar_left', types='real')
    add('copysign.Scalar xlogy.Scalar_Other special_xlog1py.other_scalar special_zeta.other_scalar', family='scalar')
    add('xlogy.Scalar_Self special_xlog1py.self_scalar special_zeta.self_scalar', family='scalar_left')
    add('igamma.default igammac.default heaviside.default special_xlog1py.default '
        'special_zeta.default sigmoid_backward.default silu_backward.default '
        'tanh_backward.default gelu_backward.default logit_backward.default '
        'threshold_backward.default', family='binary', arity=2)
    add('masked_fill.Scalar native_dropout_backward.default', family='masked', arity=2)
    add('mvlgamma.default rrelu.default', family='unary_param')
    add('polygamma.default', family='scalar_left')
    add('float_power.Tensor_Tensor', family='binary', types='number', result='float_power', arity=2)
    add('float_power.Tensor_Scalar', family='scalar', types='number', result='float_power')
    add('float_power.Scalar', family='scalar_left', types='number', result='float_power')
    add('div.Tensor_mode', family='binary', types='real', arity=2)
    add('div.Scalar_mode', family='scalar', types='real')
    add('frexp.Tensor', outputs=2, mode='exponent')
    for polynomial in ('chebyshev_polynomial_t', 'chebyshev_polynomial_u',
                       'chebyshev_polynomial_v', 'chebyshev_polynomial_w',
                       'hermite_polynomial_h', 'hermite_polynomial_he',
                       'laguerre_polynomial_l', 'legendre_polynomial_p',
                       'shifted_chebyshev_polynomial_t', 'shifted_chebyshev_polynomial_u',
                       'shifted_chebyshev_polynomial_v', 'shifted_chebyshev_polynomial_w'):
        name = 'special_' + polynomial
        add(name+'.default', family='binary', arity=2)
        add(name+'.n_scalar', family='scalar')
        add(name+'.x_scalar', family='scalar_left')
    if len({r.target for r in rows}) != len(rows):
        raise ValueError('duplicate ATen rule')
    return tuple(sorted(rows, key=lambda r: r.target))


RULES = _build()
BY_TARGET = {r.target: r for r in RULES}


def inventory():
    """Audit named operator packets: modeled overloads versus explicit gaps."""
    import torch
    packets = {r.target.split('.')[1] for r in RULES}
    packets |= {'as_strided', 'as_strided_scatter', 'resize_', 'set_', 'equal',
                'quantile', 'nanquantile', 'unique_dim', '_unique2', 'sort', 'argsort',
                'squeeze_', 'unsqueeze_', 'transpose_', 'swapaxes_', 'swapdims_'}
    packets |= {name + '_' for name in tuple(packets)}
    result = []
    for schema in torch._C._jit_get_all_schemas():
        if not schema.name.startswith('aten::') or schema.name[6:] not in packets:
            continue
        target = 'aten.' + schema.name[6:] + '.' + (schema.overload_name or 'default')
        if target in BY_TARGET:
            reason = 'modeled'
        elif any(a.alias_info is not None and a.alias_info.is_write for a in schema.arguments):
            reason = 'mutation_or_out'
        elif schema.name[6:] == 'equal':
            reason = 'python_scalar_output'
        elif schema.name[6:] in {'as_strided', 'as_strided_scatter'}:
            reason = 'requires_storage_alias_model'
        else:
            reason = 'overload_not_modeled'
        result.append({'target': target, 'status': reason, 'schema': str(schema)})
    return sorted(result, key=lambda row: row['target'])
