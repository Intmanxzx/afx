"""Bounded strict-comparison evidence; masks use the same predicate as pass/fail."""
import math


def nonfinite_evidence(actual, reference):
    """Count elements; complex alignment checks each component and Inf sign."""
    import torch
    a = torch.view_as_real(actual) if actual.is_complex() else actual.unsqueeze(-1)
    r = torch.view_as_real(reference) if reference.is_complex() else reference.unsqueeze(-1)
    def counts(x):
        return {name: int(predicate(x).any(dim=-1).sum()) for name, predicate in (
            ('nan', torch.isnan), ('pos_inf', torch.isposinf), ('neg_inf', torch.isneginf))}
    af, rf = torch.isfinite(a), torch.isfinite(r)
    ap, rp = (~af).any(dim=-1), (~rf).any(dim=-1)
    present = ap | rp
    aligned = ((af & rf) | (torch.isnan(a) & torch.isnan(r)) |
               (torch.isposinf(a) & torch.isposinf(r)) |
               (torch.isneginf(a) & torch.isneginf(r))).all(dim=-1)
    return dict(actual=counts(a), reference=counts(r),
                actual_only=int((ap & ~rp).sum()), reference_only=int((rp & ~ap).sum()),
                aligned_nan=int((aligned & torch.isnan(a).any(dim=-1)).sum()), aligned=int((present & aligned).sum()), different=int((present & ~aligned).sum()))


def summarize(actuals, references, close):
    import torch
    from autofusex.execution.compare import resolve_tolerances
    if len(actuals) != len(references):
        return {'error': 'output count mismatch'}
    outputs = []
    for index, (actual, reference) in enumerate(zip(actuals, references)):
        if actual.shape != reference.shape or actual.dtype != reference.dtype:
            return {'error': 'output shape/dtype mismatch'}
        actual, reference = actual.detach().cpu(), reference.detach().cpu()
        mask = ~close(actual, reference).reshape(-1)
        count, total = int(mask.sum()), mask.numel()
        finite = torch.isfinite(actual) & torch.isfinite(reference)
        # Complex magnitudes require preserving both components.
        dtype = torch.complex128 if actual.is_complex() else torch.float64
        a, r = actual.to(dtype), reference.to(dtype)
        diff = (a-r).abs()[finite]
        denom = r.abs()[finite]
        rtol, atol, _ = resolve_tolerances(reference.dtype, actual.dtype)
        if not actual.is_floating_point() and not actual.is_complex():
            rtol = atol = 0.0
        tolerance = atol + rtol * denom
        ratio = diff / tolerance.clamp_min(1e-300)
        unbounded = bool(((tolerance == 0) & mask.reshape(actual.shape)[finite]).any())
        maximum = float(ratio.max()) if ratio.numel() and not unbounded else None
        if maximum is not None and not math.isfinite(maximum): maximum = None
        relative = diff[denom > 0] / denom[denom > 0]
        # Run statistics are exact without persisting every mismatch coordinate.
        starts = mask & ~torch.cat((torch.zeros(1, dtype=torch.bool), mask[:-1])) if total else mask
        runs = int(starts.sum())
        # Bound coordinate memory for large failures; never sample pass/fail.
        positions = torch.nonzero(mask).flatten() if 3 <= count <= 65536 and runs > 1 else None
        periodic = positions is not None and bool(((positions[1:] - positions[:-1]) == (positions[1] - positions[0])).all())
        pattern = 'none' if not count else 'all' if count == total else 'contiguous' if runs == 1 else 'periodic' if periodic else 'scattered'
        outputs.append(dict(output_index=index, shape=list(actual.shape), dtype=str(actual.dtype),
            rtol=rtol, atol=atol, elements=total, mismatches=count,
            finite_pairs=int(finite.sum()), finite_mismatches=int((mask & finite.reshape(-1)).sum()),
            nonfinite_evidence=nonfinite_evidence(actual, reference),
            mismatch_ratio=count/max(1,total), nonfinite=int((~finite).sum()),
            nan_mismatches=int(((torch.isnan(actual)|torch.isnan(reference)).reshape(-1)&mask).sum()),
            inf_mismatches=int(((torch.isinf(actual)|torch.isinf(reference)).reshape(-1)&mask).sum()),
            max_abs=float(diff.max()) if diff.numel() else None,
            mean_abs=float(diff.mean()) if diff.numel() else None,
            rmse=float(torch.linalg.vector_norm(diff)/math.sqrt(diff.numel())) if diff.numel() else None,
            max_rel=float(relative.max()) if relative.numel() else None,
            tolerance_ratio_max=maximum, tolerance_unbounded=unbounded,
            error_runs=runs, spatial_pattern=pattern))
    # Finite inputs can overflow diagnostic arithmetic; keep JSON serializable.
    for output in outputs:
        for key, value in output.items():
            if isinstance(value, float) and not math.isfinite(value):
                output[key] = None
    def maximum(key):
        values=[o[key] for o in outputs if o[key] is not None and math.isfinite(o[key])]
        return max(values) if values else None
    total=sum(o['elements'] for o in outputs); count=sum(o['mismatches'] for o in outputs)
    finite=sum(o['elements']-o['nonfinite'] for o in outputs)
    return dict(outputs=outputs, elements=total, mismatches=count, mismatch_ratio=count/max(1,total),
        nonfinite=sum(o['nonfinite'] for o in outputs), max_abs=maximum('max_abs'),
        max_rel=maximum('max_rel'), mean_abs=sum((o['mean_abs'] or 0)*(o['elements']-o['nonfinite']) for o in outputs)/max(1,finite))


def classify(evidence, features=()):
    comparisons=evidence.get('comparisons', [])
    if not comparisons: return None
    outputs=[o for s in comparisons[-1].get('shapes', []) for o in s.get('metrics', {}).get('outputs', []) if o.get('mismatches')]
    if not outputs: return None
    o=max(outputs,key=lambda o:(bool(o.get('nan_mismatches') or o.get('inf_mismatches')),o.get('tolerance_unbounded',False),o.get('tolerance_ratio_max') or 0,o['mismatch_ratio']))
    strength=o.get('tolerance_ratio_max')
    severity='nonfinite' if o.get('nan_mismatches') or o.get('inf_mismatches') else '>100x' if o.get('tolerance_unbounded') else 'unknown' if strength is None else '1-2x' if strength<=2 else '2-10x' if strength<=10 else '10-100x' if strength<=100 else '>100x'
    ratio=o['mismatch_ratio']
    bucket='100%' if ratio==1 else '<=0.1%' if ratio<=.001 else '<=1%' if ratio<=.01 else '<=10%' if ratio<=.1 else '<100%'
    anomaly = 'nan+inf' if o.get('nan_mismatches') and o.get('inf_mismatches') else 'nan' if o.get('nan_mismatches') else 'inf' if o.get('inf_mismatches') else 'finite'
    return dict(version=1,anomaly=anomaly,severity=severity,ratio=bucket,pattern=o['spatial_pattern'],
        features=sorted(set(features)),output_index=o['output_index'])
