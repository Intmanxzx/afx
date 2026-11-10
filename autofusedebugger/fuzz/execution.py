"""One sample loop for static/dynamic correctness and optional measurement.

Backend adapters own only capability checks, preparation, invocation and sync.
No artifact pickling or claim that a lazy callable is already compiled.
"""
import time
from dataclasses import asdict

from .generation import reference, workload_id, oracle_id
from .backends.base import UnsupportedBackend


def execute(bundle, adapter, evaluation, *, deferred=False, outputs=None, shape_starts=None):
    started=time.perf_counter_ns()
    try:
        result = _execute(bundle, adapter, evaluation, deferred=deferred, outputs=outputs, shape_starts=shape_starts)
        import json
        stages = result.pop('_timings', {})
        stages['execution_total'] = time.perf_counter_ns()-started
        diagnostics = result.pop('_diagnostics', {'version':1,'shapes':{}})
        result.setdefault('execution', {})['diagnostics'] = json.dumps(diagnostics)
        shapes = {str(row['shape_index']): row.pop('_timings', {}) for row in result['shape_results']}
        result.setdefault('execution', {})['telemetry'] = json.dumps({'version':2,'stages':stages,'shapes':shapes})
        result.pop('_secondary', None)
        return result
    finally:
        if hasattr(adapter, 'close'):
            adapter.close()


def _execute(bundle, adapter, evaluation, *, deferred=False, outputs=None, shape_starts=None):
    result = {"seed": str(bundle.spec.seed), "workload_id": workload_id(bundle),
              "oracle_id": oracle_id(bundle), "backend": adapter.name, "shape_results": [],
              "_timings": {}, "_diagnostics": {"version":1,"shapes":{}}, "compile_mode": "combined", "preview": True, "evaluation": asdict(evaluation)}
    if hasattr(adapter, "config"):
        result["execution"] = asdict(adapter.config)
    if bundle.spec.seed.version in {3,4}:
        from .case_spec import encode
        result.setdefault('execution', {})['case_spec'] = encode(bundle.spec)
    if hasattr(adapter, 'cpu_reference'):
        result['oracle_id'] = 'torch-eager-cpu/1'
    stage = "capability"
    try:
        started = time.perf_counter_ns()
        adapter.check(bundle, evaluation)
        result['_timings']['capability'] = time.perf_counter_ns()-started
        started = time.perf_counter_ns()
        stage = "prepare"
        prepared = adapter.prepare(bundle)
        result["prepare_ns"] = time.perf_counter_ns() - started
        result['_timings']['prepare'] = result['prepare_ns']
        result["prepare_is_compile_complete"] = False
        compile_duration = getattr(adapter, 'compile_duration_ns', None)
        if callable(compile_duration):
            value = compile_duration()
            if value is not None:
                if type(value) is not int or not 0 <= value < 2**63:
                    raise ValueError('invalid isolated compile duration')
                result['_timings']['compile'] = value
    except Exception as exc:
        result['_timings'][stage] = time.perf_counter_ns()-started
        from .failure_details import excerpt
        result['_diagnostics']['error'] = excerpt(exc)
        result.update(status="unsupported" if isinstance(exc, UnsupportedBackend) else "error",
                      error_stage=stage, error_summary=f"{type(exc).__name__}: {exc}"[:500])
        return result
    static_only = getattr(getattr(adapter, "config", None), "static_only", False)
    indices = range(1 if static_only else len(bundle.spec.shape_samples))
    result["sample_selection"] = "base_only" if static_only else "all"
    for index in indices:
        row = {"shape_index": index, "correctness": "not_checked", "performance": "not_requested", "_timings": {}}
        result["shape_results"].append(row)
        processing_started = time.perf_counter_ns()
        if shape_starts is not None:
            shape_starts[str(index)] = time.monotonic_ns()
        if outputs is not None:
            outputs.append(())
        stage = "input_binding"
        try:
            started=time.perf_counter_ns()
            inputs = adapter.inputs(bundle, index)
            row['_timings']['input_binding']=time.perf_counter_ns()-started
            stage = "first_call_combined"
            started = time.perf_counter_ns()
            output = adapter.invoke(prepared, inputs)
            adapter.synchronize()
            row["first_call_combined_ns"] = time.perf_counter_ns() - started
            row['_timings']['first_call_combined']=row['first_call_combined_ns']
            if outputs is not None:
                transfer_started = time.perf_counter_ns()
                outputs[index] = tuple(t.detach().cpu().clone() for t in output)
                adapter.synchronize()
                row['_timings']['output_transfer'] = time.perf_counter_ns() - transfer_started
            if evaluation.correctness and not deferred:
                stage = "correctness"
                started=time.perf_counter_ns()
                expected=(adapter.cpu_reference(bundle,index) if hasattr(adapter,'cpu_reference')
                          else reference(bundle,index))
                cpu_pass=adapter.compare(output,expected)
                row['correctness']='pass' if cpu_pass else 'fail'
                if not cpu_pass and hasattr(adapter,'diagnose_accuracy'):
                    diagnostic=adapter.diagnose_accuracy(output,expected)
                    diagnostic.update(cpu_pass=False,secondary_used=False,basis='cpu')
                    result['_diagnostics']['shapes'][str(index)]=diagnostic
                row['_timings']['correctness']=time.perf_counter_ns()-started
            if evaluation.performance:
                # Only explicit device-profiler measurements qualify. Wall time
                # and first invocation cannot stand in for kernel or compile time.
                measure = getattr(adapter, 'measure_kernel', None)
                if callable(measure) and adapter.name not in {'cpu', 'cpu-eager'}:
                    stage = 'measurement'
                    started = time.perf_counter_ns()
                    value = measure(prepared, inputs, evaluation)
                    if type(value) is not int or not 0 <= value < 2**63:
                        raise ValueError('invalid kernel profiler duration')
                    row['_timings']['kernel_mean'] = value
                    row['performance'] = 'measured'
                else:
                    row['performance'] = 'skipped'
        except Exception as exc:
            from .failure_details import excerpt
            if index<3:result['_diagnostics']['shapes'].setdefault(str(index),{})['error']=excerpt(exc)
            if stage not in row['_timings']:
                row['_timings'][stage]=time.perf_counter_ns()-started
            row.update(error_stage=stage, error_summary=f"{type(exc).__name__}: {exc}"[:500])
            if stage in {"measurement", "warmup"}:
                row["performance"] = "error"  # Do not erase correctness.
            else:
                row["correctness"] = "error"
                if evaluation.performance:
                    row["performance"] = "skipped"
        finally:
            row["_timings"]["shape_processing"] = time.perf_counter_ns() - processing_started
    result["status"] = "error" if any("error_stage" in r for r in result["shape_results"]) else (
        "fail" if any(r["correctness"] == "fail" for r in result["shape_results"]) else "ok")
    return result
