"""One sample loop for static/dynamic correctness and optional measurement.

Backend adapters own only capability checks, preparation, invocation and sync.
No artifact pickling or claim that a lazy callable is already compiled.
"""
import time
from dataclasses import asdict

from .generation import reference, workload_id, oracle_id
from .backends.base import UnsupportedBackend


def execute(bundle, adapter, evaluation):
    started=time.perf_counter_ns()
    try:
        result = _execute(bundle, adapter, evaluation)
        import json
        stages = result.pop('_timings', {})
        stages['execution_total'] = time.perf_counter_ns()-started
        shapes = {str(row['shape_index']): row.pop('_timings', {}) for row in result['shape_results']}
        result.setdefault('execution', {})['telemetry'] = json.dumps({'version':1,'stages':stages,'shapes':shapes})
        return result
    finally:
        if hasattr(adapter, 'close'):
            adapter.close()


def _execute(bundle, adapter, evaluation):
    result = {"seed": str(bundle.spec.seed), "workload_id": workload_id(bundle),
              "oracle_id": oracle_id(bundle), "backend": adapter.name, "shape_results": [],
              "_timings": {}, "compile_mode": "combined", "preview": True, "evaluation": asdict(evaluation)}
    if hasattr(adapter, "config"):
        result["execution"] = asdict(adapter.config)
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
    except Exception as exc:
        result['_timings'][stage] = time.perf_counter_ns()-started
        result.update(status="unsupported" if isinstance(exc, UnsupportedBackend) else "error",
                      error_stage=stage, error_summary=f"{type(exc).__name__}: {exc}"[:500])
        return result
    static_only = getattr(getattr(adapter, "config", None), "static_only", False)
    indices = range(1 if static_only else len(bundle.spec.shape_samples))
    result["sample_selection"] = "base_only" if static_only else "all"
    for index in indices:
        row = {"shape_index": index, "correctness": "not_checked", "performance": "not_requested", "_timings": {}}
        result["shape_results"].append(row)
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
            if evaluation.correctness:
                stage = "correctness"
                started=time.perf_counter_ns()
                row["correctness"] = "pass" if adapter.compare(output, reference(bundle, index)) else "fail"
                row['_timings']['correctness']=time.perf_counter_ns()-started
            if evaluation.performance:
                stage = "warmup"
                # Reset inputs outside timing even though the frozen minimal profile is functional.
                started=time.perf_counter_ns()
                for _ in range(evaluation.warmup_steps):
                    adapter.invoke(prepared, adapter.inputs(bundle, index))
                    adapter.synchronize()
                row['_timings']['warmup']=time.perf_counter_ns()-started
                stage = "measurement"
                measurement_start=time.perf_counter_ns()
                durations = []
                for _ in range(evaluation.measured_steps):
                    inputs = adapter.inputs(bundle, index)
                    adapter.synchronize()
                    started = time.perf_counter_ns()
                    adapter.invoke(prepared, inputs)
                    adapter.synchronize()
                    durations.append(time.perf_counter_ns() - started)
                row['_timings']['measurement']=time.perf_counter_ns()-measurement_start
                row.update(performance="measured", mean_wall_ns=sum(durations) // len(durations),
                           measured_steps=len(durations), metric="synchronized_wall_time_not_kernel_profiler")
        except Exception as exc:
            if stage not in row['_timings']:
                row['_timings'][stage]=time.perf_counter_ns()-started
            row.update(error_stage=stage, error_summary=f"{type(exc).__name__}: {exc}"[:500])
            if stage in {"measurement", "warmup"}:
                row["performance"] = "error"  # Do not erase correctness.
            else:
                row["correctness"] = "error"
                if evaluation.performance:
                    row["performance"] = "skipped"
    result["status"] = "error" if any("error_stage" in r for r in result["shape_results"]) else (
        "fail" if any(r["correctness"] == "fail" for r in result["shape_results"]) else "ok")
    return result
