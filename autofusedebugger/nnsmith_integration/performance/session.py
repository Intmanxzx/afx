"""Serial paired measurements, with continuous operation and resumable checkpoints."""
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback

from .artifacts import ArtifactCleanupError, compact_pending_case
from .backends import BACKENDS
from .cases import case_fingerprint, describe_case, snapshot_opset, restore_opset
from .checkpoint import load_checkpoint, software_signature, tuple_state
from .lifecycle import SessionStopped, stop_signals, defer_stop_signals
from .metrics import comparable_speedup, performance_status, correctness_status
from .report import write_reports
from .runner import run_backend
from .schema import ProfileConfig
from .storage import atomic_write_text, session_lock


def _save_checkpoint(summary, out):
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False))


def _measure_case(*, index, case_seed, out, generation, profile, opset, model_cls, target, devices):
    from ..generation import generate_case
    from .shape_modes import resolve_generation
    generation = resolve_generation(generation, case_seed)
    case = {"seed": case_seed, "index": index, "case_id": f"{index:05d}-{case_seed}", "backends": {},
            "shape_mode": "dynamic" if generation.dynamic_shapes else "static", "target": target}
    started, stage = time.monotonic(), "generation"
    try:
        generated = generate_case(opset=opset, model_cls=model_cls, case_seed=case_seed, config=generation)
        expected = case_fingerprint(generated)
        case.update(describe_case(generated), fingerprint=expected)
        del generated
        gc.collect()
        case_target = f"npu:{devices[index % len(devices)]}" if devices else target
        request = {"schema_version": 1, "seed": case_seed, "generation": asdict(generation),
                   "opset": snapshot_opset(opset), "expected_fingerprint": expected, "target": case_target}
        order = list(BACKENDS)
        if index % 2:
            order.reverse()
        stage = "benchmark"
        for backend in order:
            result = run_backend(spec=BACKENDS[backend], request=request,
                output_dir=out/"cases"/case["case_id"]/backend,
                cache_dir=out/"compile_cache"/backend, profile=profile)
            case["backends"][backend] = asdict(result)
        case["ascendc_speedup"] = comparable_speedup(case["backends"], expected)
        case["status"] = "compared" if case["ascendc_speedup"] is not None else "not_comparable"
    except ArtifactCleanupError:
        raise  # Do not keep running while large artifacts cannot be removed.
    except Exception:
        if stage == "benchmark":
            raise  # A runner/storage failure is fatal; backend errors are result records.
        case["status"] = "gen_error" if stage == "generation" else "error"
        case["error_stage"] = stage
        case["error"] = traceback.format_exc()[-2000:]
    case["duration_s"] = time.monotonic() - started
    case["performance_status"] = performance_status(case)
    case["correctness_status"] = correctness_status(case)
    return case


def _commit_case(summary, case, out):
    summary["cases"].append(case)
    summary["cases"].sort(key=lambda c: c["index"])
    summary["total"] += 1
    summary["comparable"] += int(case["status"] == "compared")
    summary["errors"] += int(case["status"] != "compared" or case["correctness_status"] != "pass")
    summary["performance_passed"] += int(case["performance_status"] == "pass")
    summary["performance_failed"] += int(case["performance_status"] == "fail")
    write_reports(summary, out)
    print(f"performance case={case['case_id']} status={case['status']} "
          f"speedup={case.get('ascendc_speedup')} completed={summary['total']}", flush=True)


def run_performance_session(*, out_dir, generation, profile: ProfileConfig, n_cases=None,
                            seed=None, repro_seed=None, target="npu", soc_tier="A3",
                            op_categories=None, devices=None, opset=None, resume=False, jobs=1):
    from .parallel import worker_targets
    targets = worker_targets(jobs, devices, target)
    if target.split(":", 1)[0] != "npu":
        raise ValueError("Performance comparison requires --target npu")
    if n_cases is not None and n_cases < 1:
        raise ValueError("n_cases must be positive or omitted for continuous operation")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with session_lock(out), stop_signals():
        return _run_locked(out=out, generation=generation, profile=profile, n_cases=n_cases,
            seed=seed, repro_seed=repro_seed, target=target, soc_tier=soc_tier,
            op_categories=op_categories, devices=devices, opset=opset, resume=resume, targets=targets)


def _run_locked(*, out, generation, profile, n_cases, seed, repro_seed, target, soc_tier,
                op_categories, devices, opset, resume, targets):
    if resume:
        summary = load_checkpoint(out, generation=generation, profile=profile, seed=seed,
            repro_seed=repro_seed, target=target, soc_tier=soc_tier,
            op_categories=op_categories, devices=devices)
        if n_cases is None:
            n_cases = summary["requested_cases"]
    else:
        if (out / "summary.json").exists():
            raise ValueError(f"{out} already has results; use --resume or a new --out-dir")
        batch_seed = seed if seed is not None else random.SystemRandom().getrandbits(32)
        summary = {"schema_version": 3, "mode": "performance_compare", "batch_seed": batch_seed,
            "generation": asdict(generation), "profile": asdict(profile), "soc_tier": soc_tier,
            "op_categories": op_categories, "devices": devices, "target": target,
            "repro_seed": repro_seed, "software": software_signature(), "category_ops": {},
            "total": 0, "comparable": 0, "errors": 0, "performance_passed": 0,
            "performance_failed": 0, "cases": [], "checkpoint": {
                "opset": None, "pending_seed": None, "rng_state": random.Random(batch_seed).getstate()}}
    limit = 1 if repro_seed is not None else n_cases
    if limit is not None and limit < summary["total"]:
        raise ValueError("--n-cases is the total session limit and cannot be below completed cases")
    summary["worker_targets"] = targets
    summary["requested_cases"] = limit
    summary["session_status"] = "running"
    summary.pop("stop_signal", None)
    summary.pop("session_error", None)
    rng = random.Random()
    rng.setstate(tuple_state(summary["checkpoint"]["rng_state"]))
    try:
        # A valid empty report exists even if stopped during initial opset probing.
        write_reports(summary, out)
        print(f"performance session pid={os.getpid()} batch_seed={summary['batch_seed']} "
              f"limit={limit if limit is not None else 'continuous'} out_dir={out}", flush=True)
        from nnsmith.materialize import Model
        from ..ascendc_backend import AscendC
        from ..ascendc_opset import probe_ascendc_opset, OP_CATEGORIES
        model_cls = Model.init("torch", backend_target="cpu")
        model_cls.add_seed_setter()
        if summary["checkpoint"]["opset"] is not None:
            saved = summary["checkpoint"]["opset"]
            if opset is not None and snapshot_opset(opset) != saved:
                raise ValueError("Cannot resume with a different op/dtype candidate pool")
            if opset is None:
                opset = restore_opset(saved, model_cls)
        elif opset is None:
            factory = AscendC(target=target)
            opset = probe_ascendc_opset(model_cls, factory, soc_tier=soc_tier,
                                       op_categories=op_categories, grad=False)
        summary["category_ops"] = {k: sorted(v) for k, v in OP_CATEGORIES.items()}
        summary["checkpoint"]["opset"] = snapshot_opset(opset)
        if len(targets) > 1 or 'pending_tasks' in summary['checkpoint']:
            from .parallel import run_parallel
            run_parallel(summary=summary, out=out, generation=generation, profile=profile,
                rng=rng, limit=limit, targets=targets, resume=resume)
        else:
            pending = summary["checkpoint"]["pending_seed"]
            if resume and pending is not None and not profile.keep_artifacts:
                compact_pending_case(out, f"{summary['total']:05d}-{pending}")
            while limit is None or summary["total"] < limit:
                checkpoint = summary["checkpoint"]
                case_seed = checkpoint["pending_seed"]
                if case_seed is None:
                    case_seed = repro_seed if repro_seed is not None else rng.getrandbits(32)
                    checkpoint.update(pending_seed=case_seed, rng_state=rng.getstate())
                # Persist the active seed before starting any potentially interrupted work.
                _save_checkpoint(summary, out)
                case = _measure_case(index=summary["total"], case_seed=case_seed, out=out,
                    generation=generation, profile=profile, opset=opset, model_cls=model_cls,
                    target=targets[0], devices=None)
                with defer_stop_signals():
                    checkpoint["pending_seed"] = None
                    _commit_case(summary, case, out)
        summary["session_status"] = "completed"
    except SessionStopped as exc:
        summary["session_status"] = "stopped"
        summary["stop_signal"] = int(exc.signum)
    except KeyboardInterrupt:
        summary["session_status"] = "stopped"
        summary["stop_signal"] = int(signal.SIGINT)
    except Exception:
        summary["session_status"] = "error"
        summary["session_error"] = traceback.format_exc()[-4000:]
        raise
    finally:
        write_reports(summary, out)
    return summary
