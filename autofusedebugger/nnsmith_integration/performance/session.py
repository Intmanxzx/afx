"""Serial case/backend scheduling keeps timed kernels free of benchmark overlap."""
from dataclasses import asdict
import gc
from pathlib import Path
import random
import traceback
import time

from .backends import BACKENDS
from .cases import case_fingerprint, describe_case, snapshot_opset
from .report import write_reports
from .metrics import comparable_speedup, performance_status, correctness_status
from .runner import run_backend
from .schema import ProfileConfig


def run_performance_session(*, out_dir, generation, profile: ProfileConfig, n_cases=100,
                            seed=None, repro_seed=None, target="npu", soc_tier="A3",
                            op_categories=None, devices=None, opset=None):
    from nnsmith.materialize import Model
    from ..ascendc_backend import AscendC
    from ..ascendc_opset import probe_ascendc_opset, OP_CATEGORIES
    from ..generation import generate_case

    if target.split(":", 1)[0] != "npu":
        raise ValueError("Performance comparison requires --target npu")
    if n_cases < 1:
        raise ValueError("n_cases must be positive")
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_cls = Model.init("torch", backend_target="cpu")
    model_cls.add_seed_setter()
    if opset is None:
        factory = AscendC(target=target)
        opset = probe_ascendc_opset(model_cls, factory, soc_tier=soc_tier,
                                   op_categories=op_categories, grad=False)
    snapshot = snapshot_opset(opset)
    batch_seed = seed if seed is not None else random.SystemRandom().getrandbits(32)
    rng = random.Random(batch_seed)
    seeds = [repro_seed] if repro_seed is not None else [rng.getrandbits(32) for _ in range(n_cases)]
    summary = {"schema_version": 2, "mode": "performance_compare", "batch_seed": batch_seed,
               "generation": asdict(generation), "profile": asdict(profile), "soc_tier": soc_tier,
               "category_ops": {k: sorted(v) for k, v in OP_CATEGORIES.items()},
               "performance_passed": 0, "performance_failed": 0,
               "op_categories": op_categories, "total": 0, "comparable": 0, "errors": 0, "cases": []}
    for index, case_seed in enumerate(seeds):
        case = {"seed": case_seed, "index": index, "case_id": f"{index:05d}-{case_seed}", "backends": {}}
        started = time.monotonic()
        stage = "generation"
        try:
            generated = generate_case(opset=opset, model_cls=model_cls, case_seed=case_seed, config=generation)
            expected = case_fingerprint(generated)
            case.update(describe_case(generated), fingerprint=expected)
            del generated
            gc.collect()
            case_target = f"npu:{devices[index % len(devices)]}" if devices else target
            request = {"schema_version": 1, "seed": case_seed, "generation": asdict(generation),
                       "opset": snapshot, "expected_fingerprint": expected, "target": case_target}
            # Alternate which backend runs first across cases to reduce order bias.
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
        except Exception:
            case["status"] = "gen_error" if stage == "generation" else "error"
            case["error_stage"] = stage
            case["error"] = traceback.format_exc()[-2000:]
        case["duration_s"] = time.monotonic() - started
        case["performance_status"] = performance_status(case)
        case["correctness_status"] = correctness_status(case)
        summary["performance_passed"] += int(case["performance_status"] == "pass")
        summary["performance_failed"] += int(case["performance_status"] == "fail")
        summary["cases"].append(case)
        summary["total"] += 1
        summary["comparable"] += int(case["status"] == "compared")
        summary["errors"] += int(case["status"] != "compared" or case["correctness_status"] != "pass")
        write_reports(summary, out)
        print(f"performance case={case['case_id']} status={case['status']} speedup={case.get('ascendc_speedup')}", flush=True)
    return summary
