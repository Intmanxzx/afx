"""Explicit diagnostic capture on an isolated dependency copy, not a replay cache."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import resource

from autofusex.fuzz.diagnostics import GenerationTrace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--previous-seed", type=int)
    p.add_argument("--dynamic", action="store_true")
    p.add_argument("--catalog", choices=("core", "extended"), default="core")
    p.add_argument("--dtype", choices=("native", "float32", "float16"), default="native")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from nnsmith.abstract.op import Add, Mul, Neg, ReLU
    from nnsmith.materialize import Model
    from autofusex.nnsmith_integration.generation import GenerationConfig, generate_case
    config = GenerationConfig(max_nodes=4, tensor_scale="small", concr_ph_dim_rng=(1, 4),
                              max_elem_per_tensor=256, dynamic_shapes=args.dynamic)
    model = Model.init("torch", "cpu")
    ops = [Add, Mul, Neg, ReLU]  # Explicit catalog; no device capability filtering.
    if args.catalog == "extended":
        from nnsmith.abstract.op import Transpose, Reshape, ReduceMin
        ops += [Transpose, Reshape, ReduceMin]
    if args.dtype != "native":
        from nnsmith.abstract.dtype import DType
        dtype = getattr(DType, args.dtype)
        for op in ops:
            pairs = [(i, o) for i, o in zip(op.in_dtypes, op.out_dtypes)
                     if all(d == dtype for d in (*i, *o))]
            if not pairs:
                raise ValueError(f"{op.name()} has no paired signature for {args.dtype}")
            op.in_dtypes = [i for i, _ in pairs]
            op.out_dtypes = [o for _, o in pairs]
    history = {"status": "not_requested"}
    if args.previous_seed is not None:
        # A rejected predecessor must not prevent observing the target case.
        try:
            generate_case(opset=ops, model_cls=model, case_seed=args.previous_seed, config=config)
            history = {"status": "generated", "seed": args.previous_seed}
        except Exception as exc:
            history = {"status": "error", "seed": args.previous_seed,
                       "type": type(exc).__name__, "message": str(exc)[:500]}
    trace = GenerationTrace()
    configuration = {"seed": args.seed, "config": asdict(config),
                     "catalog": [op.name() for op in ops], "protocol": "legacy-unverified"}
    # Preserve the original core/native diagnostic golden encoding.
    if args.dtype != "native":
        configuration["dtype"] = args.dtype
    trace.record("configuration", configuration)
    start = time.monotonic()
    status, error = "generated", None
    try:
        generate_case(opset=ops, model_cls=model, case_seed=args.seed, config=config, trace=trace)
    except Exception as exc:
        status = "error"
        error = {"type": type(exc).__name__, "message": str(exc)[:500]}
        trace.record("error", error)
    output = Path(args.output)
    output.write_text(json.dumps(trace.stages, indent=2), encoding="utf-8")
    metadata = {"status": status, "error": error, "history": history,
                "seconds": time.monotonic()-start,
                "maxrss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "trace_bytes": output.stat().st_size, "digest": trace.digest}
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata))
    return 0 if status == "generated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
