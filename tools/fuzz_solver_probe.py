"""Locate legacy solver-history drift without claiming the entire generator fixed.

Writes raw and fresh-context traces for seeds 0,42,389609433, with preceding
cases and reversed catalog. Fresh-context mode uses the existing production
_seeded_solver_context; no shared solver/dependency source is modified.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

from autofusex.fuzz.diagnostics import GenerationTrace, first_divergence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    from nnsmith.abstract.op import Add, Mul, Neg, ReLU
    from nnsmith.graph_gen import model_gen
    from autofusex.nnsmith_integration.generation import _seeded_solver_context
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    ops = [Add, Mul, Neg, ReLU]
    def graph(seed, catalog, isolated):
        with _seeded_solver_context(seed) if isolated else nullcontext():
            ir = model_gen(catalog, method="symbolic-cinit", seed=seed, init_fp=True,
                           max_nodes=4, timeout_ms=10000, concr_ph_dim_rng=(1, 8)).make_concrete()
            trace = GenerationTrace()
            trace.graph(ir)
            return trace.stages
    results = []
    for isolated in (False, True):
        for seed in (0, 42, 389609433):
            first = graph(seed, ops, isolated)
            graph(seed+1, ops, isolated)
            second = graph(seed, list(reversed(ops)), isolated)
            label = f"{'context' if isolated else 'raw'}-{seed}"
            for suffix, value in (("a", first), ("b", second)):
                (directory / f"{label}-{suffix}.json").write_text(json.dumps(value, indent=2))
            results.append({"mode": label, "first_divergence": first_divergence(first, second)})
    (directory / "comparison.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
