"""Bounded fresh-process legacy replay audit. Run only on the remote isolated copy.

Each pair varies hash seed, CPU thread environment and predecessor history.
Equal errors are NOT counted as successful replay. Traces/logs are opt-in
investigation artifacts, not default archived results or v1 golden acceptance.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from autofusedebugger.fuzz.diagnostics import first_divergence


def classify_pair(a, b, metadata_a, metadata_b):
    diff = first_divergence(a, b)
    if diff:
        return {"status": "diverged", "first_divergence": diff}
    if metadata_a["status"] != "generated" or metadata_b["status"] != "generated":
        return {"status": "matching_errors", "first_divergence": None}
    return {"status": "replayed", "first_divergence": None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", default="0,42,389609433")
    p.add_argument("--timeout", type=float, default=60)
    args = p.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)  # Never mingle old evidence with a new run.
    seeds = [int(value) for value in args.seeds.split(",")]
    records = []
    for catalog in ("core", "extended"):
        for dtype in ("float32", "float16"):
            for dynamic in (False, True):
                for seed in seeds:
                    label = f"{catalog}-{dtype}-{'dynamic' if dynamic else 'static'}-{seed}"
                    paths, metadata, failures = [], [], []
                    for index in range(2):
                        path = output / f"{label}-{index}.json"
                        command = [sys.executable, str(Path(__file__).with_name("fuzz_seed_probe.py")),
                                   "--seed", str(seed), "--catalog", catalog, "--dtype", dtype,
                                   "--output", str(path)]
                        if dynamic:
                            command.append("--dynamic")
                        if index:
                            command += ["--previous-seed", str(seed+1)]
                        env = dict(os.environ, PYTHONHASHSEED=str(index*123),
                                   OMP_NUM_THREADS=str(index*3+1), MKL_NUM_THREADS=str(index*3+1))
                        with path.with_suffix(".log").open("w") as log:
                            try:
                                result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                        timeout=args.timeout)
                                if result.returncode not in (0, 1) or not path.with_suffix(".meta.json").is_file():
                                    failures.append({"worker": index, "exit": result.returncode})
                            except subprocess.TimeoutExpired:
                                failures.append({"worker": index, "error": "timeout"})
                        if not failures:
                            paths.append(json.loads(path.read_text()))
                            metadata.append(json.loads(path.with_suffix(".meta.json").read_text()))
                    if failures:
                        record = {"case": label, "status": "worker_error", "failures": failures}
                    else:
                        record = {"case": label, **classify_pair(*paths, *metadata),
                                  "history": metadata[1]["history"],
                                  "generation_errors": [m["error"] for m in metadata]}
                    records.append(record)
                    (output / "summary.json").write_text(json.dumps(records, indent=2))
                    print(json.dumps(record), flush=True)
    return 1 if any(r["status"] in {"diverged", "worker_error"} for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
