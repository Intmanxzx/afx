"""Capacity benchmark for the platform store (design 8, P0 acceptance).

Ingests synthetic cases through the real Ingestor path (validation, case and
environment upsert, heads, incremental aggregates; HTTP and NPU execution are
not exercised), then times the read queries used by the overview and the case
list. Synthetic distributions are not production data; treat results as a
query-capacity baseline only.

usage: python tools/platform/bench_platform.py DATA_DIR --cases 1000000
"""
from __future__ import annotations

def _short(n):
    digits, alphabet = "", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    while n:
        n, d = divmod(n, 36)
        digits = alphabet[d] + digits
    return "E" + (digits or "0").zfill(26)


import argparse
import json
import random
import statistics
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autofusex.platform import auth, queries  # noqa: E402
from autofusex.platform.ingest import Ingestor  # noqa: E402
from autofusex.platform.store import Store  # noqa: E402

BACKENDS = ("af-mlir", "af-legacy", "triton-npu")
CATEGORIES = ("elementwise", "reduce", "broadcast", "view", "compare", "cast")
DTYPES = ("float16", "bfloat16", "float32")
STAGES = ("schedule", "lowering", "codegen", "launch")
ENVS = [{"cann": f"8.{i % 3}", "torch": "2.6", "backend_version": f"0.{i}", "os": "ubuntu22",
         "cpu_arch": "arm", "afx": "0.4.0"} for i in range(6)]
ALPHABET = string.digits + string.ascii_lowercase


def seed(rng: random.Random) -> str:
    return _short(rng.getrandbits(128))


def make_items(rng: random.Random, start: int, count: int):
    items = []
    for n in range(start, start + count):
        cats = sorted(rng.sample(CATEGORIES, rng.randint(1, 3)))
        dtypes = sorted(rng.sample(DTYPES, rng.randint(1, 2)))
        numel = 2 ** rng.randint(4, 24)
        case = {"case_key": seed(rng), "case_source": "nnsmith", "is_dynamic": rng.random() < 0.3,
                "shape_count": 1, "node_count": rng.randint(2, 64), "op_types": [c + "_op" for c in cats],
                "dtypes": dtypes, "max_dim": 2 ** rng.randint(2, 12), "max_numel": numel,
                "features": {"op_category": cats}}
        env = rng.choice(ENVS)
        base_kernel = rng.lognormvariate(10, 1)
        for backend in BACKENDS:
            roll = rng.random()
            item = {"exec_uuid": f"x{n:09d}{backend}", "case": case, "backend": backend, "npu_arch": "910b",
                    "env": env, "e2e_ns": rng.randint(10**6, 10**9), "created_at": 1_700_000_000_000 + n}
            if roll < 0.25:
                item.update(function_status="compile_fail", fail_stage=rng.choice(STAGES),
                            log_summary=f"error in {rng.choice(STAGES)}: code {rng.randint(1, 300)}",
                            log=f"[E] failure {n} " + "trace line\n" * rng.randint(5, 40))
            elif roll < 0.32:
                item.update(function_status="runtime_fail", log_summary="aclrtSynchronize failed")
            elif roll < 0.42:
                item.update(function_status="pass", accuracy_status="fail", accuracy_ref="cpu",
                            max_abs_err=rng.random(), kernel_ns=int(base_kernel * rng.uniform(0.5, 2)))
            else:
                item.update(function_status="pass", accuracy_status="pass", accuracy_ref="cpu",
                            kernel_ns=int(base_kernel * rng.uniform(0.5, 2)))
            items.append(item)
    return items


def timed(fn, runs=30):
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    pick = lambda q: samples[min(len(samples) - 1, int(q * len(samples)))]  # noqa: E731
    return {"p50": round(statistics.median(samples), 2), "p95": round(pick(0.95), 2), "p99": round(pick(0.99), 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--batch", type=int, default=150, help="cases per upload (x3 executions)")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()
    store = Store(args.data_dir)
    report = {"cases": args.cases, "executions": args.cases * len(BACKENDS)}
    if not args.skip_ingest:
        if not store.reader().execute("SELECT 1 FROM users WHERE user_id='bench'").fetchone():
            auth.add_user(store, "bench", "developer")
        principal = auth.authenticate(store, "Bearer " + auth.issue_token(store, "bench"))
        ingestor, rng = Ingestor(store), random.Random(7)
        t0, done = time.perf_counter(), 0
        while done < args.cases:
            count = min(args.batch, args.cases - done)
            body = {"request_key": f"bench-{done}", "client": {"machine_id": "bench"},
                    "run": {"run_uuid": f"bench-run-{done // 30000}", "kind": "fuzz"},
                    "items": make_items(rng, done, count)}
            raw = json.dumps(body).encode()
            result = ingestor.upload(principal, body, raw)
            assert result["accepted"] == count * len(BACKENDS), result["results"][:3]
            done += count
            if done % 50_000 == 0 or done == args.cases:
                rate = done * len(BACKENDS) / (time.perf_counter() - t0)
                print(f"ingested {done} cases, {rate:.0f} executions/s", file=sys.stderr, flush=True)
        report["ingest_seconds"] = round(time.perf_counter() - t0, 1)
        report["ingest_executions_per_s"] = round(report["executions"] / report["ingest_seconds"])
    some_key = store.reader().execute("SELECT case_key FROM cases ORDER BY case_id LIMIT 1 OFFSET 1000").fetchone()[0]
    cursor = queries.case_page(store, limit=50)["next_cursor"]
    report["queries_ms"] = {
        "case_page_first": timed(lambda: queries.case_page(store, limit=50)),
        "case_page_next": timed(lambda: queries.case_page(store, limit=50, after=cursor)),
        "case_page_mlir_fail": timed(lambda: queries.case_page(store, limit=50, backend="af-mlir", status="fail")),
        "case_page_mlir_accuracy_fail": timed(
            lambda: queries.case_page(store, limit=50, backend="af-mlir", status="accuracy_fail")),
        "case_by_key": timed(lambda: queries.case_page(store, case_key=some_key)),
        "overview_pipeline": timed(lambda: queries.pipeline(store, "910b")),
        "overview_backend_all": timed(lambda: queries.backend_stats(store, "af-mlir", "all", "910b")),
        "overview_backend_category": timed(lambda: queries.backend_stats(store, "af-mlir", "op_category", "910b")),
        "overview_perf_all": timed(lambda: queries.perf_pair(store, "af-mlir", "triton-npu", "all", "910b")),
        "overview_perf_dtype": timed(lambda: queries.perf_pair(store, "af-legacy", "triton-npu", "dtype", "910b")),
    }
    report["disk_bytes"] = {p.name: p.stat().st_size for p in sorted(store.dir.glob("*.db*"))}
    conn = store.reader()
    report["stats_rows"] = {
        "stat_buckets": conn.execute("SELECT COUNT(*) FROM stats.stat_buckets").fetchone()[0],
        "perf_pair_buckets": conn.execute("SELECT COUNT(*) FROM stats.perf_pair_buckets").fetchone()[0]}
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
