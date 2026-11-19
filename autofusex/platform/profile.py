"""Capability profile: where af-* kernels beat or lose to another backend (design 6.6).

Point clouds are sampled (deterministic case_id stride) to at most MAX_POINTS
pairs so a request never scans more than one backend's heads. Results are
cached for CACHE_SECONDS because the data only moves with uploads.
"""
from __future__ import annotations

import math
import threading
import time

from . import queries
from .auth import Principal
from .issues import IssueError, idempotent
from .store import Store

MAX_POINTS = 20_000
CACHE_SECONDS = 300
MIN_PATTERN_CASES = 20
DRAFT_MIN_CASES = 50
DRAFT_MIN_LOG2 = 0.5  # |log2 ratio| >= 0.5, i.e. at least ~1.41x faster or slower

_cache: dict = {}
_lock = threading.Lock()


def _cached(key, compute):
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    value = compute()
    with _lock:
        if len(_cache) > 256:
            _cache.clear()
        _cache[key] = (now, value)
    return value


def axes(store: Store) -> list:
    return [r[0] for r in store.reader().execute(
        "SELECT name FROM feature_registry WHERE kind='numeric' AND axis=1 ORDER BY name")]


def _pairs(conn, x, y, arch, limit):
    total = conn.execute("SELECT COALESCE(SUM(count),0) FROM stats.perf_pair_buckets WHERE backend_x=? AND "
                         "backend_y=? AND dim='all'" + (" AND npu_arch=?" if arch else ""),
                         (min(x, y), max(x, y)) + ((arch,) if arch else ())).fetchone()[0]
    step = max(1, math.ceil(total / limit)) if total else 1
    sql = ("SELECT hx.case_id, ex.kernel_ns, ey.kernel_ns FROM execution_heads hx "
           "JOIN execution_heads hy ON hy.case_id=hx.case_id AND hy.npu_arch=hx.npu_arch AND hy.backend=? "
           "JOIN executions ex ON ex.exec_id=hx.exec_id JOIN executions ey ON ey.exec_id=hy.exec_id "
           "WHERE hx.backend=? AND hx.case_id % ? = 0 AND ex.function_status='pass' AND ey.function_status='pass' "
           "AND COALESCE(ex.accuracy_status,'')<>'fail' AND COALESCE(ey.accuracy_status,'')<>'fail' "
           "AND ex.kernel_ns>0 AND ey.kernel_ns>0")
    args = [y, x, step]
    if arch:
        sql += " AND hx.npu_arch=?"
        args.append(arch)
    rows = conn.execute(sql + " LIMIT ?", args + [limit]).fetchall()
    return [(r[0], math.log2(r[1] / r[2])) for r in rows], total, step


def points(store: Store, x: str, y: str, fx: str, fy: str, arch=None) -> dict:
    if x == y:
        raise IssueError("invalid", "x and y must differ")

    def compute():
        conn = store.reader()
        ids = {r[0]: r[1] for r in conn.execute(
            "SELECT name, feature_id FROM feature_registry WHERE kind='numeric' AND name IN (?,?)", (fx, fy))}
        if fx not in ids or fy not in ids:
            raise IssueError("invalid", "fx and fy must be numeric features, see /api/profile/axes")
        pairs, total, step = _pairs(conn, x, y, arch, MAX_POINTS)
        values = {fx: {}, fy: {}}
        case_ids = [p[0] for p in pairs]
        for name in {fx, fy}:
            for chunk in _chunks(case_ids, 500):
                for case_id, value in conn.execute(
                        f"SELECT case_id, value FROM case_metrics WHERE feature_id=? AND case_id IN "
                        f"({','.join('?' * len(chunk))})", [ids[name], *chunk]):
                    values[name][case_id] = value
        out = [[values[fx][c], values[fy][c], round(r, 4), c] for c, r in pairs
               if c in values[fx] and c in values[fy]]
        return {"x": x, "y": y, "fx": fx, "fy": fy, "npu_arch": arch, "total_pairs": total, "sample_step": step,
                "columns": [fx, fy, "log2_ratio", "case_id"], "points": out}
    return _cached(("points", x, y, fx, fy, arch), compute)


def patterns(store: Store, x: str, y: str, arch=None) -> dict:
    """Per-op geometric mean ratio on a sample, plus how strongly each dimension separates results."""
    if x == y:
        raise IssueError("invalid", "x and y must differ")

    def compute():
        conn = store.reader()
        pairs, total, step = _pairs(conn, x, y, arch, MAX_POINTS)
        ratios = dict(pairs)
        op_feature = (conn.execute("SELECT feature_id FROM feature_registry WHERE name='op_display'").fetchone()
                      or conn.execute("SELECT feature_id FROM feature_registry WHERE name='op'").fetchone())
        sums: dict = {}
        if op_feature is not None:
            for chunk in _chunks(list(ratios), 500):
                for case_id, op in conn.execute(
                        "SELECT f.case_id, v.value FROM case_features f JOIN feature_values v ON v.value_id=f.value_id "
                        f"WHERE f.feature_id=? AND f.case_id IN ({','.join('?' * len(chunk))})",
                        [op_feature[0], *chunk]):
                    entry = sums.setdefault(op, [0, 0.0])
                    entry[0] += 1
                    entry[1] += ratios[case_id]
        # Small populations still get a profile; large ones need more cases per op.
        min_cases = max(5, min(MIN_PATTERN_CASES, len(pairs) // 20))
        ops = [{"op": op, "cases": n, "geomean_ratio": round(2 ** (s / n), 4), "log2": round(s / n, 4)}
               for op, (n, s) in sums.items() if n >= min_cases]
        ops.sort(key=lambda o: o["log2"])
        importance = []
        for dim in queries.dims(store):
            if dim == "all":
                continue
            groups = [g for g in queries.perf_pair(store, x, y, dim, arch)["groups"] if g["total"]]
            n = sum(g["total"] for g in groups)
            if len(groups) < 2 or not n:
                continue
            means = [(g["total"], math.log2(g["geomean_ratio"])) for g in groups]
            overall = sum(t * m for t, m in means) / n
            spread = math.sqrt(sum(t * (m - overall) ** 2 for t, m in means) / n)
            importance.append({"dim": dim, "spread_log2": round(spread, 4), "groups": len(groups)})
        importance.sort(key=lambda d: -d["spread_log2"])
        return {"x": x, "y": y, "npu_arch": arch, "sampled_pairs": len(pairs), "total_pairs": total, "min_cases": min_cases,
                "best_ops": ops[:15], "worst_ops": list(reversed(ops[-15:])), "importance": importance}
    return _cached(("patterns", x, y, arch), compute)


def create_drafts(store: Store, principal: Principal, body: dict):
    """Turn strong, well-sampled op patterns into 'measured' knowledge drafts for admin review."""
    principal.require_admin()
    x, y, arch = body.get("x"), body.get("y"), body.get("npu_arch")
    if not all(isinstance(v, str) for v in (x, y)) or (arch is not None and not isinstance(arch, str)):
        raise IssueError("invalid", "x and y are required backend names")
    found = patterns(store, x, y, arch)
    candidates = [o for o in found["best_ops"] + found["worst_ops"]
                  if o["cases"] >= DRAFT_MIN_CASES and abs(o["log2"]) >= DRAFT_MIN_LOG2]

    def action(conn):
        from . import knowledge
        created = []
        for item in {o["op"]: o for o in candidates}.values():
            faster = item["log2"] < 0
            factor = 2 ** abs(item["log2"])
            title = f"{item['op']}：{x} kernel 比 {y} {'快' if faster else '慢'} {factor:.2f}x"
            if conn.execute("SELECT 1 FROM knowledge WHERE title=?", (title,)).fetchone():
                continue
            fields = knowledge.validate({
                "kind": "capability", "title": title,
                "keywords": [item["op"].replace(" ", "_")[:64], x, y, "性能画像"],
                "ops": [item["op"].replace(" ", "_")[:64]], "backends": [x, y],
                "symptom": f"含 {item['op']} 的用例中，{x} 与 {y} 的 kernel 耗时几何平均比为 "
                           f"{item['geomean_ratio']}（抽样 {item['cases']} 个用例，arch={arch or '全部'}）。",
                "root_cause": "统计结论，尚未分析根因。",
                "solution": (f"含 {item['op']} 的子图可优先交给 {x}。" if faster
                             else f"含 {item['op']} 的子图在 {x} 上偏慢，可考虑交给 {y} 或分析调度/融合策略。"),
                "applicability": "基于平台当前执行头的抽样统计，随版本变化可能失效；发布前需人工复核。"})
            created.append(knowledge.create(conn, principal, fields, confidence="measured", status="draft"))
        return {"created": created, "candidates": len(candidates)}
    return idempotent(store, principal, "profile-drafts", body.pop("request_key", None), body, action)


def _chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]
