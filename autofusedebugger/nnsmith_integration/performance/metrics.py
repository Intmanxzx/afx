"""Measurement validity and performance verdicts, independent of HTML."""
import math


def comparable_speedup(results: dict, expected_fingerprint: str):
    a, t = results.get("ascendc", {}), results.get("triton-ascend", {})
    if not expected_fingerprint:
        return None
    for value in (a, t):
        time = value.get("mean_kernel_us")
        if (value.get("status") != "ok" or value.get("fingerprint") != expected_fingerprint
                or not isinstance(time, (int, float)) or not math.isfinite(time) or time <= 0):
            return None
    return t["mean_kernel_us"] / a["mean_kernel_us"]


def performance_status(case: dict):
    ratio = comparable_speedup(case.get("backends", {}), case.get("fingerprint"))
    return "error" if ratio is None else "pass" if ratio >= 1 else "fail"


def correctness_status(case: dict):
    backends = [case.get("backends", {}).get(name, {}) for name in ("ascendc", "triton-ascend")]
    if any(b.get("correctness") == "fail" or (b and b.get("correctness") != "pass") for b in backends):
        return "fail"
    if all(b.get("correctness") == "pass" for b in backends):
        return "pass"
    return "gen_error"
