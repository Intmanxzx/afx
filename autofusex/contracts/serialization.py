"""Deterministic wire encoding shared by workloads and execution records."""
import hashlib
import json


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()
