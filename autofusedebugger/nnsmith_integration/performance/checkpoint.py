"""Resume validation and lazy seed state; no models/tensors are serialized."""
from dataclasses import asdict
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import sys


def software_signature():
    versions = {}
    for name in ("torch", "torch-npu", "triton", "nnsmith", "z3-solver", "numpy", "ml-dtypes"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    spec = util.find_spec("nnsmith")
    digest = hashlib.sha256()
    if spec and spec.submodule_search_locations:
        root = Path(next(iter(spec.submodule_search_locations)))
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    generator = Path(__file__).parent.parent / "generation.py"
    return {"python": list(sys.version_info[:3]), "packages": versions,
            "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
            "nnsmith_sha256": digest.hexdigest()}


def tuple_state(value):
    return tuple(tuple_state(v) for v in value) if isinstance(value, list) else value


def load_checkpoint(out, *, generation, profile, seed, repro_seed, target, soc_tier,
                    op_categories, devices):
    summary = json.loads((out / "summary.json").read_text())
    if summary.get("mode") != "performance_compare" or "checkpoint" not in summary:
        raise ValueError("This directory has no resumable performance checkpoint")
    expected = {"generation": asdict(generation), "target": target, "soc_tier": soc_tier,
                "op_categories": op_categories, "devices": devices, "repro_seed": repro_seed,
                "software": software_signature()}
    expected = json.loads(json.dumps(expected))
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"Cannot resume with different {key}; use the original settings")
    for key in ("warmup_steps", "steps"):
        if summary["profile"][key] != getattr(profile, key):
            raise ValueError(f"Cannot resume with different profile.{key}")
    if seed is not None and seed != summary["batch_seed"]:
        raise ValueError("Cannot resume with a different batch seed")
    cases = summary['cases']
    indices = [c['index'] for c in cases]
    cp = summary['checkpoint']
    if summary['total'] != len(cases) or indices != sorted(set(indices)):
        raise ValueError('Inconsistent performance checkpoint case indices')
    if 'pending_tasks' in cp:
        assigned = indices + [t['index'] for t in cp['pending_tasks']]
        if sorted(assigned) != list(range(cp['next_index'])):
            raise ValueError('Inconsistent pending case indices in performance checkpoint')
    elif indices != list(range(summary['total'])):
        raise ValueError('Inconsistent serial performance checkpoint indices')
    # Timeout and artifact retention may change; measured workload must match.
    summary["profile"] = asdict(profile)
    return summary
