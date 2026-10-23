"""Generates a human-readable repro script alongside a dumped failing case.

The script is a thin, readable shell around the same deterministic data
`BugReport.dump()` already wrote (model.pth / gir.pkl / oracle.pkl) -- it
loads them back and re-runs the ascendc backend's verify step. It never
re-invokes NNSmith's random generation, so running it twice reproduces the
exact same compile/execute/compare that originally failed.
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = '''"""Repro script for a failed NNSmith -> autofuse case.

Generated alongside this directory's model.pth / gir.pkl / oracle.pkl.
Loads that exact (weights, graph, input/output) triple back -- no random
generation involved -- and re-runs the same compile + execute + compare that
originally failed.

Usage: python3 repro.py
"""
from pathlib import Path

from nnsmith.materialize import Model, TestCase

from autofusedebugger.nnsmith_integration.ascendc_backend import AscendC

CASE_DIR = Path(__file__).parent


def main() -> None:
    model_cls = Model.init("torch", backend_target={target!r})
    testcase = TestCase.load(model_cls, str(CASE_DIR))

    factory = AscendC(target={target!r})
    bug = factory.verify_testcase(testcase, equal_nan=True)

    if bug is None:
        print("PASS -- this case no longer reproduces the failure.")
    else:
        print(f"FAIL -- symptom={{bug.symptom.value}} stage={{bug.stage.value}}")
        print(bug.log)


if __name__ == "__main__":
    main()
'''


def write_repro_script(case_dir: str | Path, target: str = "npu") -> Path:
    case_dir = Path(case_dir)
    script_path = case_dir / "repro.py"
    script_path.write_text(_TEMPLATE.format(target=target), encoding="utf-8")
    return script_path
