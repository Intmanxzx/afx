"""Generates a human-readable repro script alongside a dumped failing case.

The script is a thin, readable shell around the same deterministic data
`BugReport.dump()` already wrote (model.pth / gir.pkl / oracle.pkl) -- it
loads them back and re-runs the ascendc backend's verify step. It never
re-invokes NNSmith's random generation, so running it twice reproduces the
exact same compile/execute/compare that originally failed.

The rtol/atol baked into the generated script are whatever the fuzz loop's
`_tolerance_for_ir(ir)` decided for this specific case at generation time
(fp32/fp64/int graphs get nnsmith's original fixed defaults, rtol=1e-2 /
atol=1e-3; graphs touching a low-precision float dtype such as fp16 get the
10x-looser rtol=1e-1 / atol=1e-2) -- not a value hardcoded in this module.
This keeps the repro script's pass/fail verdict consistent with the verdict
the main fuzz loop already reached for the same case.
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = '''"""Repro script for a failed NNSmith -> autofuse case.

Generated alongside this directory's model.pth / gir.pkl / oracle.pkl.
Loads that exact (weights, graph, input/output) triple back -- no random
generation involved -- and re-runs the same compile + execute + compare that
originally failed.

The rtol/atol below are not nnsmith's fixed defaults -- they are the
dtype-aware tolerance `_tolerance_for_ir` picked for this case when it was
generated (loosened for low-precision float dtypes such as fp16, otherwise
nnsmith's original fp32 defaults), baked in at generation time so this
script's verdict matches the fuzz loop's verdict for the same case.

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
    bug = factory.verify_testcase(testcase, equal_nan=True, rtol={rtol!r}, atol={atol!r})

    if bug is None:
        print("PASS -- this case no longer reproduces the failure.")
    else:
        print(f"FAIL -- symptom={{bug.symptom.value}} stage={{bug.stage.value}}")
        print(bug.log)


if __name__ == "__main__":
    main()
'''


def write_repro_script(
    case_dir: str | Path,
    target: str = "npu",
    rtol: float = 1e-2,
    atol: float = 1e-3,
) -> Path:
    """Write repro.py into `case_dir`.

    `rtol`/`atol` default to nnsmith's own fixed tolerance (1e-2/1e-3), so
    callers that don't pass them get the old, unchanged behavior. Callers
    that already computed dtype-aware tolerance for this case (see
    `fuzz_loop._tolerance_for_ir`) should pass it through here so the
    generated script's verdict stays consistent with the one just reached.
    """
    case_dir = Path(case_dir)
    script_path = case_dir / "repro.py"
    script_path.write_text(
        _TEMPLATE.format(target=target, rtol=rtol, atol=atol), encoding="utf-8"
    )
    return script_path
