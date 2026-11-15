"""Bounded live-NPU probe. Use isolated caches and confirm idle hardware first.

Callable creation is not compilation; first invocation is combined compile/execute.
No artifact handoff or CPU-only compilation capability is inferred by this probe.
"""
import argparse
import json
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("ascendc", "triton"), required=True)
    args = p.parse_args()
    # Probe needs no NNSmith; set compile budget explicitly before Torch import.
    import os
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import torch
    import torch_npu  # noqa: F401
    x = torch.arange(32, dtype=torch.float32).reshape(4, 8).npu()
    y = torch.ones(1, 8).npu()
    def fn(a, b):
        return (a + b).relu()
    start = time.monotonic()
    options = {"npu_backend": "ascendc"} if args.backend == "ascendc" else {}
    compiled = torch.compile(fn, backend="inductor", fullgraph=True, options=options)
    print(json.dumps({"stage": "callable_created_not_compiled", "seconds": time.monotonic()-start}), flush=True)
    for stage in ("first_call_combined", "second_call"):
        start = time.monotonic()
        actual = compiled(x, y)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, fn(x, y))
        print(json.dumps({"stage": stage, "seconds": time.monotonic()-start}), flush=True)
    print(json.dumps({"cpu_only_compile": "unverified", "cross_process_handoff": "unverified"}))


if __name__ == "__main__":
    main()
