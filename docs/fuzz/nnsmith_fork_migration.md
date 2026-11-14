# NNSmith fork migration — 2026-11-14

NNSmith source and all 14 AscendC patches now live in
https://github.com/Intmanxzx/nnsmith at commit
9d7f4d16aaeda931e1d79a2f91dbd09d72f86ab3 (distribution version
0.1.1+autofusex.2). The fork retains upstream history and Apache-2.0 notices.

AutofuseX declares the pinned fork in its execution extra. The application-specific
ATen catalog, adapters, case serialization and orchestration stay in AutofuseX.
The old patch application module, shell installer, patch files and publisher-side
patch builder have been removed. No historical data conversion was added.

Install from a checkout with:

~~~sh
python -m pip install '.[execution]'
~~~

For an offline/local distribution, build the fork wheel with standard pip wheel
and install it alongside the AutofuseX wheel. The deployment wrapper accepts
--nnsmith-wheel. The fork declares Z3 4.12.2.0, numpy<2 and setuptools 75.8.0
(the pinned Z3 imports pkg_resources), plus its normal Hydra dependencies.

## Validation on mlir

All validation used a new home under
/mnt/workspace/af-fuzz/nnsmith-fork-1114, sourcing $HOME/env.sh and reusing the
Torch 2.10.0+cpu / torch_npu 2.10 runtime. Both Ascend910_9362 cards were checked
for occupancy before workload validation. Existing campaign data was not used.

- Clean venv: pip installed both wheels and resolved their dependencies.
- NNSmith core tests: 15 passed.
- AutofuseX generation, determinism, ATen catalog, sampling, packaging and
  dependency-boundary tests: 525 passed.
- Source fingerprint after wheel installation matches
  c47e5611db39361963be90786c7ea208647df4b6eb9546ce911122b81721cba5.
- Strict AF-MLIR compiled a fork-generated ReLU graph on npu:0 and npu:1.
  Static and dynamic (two shapes) execution matched the CPU reference on both.
- The complete client installer succeeded using the local NNSmith wheel.
- Default staged replay exposed premature initialization of the default Triton
  backend by the coverage observer. Selecting AscendC during observer import
  fixes fresh-worker initialization; the installed CLI was rechecked.

Static seed: 14615016373310691366068996273867811327526335152131.
Dynamic seed: 16076518010641594284272681106584094347182267695107.

Logs and wheels are retained in the remote validation directory. This is a
migration and installation validation, not exhaustive NPU operator coverage or
a long-running campaign. GitHub HTTPS was unreachable from mlir, so remote
installation used wheels built from the same fork sources. Reusing the full
Torch/CANN environment exposes unrelated pre-existing optional-package pip
warnings; the clean dependency installation and the tested execution paths pass.
