#!/bin/bash
# Source this before running `af-fuzz` (or anything else that imports
# torch_npu) on an Ascend NPU pod reached via the CPU-jump-host ->
# remote_tool.sh connect -> pod path (e.g. cannlab_npu_ssh).
#
# Fixes two env issues seen on these pods that otherwise fail with:
#   ImportError: .../_ctypes.cpython-312-*.so: undefined symbol: _PyErr_SetLocaleString
#   ERROR: Failed to load the backend extension: torch_npu
#
# Root causes (see docs/af_fuzz_npu_env.md for the full writeup):
# 1. The venv's python3 binary links against the system apt libpython3.12
#    (/usr/lib/aarch64-linux-gnu/...), but the venv's _ctypes.so was built
#    against /opt/buildtools/python-3.12.9's own libpython -- different
#    ABI, missing symbol. Fix: put the buildtools lib dir first in
#    LD_LIBRARY_PATH.
# 2. torch_npu needs CANN's own runtime libs (libascend_hal.so etc.), which
#    only /etc/profile's `source .../cann/set_env.sh` sets up -- and that
#    only runs for login shells, not for `ssh host "command"` one-liners.
#
# Usage:
#   source autofusedebugger/nnsmith_integration/npu_env.sh
#   afx fuzz --out-dir ... --n-cases ... --target npu

export LD_LIBRARY_PATH=/opt/buildtools/python-3.12.9/lib:${LD_LIBRARY_PATH}
source "${HOME}/Ascend/cann/set_env.sh"
