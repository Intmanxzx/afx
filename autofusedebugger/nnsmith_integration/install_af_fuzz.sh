#!/bin/bash
# One-shot installer for af-fuzz: installs Autofuse-Debugger itself (editable,
# from this checkout) plus the exact nnsmith commit its ascendc patches target.
# Run this once per environment (fresh pod / fresh venv); after that, just
# `source npu_env.sh` in each new shell and run `autofusedbg af-fuzz ...`.
#
# Usage:
#   cd Autofuse-Debugger   # this checkout, the one containing this script
#   bash autofusedebugger/nnsmith_integration/install_af_fuzz.sh
#
# What it does:
#   1. pip install -e . for Autofuse-Debugger (this checkout)
#   2. git clone nnsmith at the pinned commit the ascendc patches target,
#      into a sibling directory (../nnsmith relative to this checkout),
#      skipped if that directory already exists
#   3. pip install -e . for nnsmith
#
# It does NOT source npu_env.sh for you (that must happen in every new shell,
# not just once at install time) and does NOT run af-fuzz itself.

set -euo pipefail

NNSMITH_REPO="https://github.com/ise-uiuc/nnsmith.git"
NNSMITH_COMMIT="bc0af42"  # pinned: ascendc patches are written against this baseline, do not bump

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NNSMITH_DIR="${REPO_ROOT}/../nnsmith"

echo "== [1/3] Installing Autofuse-Debugger (editable) from ${REPO_ROOT} =="
python3 -m pip install -e "${REPO_ROOT}"

if [ -d "${NNSMITH_DIR}" ]; then
  echo "== [2/3] ${NNSMITH_DIR} already exists, skipping clone =="
else
  echo "== [2/3] Cloning nnsmith into ${NNSMITH_DIR} =="
  git clone "${NNSMITH_REPO}" "${NNSMITH_DIR}"
  git -C "${NNSMITH_DIR}" checkout "${NNSMITH_COMMIT}"
fi

echo "== [3/3] Installing nnsmith (editable) from ${NNSMITH_DIR} =="
python3 -m pip install -e "${NNSMITH_DIR}"

echo
echo "Done. Next steps in this and every new shell:"
echo "  source ${SCRIPT_DIR}/npu_env.sh   # edit the library/CANN paths inside it first if needed"
echo "  autofusedbg af-fuzz --out-dir /tmp/af_fuzz_run"
