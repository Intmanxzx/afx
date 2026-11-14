#!/usr/bin/env bash
# Install the frozen NNSmith source as a wheel, without modifying shared checkouts.
set -euo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo 'Usage: PYTHON_BIN=python3 NNSMITH_DIR=/path/to/private/dependency bash install_af_fuzz.sh'
  echo 'Optional: NNSMITH_REPO (clone URL/local repository), INSTALL_AF_FUZZ_SKIP_PIP=1 (source verification only)'
  exit 0
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
NNSMITH_COMMIT="bc0af42c7d5fc4fd201efb76e5313f6298c2d573"
NNSMITH_REPO="${NNSMITH_REPO:-https://github.com/ise-uiuc/nnsmith.git}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNSMITH_DIR="${NNSMITH_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/autofusex/nnsmith-$NNSMITH_COMMIT}"
PATCH_ORDER=(
  ascendc_index_select
  ascendc_gather_layernorm
  ascendc_tolerance_params
  ascendc_bfloat16
  ascendc_rank_and_axis_weighting
  ascendc_value_distribution
  ascendc_make_oracle_kwargs
  ascendc_refine_weights_value_dist
  ascendc_init_fp_dtype_diversity
  ascendc_dim_size_distribution
  ascendc_reduce_min_max_amin_amax
  ascendc_multi_axis_reduce
  ascendc_seed_reproducibility
  ascendc_seeded_numeric_search
)
die() { echo "ERROR: $*" >&2; exit 1; }
command -v git >/dev/null || die 'git is required'
command -v patch >/dev/null || die 'patch is required'
EXPECTED="$("$PYTHON_BIN" - "$SCRIPT_DIR/../fuzz/generation/runtime.py" <<'PY'
import re, sys
print(re.search(r"SOURCE_SHA256\s*=\s*'([0-9a-f]{64})'", open(sys.argv[1]).read()).group(1))
PY
)"
tree_hash() {
  "$PYTHON_BIN" - "$NNSMITH_DIR/nnsmith" <<'PY'
import hashlib, sys
from pathlib import Path
root=Path(sys.argv[1]); digest=hashlib.sha256()
for path in sorted(root.rglob('*.py')):
    digest.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes())
print(digest.hexdigest())
PY
}
# Never reset/clean a caller's checkout. This directory is a managed dependency.
if [[ -e $NNSMITH_DIR && ! -f $NNSMITH_DIR/.af-fuzz-managed ]]; then
  die "refusing to modify unmanaged directory: $NNSMITH_DIR; select a new NNSMITH_DIR"
fi
mkdir -p "$(dirname "$NNSMITH_DIR")"
WORK=$(mktemp -d "$(dirname "$NNSMITH_DIR")/.nnsmith-install.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
if [[ ! -d $NNSMITH_DIR/nnsmith ]] || [[ $(tree_hash) != "$EXPECTED" ]]; then
  git clone --no-checkout "$NNSMITH_REPO" "$WORK/source"
  git -C "$WORK/source" checkout --detach "$NNSMITH_COMMIT"
  for name in "${PATCH_ORDER[@]}"; do
    patch -p1 --forward --no-backup-if-mismatch -d "$WORK/source" -i "$SCRIPT_DIR/patches/$name.patch"
  done
  touch "$WORK/source/.af-fuzz-managed"
  if [[ -e $NNSMITH_DIR ]]; then mv "$NNSMITH_DIR" "$WORK/previous"; fi
  mv "$WORK/source" "$NNSMITH_DIR"
fi
ACTUAL=$(tree_hash)
[[ $ACTUAL == "$EXPECTED" ]] || die "NNSmith fingerprint mismatch: expected $EXPECTED, got $ACTUAL"
echo "Verified source: $ACTUAL"
[[ ${INSTALL_AF_FUZZ_SKIP_PIP:-0} != 1 ]] || exit 0
# Z3 4.12.2 imports pkg_resources; newer setuptools versions removed it.
"$PYTHON_BIN" -m pip install 'setuptools==75.8.0' 'setuptools-scm==8.2.0' 'wheel==0.45.1' \
  'z3-solver==4.12.2.0' 'numpy<2' 'hydra-core>=1.2,<2' 'hydra-colorlog>=1.2,<2' multipledispatch appdirs
cp -a "$NNSMITH_DIR" "$WORK/build-source"
# setuptools_scm normally injects _version.py into the source tree. That extra
# Python file invalidates the frozen fingerprint, including on editable installs.
"$PYTHON_BIN" - "$WORK/build-source/pyproject.toml" <<'PY'
import sys
from pathlib import Path
p=Path(sys.argv[1]); p.write_text('\n'.join(line for line in p.read_text().splitlines() if not line.lstrip().startswith('write_to'))+'\n')
PY
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 "$PYTHON_BIN" -m pip wheel --no-deps --no-build-isolation "$WORK/build-source" -w "$WORK/wheels"
"$PYTHON_BIN" - "$WORK/wheels" "$EXPECTED" <<'PY'
import hashlib, sys, zipfile
from pathlib import Path
wheels=list(Path(sys.argv[1]).glob('*.whl')); assert len(wheels)==1
with zipfile.ZipFile(wheels[0]) as z:
    digest=hashlib.sha256()
    for name in sorted(n for n in z.namelist() if n.startswith('nnsmith/') and n.endswith('.py')):
        digest.update(name[len('nnsmith/'):].encode()+b'\0'+z.read(name))
    assert digest.hexdigest()==sys.argv[2], ('wheel fingerprint mismatch', digest.hexdigest())
PY
"$PYTHON_BIN" -m pip install --no-deps --force-reinstall "$WORK"/wheels/*.whl
# Verify actual imports, so stale PYTHONPATH/editable installations fail here.
"$PYTHON_BIN" - "$EXPECTED" <<'PY'
import hashlib, sys, nnsmith, z3
from pathlib import Path
root=Path(nnsmith.__file__).resolve().parent; digest=hashlib.sha256()
for path in sorted(root.rglob('*.py')):
    digest.update(path.relative_to(root).as_posix().encode()+b'\0'+path.read_bytes())
assert digest.hexdigest()==sys.argv[1], f'Imported stale NNSmith from {root}: {digest.hexdigest()}'
assert z3.get_version_string()=='4.12.2'
print({'source_path': str(root), 'source_sha256': digest.hexdigest(), 'z3': z3.get_version_string()})
PY
