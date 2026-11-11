#!/bin/bash
# `autofusedbg fuzz` 的 NNSmith 环境一键安装脚本（自校验，幂等，可重复运行）。
#
# 在目标 Python 环境中运行（如 conda dev 环境）：
#   cd Autofuse-Debugger   # 本仓库 checkout
#   bash autofusedebugger/nnsmith_integration/install_af_fuzz.sh
#
# 步骤：
#   1. pip install -e .            安装 autofusedebugger 本体
#   2. clone nnsmith 到 ../nnsmith 并检出钉死的 commit
#   3. 按固定顺序应用 patches/ 下全部补丁
#   4. 安装 z3-solver==4.12.2.0 与 multipledispatch，pip install -e nnsmith
#   5. 校验源码树 SHA256 == 仓库 SOURCE_SHA256，不一致立即报错
#
# 指纹与补丁顺序强绑定：多个补丁修改同一文件的相邻区域，应用顺序不同
# 会导致最终文件内容不同、指纹不同。不要调整 PATCH_ORDER，除非同时
# 重新生成指纹并同步修改 nnsmith_v3.py / nnsmith_sampled.py。
#
# 环境变量：
#   NNSMITH_REPO               覆盖克隆源（默认 GitHub；网络受限时可换镜像）
#   INSTALL_AF_FUZZ_SKIP_PIP=1 跳过所有 pip 步骤（仅测试补丁管线时使用）

set -euo pipefail

NNSMITH_REPO_DEFAULT="https://github.com/ise-uiuc/nnsmith.git"
NNSMITH_REPO="${NNSMITH_REPO:-$NNSMITH_REPO_DEFAULT}"
NNSMITH_COMMIT="bc0af42c7d5fc4fd201efb76e5313f6298c2d573"  # pinned: ascendc patches are written against this baseline, do not bump
Z3_PIN="z3-solver==4.12.2.0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NNSMITH_DIR="${REPO_ROOT}/../nnsmith"
PATCHES_DIR="${SCRIPT_DIR}/patches"

# 固定补丁顺序：与仓库 SOURCE_SHA256 指纹对应，勿调整。
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

command -v git >/dev/null || die "需要 git"
command -v patch >/dev/null || die "需要 GNU patch（apt install patch）"

# 期望指纹：唯一事实来源是 nnsmith_v3.py 里的 SOURCE_SHA256 常量。
expected_hash() {
  python3 - "${REPO_ROOT}/autofusedebugger/fuzz/generation/nnsmith_v3.py" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"SOURCE_SHA256\s*=\s*'([0-9a-f]{64})'", text)
print(m.group(1) if m else "")
PY
}

# 与 fuzz/generation/nnsmith_v3.py source_evidence() 完全相同的算法：
# 对 nnsmith 包目录下全部 .py（按相对路径排序）逐个 path\0+bytes 拼接计算。
tree_hash() {
  python3 - "${NNSMITH_DIR}/nnsmith" <<'PY'
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(root.rglob('*.py')):
    digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
print(digest.hexdigest())
PY
}

EXPECTED="$(expected_hash)"
[ -n "$EXPECTED" ] || die "无法从 nnsmith_v3.py 读取 SOURCE_SHA256"

if [ -d "${NNSMITH_DIR}/nnsmith" ] && [ "$(tree_hash)" = "$EXPECTED" ]; then
  echo "== nnsmith 环境已就绪且指纹匹配，跳过 clone/打补丁 =="
else
  if [ ! -d "${NNSMITH_DIR}/.git" ]; then
    echo "== [1/4] 克隆 nnsmith 到 ${NNSMITH_DIR} =="
    git clone "$NNSMITH_REPO" "${NNSMITH_DIR}"
  else
    echo "== [1/4] 复用已有克隆 ${NNSMITH_DIR} =="
  fi
  git -C "${NNSMITH_DIR}" fetch origin "${NNSMITH_COMMIT}" --depth 1 2>/dev/null || true
  git -C "${NNSMITH_DIR}" checkout -f "${NNSMITH_COMMIT}" 2>/dev/null \
    || die "无法检出 commit ${NNSMITH_COMMIT}（克隆不完整时可删除 ${NNSMITH_DIR} 后重试）"
  git -C "${NNSMITH_DIR}" clean -fd

  echo "== [2/4] 按固定顺序应用 ${#PATCH_ORDER[@]} 个补丁 =="
  for name in "${PATCH_ORDER[@]}"; do
    p="${PATCHES_DIR}/${name}.patch"
    [ -f "$p" ] || die "补丁文件缺失: $p"
    if patch -p1 --forward --no-backup-if-mismatch -d "${NNSMITH_DIR}" -i "$p" >/dev/null; then
      echo "  OK  ${name}"
    else
      die "补丁应用失败: ${name}（nnsmith 基线可能已变化，勿手改指纹强行通过，请反馈维护者）"
    fi
  done

  echo "== [3/4] 校验源码树指纹 =="
  ACTUAL="$(tree_hash)"
  if [ "$ACTUAL" != "$EXPECTED" ]; then
    die "指纹不匹配！
  期望 (SOURCE_SHA256): ${EXPECTED}
  实际 (当前源码树):   ${ACTUAL}
补丁顺序与本机 patch 工具版本都会影响结果。请确认未改动 PATCH_ORDER；
若仍不一致，说明本机 patch 工具行为与指纹生成环境不同，请反馈维护者，
不要通过手改 SOURCE_SHA256 绕过校验。"
  fi
  echo "  指纹匹配: ${ACTUAL:0:16}..."
fi

if [ "${INSTALL_AF_FUZZ_SKIP_PIP:-0}" = "1" ]; then
  echo "== 跳过 pip 步骤 (INSTALL_AF_FUZZ_SKIP_PIP=1) =="
else
  echo "== [4/4] 安装 Python 依赖 =="
  python3 -m pip install -e "${REPO_ROOT}"
  python3 -m pip install "${Z3_PIN}" multipledispatch
  python3 -m pip install -e "${NNSMITH_DIR}"
fi

echo
echo "完成。NPU Pod 上每次使用前先加载运行环境："
echo "  source ${SCRIPT_DIR}/npu_env.sh"
echo "  autofusedbg fuzz replay C000000016        # 冒烟验证"
