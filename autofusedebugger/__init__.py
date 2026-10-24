"""Autofuse per-op accuracy comparison helpers."""

__all__ = ["__version__"]

__version__ = "0.1.0"

# Must be the very first thing this file does -- before even the patch
# check below, and definitely before anything imports torch_npu (directly
# or transitively through ascendc_backend.py/inductor_autofuse.py).
# ascendc's gather/embedding/index_select fused lowering
# (lowering/gather_lowering.py) is gated behind a module-level constant
# (`enable_gather_fuse = "gather" in debug_options` in ascendc/config.py)
# computed once from this env var *at import time* -- setting it any later
# (e.g. inside probe_ascendc_opset(), which is where this lived in an
# earlier, broken attempt) is a no-op if torch_npu.ascendc has already been
# imported by then, because that module-level constant was already
# computed and ascendc/__init__.py's `_finetune_lowering()` already decided
# whether to `from .lowering import gather_lowering` based on it.
#
# Confirmed empirically on A5 hardware: without this, `_LoweringGuard.has()`
# returns False for aten.index_select/gather/embedding even though
# gather_lowering.py's registration code for them is completely correct --
# the module is simply never imported, so that code never runs. With this
# set, a graph combining index_select with downstream elementwise ops
# fuses with zero lowering_fallbacks (confirmed via autofuse_diagnostics()).
#
# Unconditional (not gated on soc_tier): this is a comma-separated debug
# option set (`os.environ.get(...).split(",")`-style parsing in
# ascendc/config.py), so adding "gather" to it has zero effect on graphs
# that don't contain any gather-family op, and the gather-family ops in
# NNSmith's catalog (core.IndexSelect) are already since=Soc.A5-only, so
# this is a no-op on A3 hardware regardless. Never overwrite an existing
# value the caller may have set (e.g. "nocat,layoutcheck") -- append to it.
import os as _os

_existing_debug_opts = _os.environ.get("TORCHINDUCTOR_ASCENDC_DEBUG", "")
_debug_opts = {o.strip().lstrip("+") for o in _existing_debug_opts.split(",") if o.strip()}
_debug_opts.add("gather")
_os.environ["TORCHINDUCTOR_ASCENDC_DEBUG"] = ",".join(sorted(_debug_opts))
del _existing_debug_opts, _debug_opts

# Bound inductor's parallel-compile worker pool. torch._inductor.config sizes
# it as min(32, os.cpu_count()), and in a container os.cpu_count() reports
# the *host's* cores, not the cgroup quota. Each worker is a full python
# process that has imported torch/torch_npu (0.7-1.5GB RSS), so on a cold
# compile cache the default pool grows to ~33 workers / ~24GB and, together
# with the main process, OOM-kills a 32GB cgroup (confirmed on A5: cold-cache
# 20-case af-fuzz batch died with exit 137 at ~116s; with 4 workers the same
# batch completes with an ~11GB peak). A warm-compile-cache run of the same
# batch peaked at ~5.8GB with no large worker processes among the top RSS,
# which is why this only shows up on a cold cache. Must be set before
# torch._inductor.config is imported (read once at import time). Never
# overwrite a value the caller set explicitly. With --jobs N every shard
# process gets its own pool, so peak worker memory scales with N * this.
_os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")
del _os

# Must run here, at the top of the package's own __init__, and before
# anything else in this package -- any `import autofusedebugger.<submodule>`
# runs this file first (that's how Python packages work), so this is the
# earliest point guaranteed to run before ANY of this package's submodules
# get a chance to `from nnsmith... import ...` something that depends on
# IndexSelect existing. Putting this check later (e.g. only in
# ascendc_opset.py, where it lived originally) is NOT early enough:
# fuzz_loop.py imports nnsmith.graph_gen/materialize before it imports
# ascendc_opset, and those modules' own imports can resolve names (e.g.
# `from nnsmith.graph_gen import model_gen`) against the pre-patch
# nnsmith.abstract.op before the patch ever runs -- no amount of evicting
# sys.modules afterwards fixes an already-bound name in another module's
# namespace. See _nnsmith_patch.py's docstring for why a patch is used at
# all instead of a non-invasive extension.
try:
    from autofusedebugger.nnsmith_integration._nnsmith_patch import (
        ensure_ascendc_patch_applied,
    )

    ensure_ascendc_patch_applied()
except ImportError:
    # nnsmith itself isn't installed yet (e.g. first-time setup before the
    # caller has run its own nnsmith install step) -- nothing to patch yet;
    # whoever installs nnsmith next and then imports this package again will
    # hit the check above and patch it then.
    pass
