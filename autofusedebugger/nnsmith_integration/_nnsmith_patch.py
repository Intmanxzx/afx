"""Transparently applies the ascendc-specific NNSmith patches (new ops such
as `IndexSelect`, `Gather`, `LayerNorm`, plus smaller API tweaks such as the
`rtol`/`atol` tolerance parameters -- see `_ASCENDC_PATCHES` below and the
individual files under `patches/`) to an already-installed NNSmith package,
so af-fuzz users never have to run `git apply`/`patch` themselves.

Why a patch instead of a long-term fork/vendoring: NNSmith is installed as
an editable pip package from a local checkout (`pip install -e`), and the
project's standing rule is "NNSmith is a dependency, not modified in place"
-- but these changes genuinely cannot be made any other way (e.g. NNSmith's
operator registration is a class-definition-time decorator, not something
that can be monkeypatched in from outside after the fact; see each class's
own docstring in its patch for why). A patch file keeps each modification
small, reviewable, and reversible, and lets this module apply them
automatically on first use instead of requiring a manual step.

`_ASCENDC_PATCHES` is a list of independent patch entries, each pairing a
patch file with a `probe` callable that returns True iff that patch is
already applied. Not every patch adds an importable symbol (e.g. a patch
that only adds a keyword parameter to an existing method has nothing new to
import), so the probe is pluggable: see the `_..._probe()` factory
functions below (new ones added over time as new patch shapes demand them)
for the probing strategies in use, and `_AscendcPatch.probe`'s docstring
for why callers never need to know which strategy a given entry uses. Entries
are applied in order; each is checked and applied on its own, so one
already-applied patch is simply skipped while another still gets applied,
and a new patch only needs a new entry in the list -- no change to the
apply/verify logic below.

Call `ensure_ascendc_patch_applied()` before importing anything from
`nnsmith.abstract.op` that depends on these ops existing. It is a no-op
if all patches are already applied (idempotent -- safe to call on every
import).

Known limitation for whoever edits an *existing* patch file's content
(not just adds a new one): every probe here answers "does the installed
NNSmith look like this patch landed," not "does it match this *exact
version* of the patch file." If you change what a patch file does (e.g.
add a new mode/parameter to a function that already had some other
mode/parameter the probe was already satisfied by), the probe keeps
returning True against an installation that only has the *old* version
applied -- `ensure_ascendc_patch_applied()` will skip it, and the
installed NNSmith silently keeps running the stale code. There is no
content-hash or version check, only a structural probe (symbol exists /
parameter exists / etc). After editing a patch file's content, manually
verify (or force-reapply) the installed NNSmith yourself; don't trust
this module to detect that the file's content changed underneath an
already-satisfied probe.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_PATCHES_DIR = Path(__file__).parent / "patches"

# A probe is a zero-argument callable returning True iff the patch it
# belongs to is already applied. Keeping this as "just a callable" (rather
# than, say, an enum of probe kinds plus if/elif dispatch) means
# `_apply_one_patch()` / `ensure_ascendc_patch_applied()` never need to know
# how a given patch is detected -- they just call `patch.probe()`. Adding
# another probe strategy later (if one is ever needed) only means writing a
# new `_..._probe(...)` factory below; no existing call site changes.
_Probe = Callable[[], bool]


def _import_symbol_probe(symbol: str) -> _Probe:
    """Probe factory: "is `symbol` importable from `nnsmith.abstract.op`?"

    Used for patches that add a brand-new class/function to that module
    (e.g. `IndexSelect`, `Gather`) -- the new symbol itself is proof the
    patch landed.
    """

    def probe() -> bool:
        return _probe_symbol_importable(symbol)

    return probe


def _function_has_param_probe(
    module_path: str, qualname: str, param_name: str
) -> _Probe:
    """Probe factory: "does `<module_path>.<qualname>`'s signature contain
    a parameter named `param_name`?"

    Used for patches that only add a parameter to an *existing*
    function/method (no new symbol to import-probe). `qualname` may be
    dotted to reach a method on a class, e.g. "BackendFactory.verify_testcase".
    """

    def probe() -> bool:
        return _probe_function_has_param(module_path, qualname, param_name)

    return probe


def _function_accepts_var_keyword_probe(module_path: str, qualname: str) -> _Probe:
    """Probe factory: "does `<module_path>.<qualname>`'s signature accept
    arbitrary keyword arguments (i.e. does it have a `**kwargs`-style
    parameter)?"

    Used for patches that add a `**kwargs` catch-all to an existing
    function/method -- as opposed to `_function_has_param_probe`, which
    checks for one specific *named* parameter. A `**kwargs` addition
    introduces no new named keyword at all (the whole point is that the
    function forwards whatever keywords the caller passes without knowing
    their names ahead of time), so checking for a parameter literally named
    "kwargs" would both (a) false-positive on an unrelated pre-existing
    parameter that merely happens to be named "kwargs" and isn't a real
    `**kwargs`, and (b) miss a `**kwargs`-equivalent parameter spelled with
    a different name (e.g. `**extra`). Checking the parameter's `kind` for
    `VAR_KEYWORD` is the only way to ask the actual semantic question.
    `qualname` may be dotted to reach a method on a class, same convention
    as `_function_has_param_probe`.
    """

    def probe() -> bool:
        return _probe_function_accepts_var_keyword(module_path, qualname)

    return probe


def _attribute_exists_probe(module_path: str, qualname: str, attr_name: str) -> _Probe:
    """Probe factory: "does `<module_path>.<qualname>` have an attribute
    named `attr_name`?"

    Used for patches that add a new member to an *existing* class/enum (no
    new top-level symbol to import-probe, and not a function-signature
    change either) -- e.g. a new enum member like `DType.bfloat16`.
    `qualname` may be dotted to reach a nested class, same convention as
    `_function_has_param_probe`.
    """

    def probe() -> bool:
        return _probe_attribute_exists(module_path, qualname, attr_name)

    return probe


def _function_source_contains_probe(
    module_path: str, qualname: str, needle: str
) -> _Probe:
    """Probe factory: "does `<module_path>.<qualname>`'s source text contain
    `needle`?"

    Used for patches that only rewrite an expression *inside* an existing
    function body -- no new symbol, no signature change (same parameter
    names/count as before), no new class/enum member -- so none of the
    other probe factories above have anything to key off of. `qualname` may
    be dotted to reach a method on a class, same convention as
    `_function_has_param_probe`. `needle` should be a short, distinctive
    substring unique to the patched line (e.g. a call the unpatched source
    never makes) -- not matched against the literal unpatched text, or this
    would report "already applied" on a pristine checkout.

    Not usable for `nnsmith.materialize.torch.forward`: that module defines
    dozens of distinct functions all literally named `forward_fn` (one per
    `@operator_impl(...)`-decorated definition), so `getattr(module,
    "forward_fn")` resolves to whichever one happens to be bound *last* at
    module scope, not the one named by `qualname`. Use
    `_module_source_contains_probe` for patches that touch that file.
    """

    def probe() -> bool:
        return _probe_function_source_contains(module_path, qualname, needle)

    return probe


def _module_source_contains_probe(module_path: str, needle: str) -> _Probe:
    """Probe factory: "does `<module_path>`'s source file, as a whole,
    contain `needle`?"

    Coarser than `_function_source_contains_probe`: it checks the entire
    module's source text rather than one function's, so it works even when
    the patched function can't be looked up by name (e.g.
    `nnsmith.materialize.torch.forward`, which redefines `forward_fn` at
    module scope once per `@operator_impl(...)` registration -- `getattr`
    only ever reaches the last one). `needle` should be unique enough
    within the whole file that it can't appear anywhere by coincidence.
    """

    def probe() -> bool:
        return _probe_module_source_contains(module_path, needle)

    return probe


@dataclass(frozen=True)
class _AscendcPatch:
    """One independently-applied ascendc NNSmith patch.

    `probe` is a zero-argument callable (built via one of the
    `_..._probe()` factories above) returning True iff this patch is
    already applied. It is the sole marker used to detect whether the
    patch needs applying -- callers never branch on *how* a patch is
    probed, only on the boolean result.
    """

    filename: str
    probe: _Probe

    @property
    def path(self) -> Path:
        return _PATCHES_DIR / self.filename


# Add a new entry here for each new ascendc NNSmith patch; nothing else in
# this module needs to change.
_ASCENDC_PATCHES: tuple[_AscendcPatch, ...] = (
    _AscendcPatch(
        filename="ascendc_index_select.patch",
        probe=_import_symbol_probe("IndexSelect"),
    ),
    _AscendcPatch(
        filename="ascendc_gather_layernorm.patch",
        probe=_import_symbol_probe("Gather"),
    ),
    _AscendcPatch(
        filename="ascendc_tolerance_params.patch",
        probe=_function_has_param_probe(
            "nnsmith.backends.factory", "BackendFactory.verify_testcase", "rtol"
        ),
    ),
    _AscendcPatch(
        filename="ascendc_bfloat16.patch",
        probe=_attribute_exists_probe("nnsmith.abstract.dtype", "DType", "bfloat16"),
    ),
    _AscendcPatch(
        filename="ascendc_rank_and_axis_weighting.patch",
        probe=_import_symbol_probe("weighted_axis_choice"),
    ),
    _AscendcPatch(
        filename="ascendc_value_distribution.patch",
        probe=_function_has_param_probe(
            "nnsmith.materialize.torch.symbolnet", "random_tensor", "mode"
        ),
    ),
    _AscendcPatch(
        filename="ascendc_make_oracle_kwargs.patch",
        # Probed via TorchModel.make_oracle (the concrete implementation),
        # not the abstract Model.make_oracle in nnsmith.materialize -- this
        # patch touches both, but TorchModel's is the one actually called
        # at runtime, matching this table's existing convention of probing
        # a patch's concrete class when it touches both a base and an
        # override (e.g. ascendc_gather_layernorm.patch below probes
        # `Gather`, not a shared base it also touched).
        probe=_function_accepts_var_keyword_probe(
            "nnsmith.materialize.torch", "TorchModel.make_oracle"
        ),
    ),
    _AscendcPatch(
        filename="ascendc_refine_weights_value_dist.patch",
        # Same base/override situation as ascendc_make_oracle_kwargs.patch
        # above: this patch touches both the abstract Model.refine_weights
        # (adds a **kwargs catch-all) and the concrete
        # TorchModel.refine_weights (adds a specific, named
        # `value_dist_mode` parameter it actually consumes) -- probe the
        # concrete class, and use _function_has_param_probe (not the
        # var-keyword probe) since `value_dist_mode` is a real named
        # parameter here, not a **kwargs catch-all.
        probe=_function_has_param_probe(
            "nnsmith.materialize.torch", "TorchModel.refine_weights", "value_dist_mode"
        ),
    ),
    _AscendcPatch(
        filename="ascendc_init_fp_dtype_diversity.patch",
        # No new symbol/parameter/attribute to key off of -- this patch only
        # rewrites an expression inside three existing __init__ bodies
        # (SymbolicGen x2, ConcolicGen x1), replacing the unpatched
        # `dtype=DType.float32 if init_fp else None` with
        # `dtype=self.random_dtype_gen() if init_fp else None`. Probing
        # SymbolicGen.__init__'s source for the literal call the unpatched
        # version never makes is the only way to tell the two apart.
        probe=_function_source_contains_probe(
            "nnsmith.graph_gen", "SymbolicGen.__init__", "self.random_dtype_gen()"
        ),
    ),
    _AscendcPatch(
        filename="ascendc_dim_size_distribution.patch",
        # Adds a new `dim_dist_mode` parameter to BaseGen.__init__ (default
        # "uniform", unchanged behavior) plus a new module-level helper
        # `_sample_dim_size()` that make_random_concrete_placeholder calls
        # instead of a bare `random.randint(l, r)` -- the new parameter is
        # the simplest thing to key off of (same convention as
        # ascendc_refine_weights_value_dist.patch's `value_dist_mode` probe).
        probe=_function_has_param_probe(
            "nnsmith.graph_gen", "BaseGen.__init__", "dim_dist_mode"
        ),
    ),
    _AscendcPatch(
        filename="ascendc_reduce_min_max_amin_amax.patch",
        # No new symbol/parameter/attribute either -- this patch only
        # rewrites ReduceMin/ReduceMax's forward_fn bodies in
        # nnsmith/materialize/torch/forward.py, replacing
        # `x.min(dim).values`/`x.max(dim).values` (traces to the
        # tuple-returning aten.min.dim/aten.max.dim, which ascendc's
        # _LoweringGuard always falls back on -- the int64 `indices` half of
        # that tuple never matches the registered float/int output dtype
        # set) with `x.amin(dim)`/`x.amax(dim)` (traces to aten.amin/
        # aten.amax, single-Tensor output, registered separately and
        # actually fusable). `forward.py` redefines `forward_fn` at module
        # scope once per `@operator_impl(...)` registration, so it can't be
        # probed by qualname (see _function_source_contains_probe's
        # docstring) -- use the whole-module probe instead, keyed on a
        # substring the unpatched file never contains.
        probe=_module_source_contains_probe(
            "nnsmith.materialize.torch.forward", "x.amin(op.extra_attrs"
        ),
    ),
)


class NNSmithPatchError(RuntimeError):
    """Raised when an ascendc NNSmith patch cannot be verified or applied."""


def _forget_cached_nnsmith_op_module() -> None:
    """Evict `nnsmith.abstract.op` (and `nnsmith.abstract`, `nnsmith`) from
    `sys.modules` so the next `import` re-reads the file from disk instead
    of returning the cached pre-patch module object.

    Why this is needed: `_probe_symbol_importable()`'s first call (before
    a patch exists on disk) does `from nnsmith.abstract.op import
    <probe_symbol>`. That statement first fully imports `nnsmith.abstract.op`
    (caching it in `sys.modules`) and only then fails to find the probe
    attribute on it -- the *module* itself is now cached as "successfully
    imported", distinct from the attribute lookup that failed. Calling the
    same `from ... import <probe_symbol>` again after `patch` has rewritten
    the file on disk still resolves against that cached module object,
    which never re-reads anything -- so without this eviction, the
    post-patch check in `_apply_one_patch()` would always observe the same
    pre-patch module and incorrectly conclude the patch didn't take effect,
    even though the file on disk is correct.

    Must be called after *every* successful `patch` invocation (not just
    once at the end), so that the next patch entry's probe check in the
    loop doesn't get fooled by a module cached from before that entry's
    patch was applied.
    """
    for name in list(sys.modules):
        if name == "nnsmith" or name.startswith("nnsmith."):
            del sys.modules[name]


def _probe_symbol_importable(symbol: str) -> bool:
    try:
        module = __import__("nnsmith.abstract.op", fromlist=[symbol])
    except ImportError:
        return False
    return hasattr(module, symbol)


def _probe_function_has_param(
    module_path: str, qualname: str, param_name: str
) -> bool:
    """True iff the function/method reached by walking `qualname`'s
    dot-separated parts off of `module_path` (e.g. module_path=
    "nnsmith.backends.factory", qualname="BackendFactory.verify_testcase")
    has a parameter named `param_name` in its signature.

    If `module_path` cannot be imported at all (NNSmith not installed, or
    some unrelated environment failure), this returns False rather than
    raising -- same error-handling philosophy as `_probe_symbol_importable`
    above: an import failure here does not mean "this specific patch is
    missing", it means "we cannot currently tell", and the caller's
    fallback for "cannot tell" is to treat the patch as not-yet-applied.
    That is the conservative choice: `ensure_ascendc_patch_applied()`'s
    actual NNSmith-not-installed case is already handled explicitly and
    loudly by `_find_nnsmith_package_parent()` raising `NNSmithPatchError`
    right after this probe reports "pending" -- so swallowing the
    ImportError here doesn't hide the problem, it just defers surfacing it
    to the one place that produces an actionable message instead of a bare
    traceback from deep inside a probe helper. Letting it propagate instead
    would also break the `all(probe() for ...)` style short-circuit the
    rest of this module relies on: every probe here is expected to be a
    pure, exception-safe boolean check.
    """
    try:
        obj = importlib.import_module(module_path)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except ImportError:
        return False
    sig = inspect.signature(obj)
    return param_name in sig.parameters


def _probe_function_accepts_var_keyword(module_path: str, qualname: str) -> bool:
    """True iff the function/method reached by walking `qualname`'s
    dot-separated parts off of `module_path` has a `**kwargs`-style
    (`inspect.Parameter.VAR_KEYWORD`) parameter in its signature, under
    whatever name it was given.

    Same import-failure handling as `_probe_function_has_param` above: an
    `ImportError` means "we cannot currently tell" and is treated as
    "not yet applied" rather than propagated.
    """
    try:
        obj = importlib.import_module(module_path)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except ImportError:
        return False
    sig = inspect.signature(obj)
    return any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )


def _probe_attribute_exists(module_path: str, qualname: str, attr_name: str) -> bool:
    """True iff the object reached by walking `qualname`'s dot-separated
    parts off of `module_path` (e.g. module_path="nnsmith.abstract.dtype",
    qualname="DType") has an attribute named `attr_name` (e.g. "bfloat16").

    Unlike `_probe_function_has_param`, the object reached by walking
    `qualname` here is the *container* itself (a class/enum), not a
    function/method to introspect further -- so the last step is a
    `hasattr()` check against `attr_name` rather than one more `getattr()`
    hop followed by `inspect.signature()`.

    If `module_path` cannot be imported at all (NNSmith not installed, or
    some unrelated environment failure), this returns False rather than
    raising -- same error-handling philosophy as `_probe_function_has_param`
    and `_probe_symbol_importable` above: an import failure here does not
    mean "this specific patch is missing", it means "we cannot currently
    tell", and the caller's fallback for "cannot tell" is to treat the
    patch as not-yet-applied. By contrast, if `module_path` *does* import
    but walking `qualname` off of it fails (e.g. a typo in this module's
    own `_ASCENDC_PATCHES` table), the resulting `AttributeError` is left
    to propagate -- that failure means the patch *table* is wrong, not that
    the patch is pending, and silently returning False there would hide a
    bug in this file instead of a drift in the installed NNSmith checkout.
    """
    try:
        obj = importlib.import_module(module_path)
    except ImportError:
        return False
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return hasattr(obj, attr_name)


def _probe_function_source_contains(
    module_path: str, qualname: str, needle: str
) -> bool:
    """True iff `needle` appears in the source text of the function/method
    reached by walking `qualname`'s dot-separated parts off of `module_path`.

    Same import-failure handling as `_probe_function_has_param` above: an
    `ImportError` means "we cannot currently tell" and is treated as
    "not yet applied" rather than propagated. `inspect.getsource()` itself
    can also raise `OSError` (e.g. the source file is missing, or the
    object is a builtin with no retrievable source) -- that case gets the
    same "cannot currently tell" treatment, for the same reason: it isn't
    evidence the patch is pending, it's evidence this probe strategy cannot
    currently answer the question, which `ensure_ascendc_patch_applied()`'s
    caller handles by attempting to (re)apply the patch rather than by
    crashing.
    """
    try:
        obj = importlib.import_module(module_path)
        for part in qualname.split("."):
            obj = getattr(obj, part)
        source = inspect.getsource(obj)
    except (ImportError, OSError):
        return False
    return needle in source


def _probe_module_source_contains(module_path: str, needle: str) -> bool:
    """True iff `needle` appears anywhere in `module_path`'s source file.

    Same import-failure handling as `_probe_function_has_param` above: an
    `ImportError` or `OSError` (source file missing/unreadable) means "we
    cannot currently tell" and is treated as "not yet applied" rather than
    propagated.
    """
    try:
        module = importlib.import_module(module_path)
        source = inspect.getsource(module)
    except (ImportError, OSError):
        return False
    return needle in source


def _find_nnsmith_package_parent() -> Path:
    spec = importlib.util.find_spec("nnsmith")
    if spec is None or not spec.submodule_search_locations:
        raise NNSmithPatchError(
            "Cannot locate an installed `nnsmith` package to patch -- "
            "install NNSmith first (see the af-fuzz setup docs), then retry."
        )
    # submodule_search_locations[0] is the `nnsmith/` package directory
    # itself; patch -p1 expects to be run from its *parent* (patch paths are
    # "a/nnsmith/...").
    nnsmith_dir = Path(spec.submodule_search_locations[0])
    return nnsmith_dir.parent


def _apply_one_patch(patch: _AscendcPatch, target_dir: Path) -> None:
    """Applies a single ascendc patch entry, assuming it was already
    determined (by the caller) not to be applied yet. Raises
    `NNSmithPatchError`, naming this specific patch file, if the patch file
    is missing, `patch` fails, or `patch.probe()` still returns False
    afterwards.
    """
    if not patch.path.exists():
        raise NNSmithPatchError(
            f"ascendc NNSmith patch file not found at {patch.path} -- "
            "the af-fuzz package install may be corrupted; reinstall "
            "autofusedebugger."
        )

    result = subprocess.run(
        [
            "patch",
            "-p1",
            "--forward",
            "--no-backup-if-mismatch",
            "-i",
            str(patch.path),
        ],
        cwd=str(target_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise NNSmithPatchError(
            f"Failed to apply the ascendc NNSmith patch ({patch.path}) "
            f"against the installed NNSmith at {target_dir / 'nnsmith'} -- "
            "the installed NNSmith version has likely drifted from what "
            "this patch expects. Re-derive the patch against the current "
            "NNSmith source, or pin NNSmith to the version this patch was "
            "built against.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    _forget_cached_nnsmith_op_module()
    if not patch.probe():
        raise NNSmithPatchError(
            f"`patch` reported success against {target_dir / 'nnsmith'} "
            f"for {patch.path}, but its probe still reports the patch as "
            "not applied afterwards -- the patch may have applied to the "
            f"wrong file layout. Inspect the installed NNSmith checkout at "
            f"{target_dir / 'nnsmith'} manually."
        )

    logger.info(
        "Applied ascendc NNSmith patch (%s) to %s",
        patch.filename,
        target_dir / "nnsmith",
    )


def ensure_ascendc_patch_applied() -> None:
    """Idempotently ensures every ascendc NNSmith patch in `_ASCENDC_PATCHES`
    is applied to the installed NNSmith package. Each patch is checked and
    applied independently: a patch whose `probe()` already returns True is
    skipped, while the others are still applied. Raises `NNSmithPatchError`
    with an actionable message, naming the specific patch file involved, if
    NNSmith isn't installed or a patch can't be applied (e.g. the installed
    NNSmith version has drifted too far from what that patch expects).
    """
    pending = [patch for patch in _ASCENDC_PATCHES if not patch.probe()]
    if not pending:
        return

    target_dir = _find_nnsmith_package_parent()

    for patch in pending:
        if patch.probe():
            # Applying an earlier patch in this loop may have incidentally
            # made this one's probe pass too (unlikely given each patch
            # targets a distinct symbol/parameter, but checked for safety
            # since _forget_cached_nnsmith_op_module() changes module state
            # between iterations).
            continue
        _apply_one_patch(patch, target_dir)
