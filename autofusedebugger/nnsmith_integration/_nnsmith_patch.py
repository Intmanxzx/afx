"""Transparently applies the ascendc-specific NNSmith patch (`IndexSelect`
op, see `patches/ascendc_index_select.patch`) to an already-installed
NNSmith package, so af-fuzz users never have to run `git apply`/`patch`
themselves.

Why a patch instead of a long-term fork/vendoring: NNSmith is installed as
an editable pip package from a local checkout (`pip install -e`), and the
project's standing rule is "NNSmith is a dependency, not modified in place"
-- but `IndexSelect` genuinely cannot be added any other way (NNSmith's
operator registration is a class-definition-time decorator, not something
that can be monkeypatched in from outside after the fact; see the class's
own docstring in the patch for why). A patch file keeps the modification
small, reviewable, and reversible, and lets this module apply it
automatically on first use instead of requiring a manual step.

Call `ensure_ascendc_patch_applied()` before importing anything from
`nnsmith.abstract.op` that depends on `IndexSelect` existing. It is a no-op
if the patch is already applied (idempotent -- safe to call on every
import).
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

_PATCH_FILE = Path(__file__).parent / "patches" / "ascendc_index_select.patch"

logger = logging.getLogger(__name__)


class NNSmithPatchError(RuntimeError):
    """Raised when the ascendc NNSmith patch cannot be verified or applied."""


def _forget_cached_nnsmith_op_module() -> None:
    """Evict `nnsmith.abstract.op` (and `nnsmith.abstract`, `nnsmith`) from
    `sys.modules` so the next `import` re-reads the file from disk instead
    of returning the cached pre-patch module object.

    Why this is needed: `_is_patch_already_applied()`'s first call (before
    the patch exists on disk) does `from nnsmith.abstract.op import
    IndexSelect`. That statement first fully imports `nnsmith.abstract.op`
    (caching it in `sys.modules`) and only then fails to find the
    `IndexSelect` attribute on it -- the *module* itself is now cached as
    "successfully imported", distinct from the attribute lookup that
    failed. Calling the same `from ... import IndexSelect` again after
    `patch` has rewritten the file on disk still resolves against that
    cached module object, which never re-reads anything -- so without this
    eviction, the post-patch check in `ensure_ascendc_patch_applied()` would
    always observe the same pre-patch module and incorrectly conclude the
    patch didn't take effect, even though the file on disk is correct.
    """
    for name in list(sys.modules):
        if name == "nnsmith" or name.startswith("nnsmith."):
            del sys.modules[name]


def _is_patch_already_applied() -> bool:
    try:
        from nnsmith.abstract.op import IndexSelect  # noqa: F401
    except ImportError:
        return False
    return True


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


def ensure_ascendc_patch_applied() -> None:
    """Idempotently ensures the `IndexSelect` op patch is applied to the
    installed NNSmith package. Raises `NNSmithPatchError` with an actionable
    message if NNSmith isn't installed or the patch can't be applied
    (e.g. the installed NNSmith version has drifted too far from what the
    patch expects).
    """
    if _is_patch_already_applied():
        return

    if not _PATCH_FILE.exists():
        raise NNSmithPatchError(
            f"ascendc NNSmith patch file not found at {_PATCH_FILE} -- "
            "the af-fuzz package install may be corrupted; reinstall "
            "autofusedebugger."
        )

    target_dir = _find_nnsmith_package_parent()

    result = subprocess.run(
        [
            "patch",
            "-p1",
            "--forward",
            "--no-backup-if-mismatch",
            "-i",
            str(_PATCH_FILE),
        ],
        cwd=str(target_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise NNSmithPatchError(
            "Failed to apply the ascendc NNSmith patch "
            f"({_PATCH_FILE}) against the installed NNSmith at "
            f"{target_dir / 'nnsmith'} -- the installed NNSmith version has "
            "likely drifted from what this patch expects. Re-derive the "
            "patch against the current NNSmith source, or pin NNSmith to "
            "the version this patch was built against.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    _forget_cached_nnsmith_op_module()
    if not _is_patch_already_applied():
        raise NNSmithPatchError(
            f"`patch` reported success against {target_dir / 'nnsmith'}, "
            "but `IndexSelect` still cannot be imported from "
            "nnsmith.abstract.op afterwards -- the patch may have applied "
            "to the wrong file layout. Inspect "
            f"{target_dir / 'nnsmith' / 'abstract' / 'op.py'} manually."
        )

    logger.info(
        "Applied ascendc NNSmith patch (IndexSelect op) to %s",
        target_dir / "nnsmith",
    )
