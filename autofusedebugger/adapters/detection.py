from __future__ import annotations

import ast
from pathlib import Path


_FRAMEWORK_MODULES = {
    "torch": "pytorch",
    "torch_npu": "pytorch",
    "torchair": "pytorch",
    "inductor_npu_ext": "pytorch",
    "tensorflow": "tensorflow",
    "jax": "jax",
}


def find_cann_ge_root(script: str | Path) -> Path | None:
    """Find the repository root that provides a cann-ge/Baize ``libs`` package."""
    path = Path(script).resolve()
    for candidate in (path.parent, *path.parents):
        if (candidate / "libs" / "__init__.py").is_file() and (candidate / "feature").is_dir():
            return candidate
    return None


def _import_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = (item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = (node.module,)
        else:
            continue
        roots.update(name.split(".", 1)[0] for name in names)
    return roots


def detect_framework(script: str | Path) -> str:
    """Detect the model framework from imports without executing the script."""
    path = Path(script).resolve()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ValueError(f"cannot inspect framework imports in {path}: {exc}") from exc

    roots = _import_roots(tree)
    detected = {framework for module, framework in _FRAMEWORK_MODULES.items() if module in roots}
    if len(detected) > 1:
        names = ", ".join(sorted(detected))
        raise ValueError(
            f"framework auto-detection found multiple frameworks in {path}: {names}; "
            "pass --framework explicitly"
        )
    if detected:
        return detected.pop()

    # Keep old scripts working when the model class is imported from another module.
    return "pytorch"
