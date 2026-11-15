"""Parse shell launcher scripts to inherit their environment and model entry.

Many models are started by a shell script that exports a batch of environment
variables and then invokes the python entry. The convenience form
``afx bash launch.sh`` uses this module to:

1. extract ``export KEY=VALUE`` lines and inject them into ``os.environ``;
2. find the ``python xxx.py ...`` invocation and return it as the model entry.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autofusex.progress import progress_message


_SHELL_INTERPRETERS = {"bash", "sh", "zsh", "dash", "ksh"}
_PYTHON_EXECUTABLE_RE = re.compile(r"^python(?:\d+(?:\.\d+)*)?$")
_EXPORT_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_COMMENT_RE = re.compile(r"#.*")
_FOR_LOOP_RE = re.compile(r"^\s*for\s+")
_VARIABLE_REF_RE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?")

# Tool-owned variables: values forced by the debugger itself at execution time.
_TOOL_OWNED_VARS = {
    "TORCHINDUCTOR_NPU_BACKEND",
    "TORCH_COMPILE_DEBUG",
    "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
}


@dataclass
class ShellScriptPlan:
    script: Path
    env: dict[str, str] = field(default_factory=dict)
    python_tokens: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def python_script(self) -> Path | None:
        for index, token in enumerate(self.python_tokens):
            candidate = Path(token)
            if candidate.suffix == ".py":
                if not candidate.is_absolute():
                    resolved = (self.script.parent / token).resolve()
                    self.python_tokens[index] = str(resolved)
                    if not resolved.is_file():
                        pass  # keep resolved path; caller reports a clear file error
                    return resolved
                return candidate
        return None


def is_shell_script_invocation(argv: list[str]) -> bool:
    if not argv:
        return False
    head = Path(argv[0]).name
    if head in _SHELL_INTERPRETERS:
        return True
    if argv[0].endswith(".sh"):
        return True
    return False


def _strip_inline_comment(line: str) -> str:
    """Drop a trailing # comment, but only outside single/double quotes."""
    if line.lstrip().startswith("#"):
        return ""
    quote: str | None = None
    for index, char in enumerate(line):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#":
            return line[:index].strip()
    return line


def _expand_value(value: str, env: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(0).lstrip("$").strip("{}")
        return env.get(name, os.environ.get(name, match.group(0)))

    return _VARIABLE_REF_RE.sub(replace, value)


def _parse_export_line(line: str) -> tuple[str, str] | None:
    match = _EXPORT_RE.match(line)
    if match is None:
        return None
    key, raw_value = match.groups()
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
        value = value[1:-1]
    return key, value


def _parse_shell_tokens(line: str) -> list[str] | None:
    try:
        return shlex.split(line)
    except ValueError:
        return None


def _expand_array_value(value: str) -> str:
    """Normalize a shell array literal like ``("a" "b")`` to its first element."""
    stripped = value.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        inner = stripped[1:-1]
        try:
            items = shlex.split(inner)
        except ValueError:
            return value
        if items:
            return items[0]
    return value


_VAR_PREFIX_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)(\[@\])?\}?")


def _substitute_loop_and_array_vars(
    token: str, for_loop_vars: dict[str, list[str]], env: dict[str, str]
) -> str:
    """Replace $VAR / ${VAR} / ${VAR[@]} references inside a token."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in for_loop_vars and for_loop_vars[name]:
            return for_loop_vars[name][0]
        if name in env:
            return _expand_array_value(env[name])
        return match.group(0)

    return _VAR_PREFIX_RE.sub(replace, token)


def parse_shell_script(path: str | Path) -> ShellScriptPlan:
    script = Path(path).expanduser().resolve()
    if not script.is_file():
        raise FileNotFoundError(f"shell script not found: {script}")
    plan = ShellScriptPlan(script=script)

    env: dict[str, str] = {}
    for_loop_vars: dict[str, list[str]] = {}
    raw_lines = script.read_text(encoding="utf-8", errors="replace").splitlines()
    # Join backslash continuation lines before parsing.
    joined_lines: list[str] = []
    pending = ""
    for raw_line in raw_lines:
        if raw_line.rstrip().endswith("\\"):
            pending += raw_line.rstrip()[:-1] + " "
            continue
        joined_lines.append(pending + raw_line)
        pending = ""
    if pending:
        joined_lines.append(pending)

    for raw_line in joined_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _FOR_LOOP_RE.match(line):
            for_match = re.match(r"^\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.*?);\s*do\s*$", line)
            if for_match is not None:
                var, values = for_match.groups()
                tokens = _parse_shell_tokens(values)
                if tokens is not None:
                    for_loop_vars[var] = [
                        _expand_array_value(_substitute_loop_and_array_vars(token, {}, env))
                        for token in tokens
                    ]
            continue
        if line.startswith(("if ", "while ", "case ", "function ", "}", "esac", "fi", "done", "elif", "else")):
            continue

        line = _strip_inline_comment(line)
        if not line:
            continue

        export = _parse_export_line(line)
        if export is not None:
            key, value = export
            value = _expand_value(value, env)
            if key in _TOOL_OWNED_VARS:
                plan.warnings.append(f"{key} is managed by afx; script value ignored: {value}")
                continue
            env[key] = value
            continue

        tokens = _parse_shell_tokens(line)
        if tokens is None:
            continue
        if tokens and tokens[0] in {"echo", "rm", "mkdir", "cd", "cp", "mv", "ls", "set", "unset", "source", "."}:
            if tokens[0] in {"source", "."} and len(tokens) > 1:
                plan.warnings.append(
                    f"line 'source {tokens[1]}' is not expanded; export variables from it manually if needed"
                )
            continue

        # Expand loop variables (e.g. MODELS=("sasrec") -> $model), then env vars.
        tokens = [
            _substitute_loop_and_array_vars(
                _expand_value(
                    _substitute_loop_and_array_vars(token, for_loop_vars, env),
                    env,
                ),
                for_loop_vars,
                env,
            )
            for token in tokens
        ]

        python_index = next((i for i, t in enumerate(tokens) if _PYTHON_EXECUTABLE_RE.match(t)), None)
        if python_index is not None and not plan.python_tokens:
            plan.python_tokens = list(tokens[python_index + 1 :])
            for index, token in enumerate(plan.python_tokens):
                if Path(token).suffix == ".py":
                    if not Path(token).is_absolute():
                        candidate = (script.parent / token).resolve()
                        if candidate.is_file():
                            plan.python_tokens[index] = str(candidate)
                        else:
                            plan.warnings.append(
                                f"model script not found next to launcher: {candidate}; using path as-is"
                            )
                    break
            continue
        if tokens and Path(tokens[0]).suffix == ".sh" and not plan.python_tokens:
            plan.warnings.append(f"nested shell script not followed: {tokens[0]}")

    plan.env = env
    if not plan.python_tokens:
        raise ValueError(
            f"no python model invocation found in {script}; "
            "pass the python script explicitly: afx python3 model.py ..."
        )
    return plan


def apply_plan_env(plan: ShellScriptPlan) -> None:
    """Inject parsed environment variables into the current process."""
    for key, value in plan.env.items():
        os.environ[key] = value
    if plan.env:
        progress_message("INFO", f"inherited {len(plan.env)} env vars from {plan.script.name}: {sorted(plan.env)}")
    for warning in plan.warnings:
        progress_message("WARNING", warning)


def load_env_file(path: str | Path) -> dict[str, str]:
    """Extract export KEY=VALUE pairs from a shell file without executing it."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"env file not found: {file_path}")
    env: dict[str, str] = {}
    for raw_line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = _strip_inline_comment(line)
        if not line:
            continue
        export = _parse_export_line(line)
        if export is None:
            continue
        key, value = export
        env[key] = _expand_value(value, env)
    if not env:
        raise ValueError(f"no export KEY=VALUE lines found in {file_path}")
    return env


def apply_env_overrides(env_items: list[str] | None, env_file: str | Path | None) -> None:
    """Apply --env KEY=VALUE items and --env-file exports to the current process."""
    if env_file is not None:
        file_env = load_env_file(env_file)
        for key, value in file_env.items():
            os.environ[key] = value
        progress_message("INFO", f"applied {len(file_env)} env vars from {env_file}: {sorted(file_env)}")
    for item in env_items or []:
        if "=" not in item:
            raise ValueError(f"env item must be KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"env key must be non-empty: {item}")
        os.environ[key] = value
        progress_message("INFO", f"applied env {key}={value}")
