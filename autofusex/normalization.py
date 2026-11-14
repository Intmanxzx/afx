from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autofusex.agent import AgentClient, AgentConfig, AgentError


NORMALIZATION_SCHEMA = "autofusex-normalization/1"
RECOVERY_SCHEMA = "autofusex-recovery/1"
_NORMALIZATION_REQUIRED = (
    "schema",
    "status",
    "framework",
    "source",
    "model",
    "graph",
    "inputs",
    "weights",
    "execution",
    "diagnostics",
)
_RECOVERY_REQUIRED = ("schema", "status", "phase", "diagnosis", "actions", "retry")
_RECOVERY_ACTION_KINDS = {"cli_override", "environment", "dependency", "script_patch", "manual"}


def normalize_script(
    *,
    script: str | Path,
    framework: str,
    script_args: tuple[str, ...] = (),
    config: AgentConfig,
    output: str | Path,
    max_script_bytes: int = 512 * 1024,
) -> dict[str, Any]:
    if max_script_bytes <= 0:
        raise ValueError("max_script_bytes must be positive")
    path = Path(script).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))
    source = path.read_bytes()
    if len(source) > max_script_bytes:
        source = source[:max_script_bytes]
        truncation = f"\n\n# [source truncated at {max_script_bytes} bytes]"
    else:
        truncation = ""
    resolved_framework = _resolve_framework(path, framework)
    client = AgentClient(config)
    raw = client.complete_json(
        system_prompt=_normalization_system_prompt(),
        user_prompt=_normalization_user_prompt(
            path,
            source.decode("utf-8", errors="replace") + truncation,
            resolved_framework,
            script_args,
        ),
    )
    normalized = validate_normalization(raw)
    if normalized["framework"] != resolved_framework:
        raise AgentError(
            f"Agent framework mismatch: requested/detected {resolved_framework!r}, "
            f"returned {normalized['framework']!r}"
        )
    normalized["source"] = dict(normalized["source"])
    normalized["source"]["script"] = str(path)
    normalized["execution"] = dict(normalized["execution"])
    normalized["execution"]["requested_framework"] = framework
    normalized["execution"]["script_args"] = list(script_args)
    normalized["agent"] = {
        "transport": config.transport,
        "endpoint": config.endpoint,
        "model": config.model,
        "timeout": config.timeout,
        "command": list(config.command) if config.command else None,
    }
    write_json(output, normalized)
    return normalized


def recover_script_error(
    *,
    script: str | Path,
    framework: str,
    phase: str,
    error: BaseException,
    normalized: dict[str, Any] | None,
    config: AgentConfig,
    output: str | Path,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    path = Path(script).resolve()
    client = AgentClient(config)
    raw = client.complete_json(
        system_prompt=_recovery_system_prompt(),
        user_prompt=_recovery_user_prompt(path, framework, phase, error, normalized, traceback_text),
    )
    try:
        recovery = validate_recovery(raw)
    except AgentError as exc:
        # CANNBot can return useful prose or a partial object despite the JSON
        # instruction. Preserve it without hiding the original execution error.
        recovery = {
            "schema": RECOVERY_SCHEMA,
            "status": "not_actionable",
            "phase": phase,
            "diagnosis": {
                "category": "agent_protocol",
                "summary": "Agent response did not match the recovery schema",
                "validation_error": str(exc),
                "agent_response": raw,
            },
            "actions": [
                {
                    "kind": "manual",
                    "priority": "high",
                    "description": "Review the original execution error and the saved Agent response",
                }
            ],
            "retry": {"recommended": False, "reason": "Agent recovery response is incomplete"},
        }
    recovery["source"] = {"script": str(path), "framework": framework}
    recovery["agent"] = {
        "transport": config.transport,
        "endpoint": config.endpoint,
        "model": config.model,
        "timeout": config.timeout,
        "command": list(config.command) if config.command else None,
    }
    write_json(output, recovery)
    return recovery


def validate_normalization(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AgentError("normalization response must be a JSON object")
    missing = [key for key in _NORMALIZATION_REQUIRED if key not in value]
    if missing:
        raise AgentError(f"normalization response missing fields: {', '.join(missing)}")
    if value["schema"] != NORMALIZATION_SCHEMA:
        raise AgentError(f"unsupported normalization schema: {value['schema']!r}")
    if value["status"] not in {"ready", "needs_user_input"}:
        raise AgentError(f"invalid normalization status: {value['status']!r}")
    if not isinstance(value["framework"], str) or not value["framework"]:
        raise AgentError("normalization framework must be a non-empty string")
    for field in ("source", "model", "graph", "inputs", "weights", "execution", "diagnostics"):
        if not isinstance(value[field], dict):
            raise AgentError(f"normalization field {field!r} must be an object")
    entrypoint = value["source"].get("entrypoint")
    if not isinstance(entrypoint, dict):
        raise AgentError("normalization source.entrypoint must be an object")
    entrypoint_kind = entrypoint.get("kind", "unknown")
    if not isinstance(entrypoint_kind, str) or not entrypoint_kind:
        raise AgentError("normalization source.entrypoint.kind must be a non-empty string")
    capture_call = value["execution"].get("capture_call")
    if capture_call is not None and (not isinstance(capture_call, int) or capture_call < 0):
        raise AgentError("normalization execution.capture_call must be a non-negative integer")
    strategy = value["inputs"].get("strategy", "unknown")
    if not isinstance(strategy, str) or not strategy:
        raise AgentError("normalization inputs.strategy must be a non-empty string")
    confidence = value["diagnostics"].get("confidence")
    if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
        raise AgentError("normalization diagnostics.confidence must be between 0 and 1")
    return value


def validate_recovery(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AgentError("recovery response must be a JSON object")
    missing = [key for key in _RECOVERY_REQUIRED if key not in value]
    if missing:
        raise AgentError(f"recovery response missing fields: {', '.join(missing)}")
    if value["schema"] != RECOVERY_SCHEMA:
        raise AgentError(f"unsupported recovery schema: {value['schema']!r}")
    if value["status"] not in {"retryable", "requires_user_action", "not_actionable"}:
        raise AgentError(f"invalid recovery status: {value['status']!r}")
    if not isinstance(value["phase"], str) or not isinstance(value["diagnosis"], dict):
        raise AgentError("recovery phase and diagnosis must be valid values")
    if not isinstance(value["actions"], list):
        raise AgentError("recovery actions must be a list")
    for action in value["actions"]:
        if not isinstance(action, dict) or action.get("kind") not in _RECOVERY_ACTION_KINDS:
            raise AgentError("recovery actions contain an unsupported kind")
    if not isinstance(value["retry"], dict):
        raise AgentError("recovery retry must be an object")
    return value


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_framework(script: Path, framework: str) -> str:
    if framework != "auto":
        return framework
    from autofusex.adapters.detection import detect_framework

    return detect_framework(script)


def _normalization_system_prompt() -> str:
    return """你是 AutofuseX 的模型脚本归一化 Agent。你只能静态分析用户提供的 Python 源码，不能执行源码、导入用户模块、访问网络、读取权重文件或猜测未出现的信息。请严格只返回一个 JSON 对象，不要 Markdown，不要解释文字。

你的任务是识别 PyTorch 模型脚本的入口、模型构造方式、输入生成方式、权重来源和执行约束，供本地 runner 选择已有的安全捕获流程。inputs 和 weights 只描述来源、路径、字段、shape/dtype 等元信息，不要把大型 tensor 或权重编码进 JSON。不要返回 Python 代码、shell 命令或需要自动执行的补丁。

JSON 必须符合 autofusex-normalization/1 schema，字段模板见用户请求中的示例。"""


def _normalization_user_prompt(
    path: Path, source: str, framework: str, script_args: tuple[str, ...]
) -> str:
    template = {
        "schema": NORMALIZATION_SCHEMA,
        "status": "ready",
        "framework": framework,
        "source": {
            "script": str(path),
            "project_root": None,
            "imports": [],
            "entrypoint": {
                "kind": "module_call",
                "symbol": None,
                "module_pattern": None,
                "capture_call": 0,
            },
        },
        "model": {
            "kind": "nn.Module",
            "class_name": None,
            "construction": {"strategy": "script_runtime", "symbol": None, "location": None, "arguments": {}},
            "training": "unknown",
        },
        "graph": {
            "strategy": "runtime_compile",
            "forward_symbol": "forward",
            "graph_stage": "inductor_transformed_fx",
            "graph_breaks": "unknown",
            "notes": [],
        },
        "inputs": {"strategy": "captured_call", "parameters": [], "kwargs": [], "artifacts": []},
        "weights": {"strategy": "inline|checkpoint|none|unknown", "artifacts": [], "load_sites": [], "required": False},
        "execution": {
            "backend": "inductor_autofuse",
            "device": "npu:0",
            "script_args": list(script_args),
            "capture_call": 0,
            "module_pattern": None,
            "fullgraph": False,
            "dynamic": False,
        },
        "diagnostics": {"confidence": 0.0, "assumptions": [], "missing": [], "candidates": []},
    }
    return (
        f"目标框架：{framework}\n"
        f"脚本路径：{path}\n"
        f"脚本参数（仅供识别，不要修改）：{json.dumps(list(script_args), ensure_ascii=False)}\n"
        "请按下面 JSON 模板返回完整对象，所有字段都必须保留；无法确认的值使用 null、空数组或 unknown。\n"
        f"模板：{json.dumps(template, ensure_ascii=False, indent=2)}\n\n"
        "待分析源码：\n"
        "```python\n"
        f"{source}\n"
        "```"
    )


def _recovery_system_prompt() -> str:
    return """你是 AutofuseX 的错误诊断 Agent。你只能分析错误文本、运行阶段、脚本路径和已有归一化 JSON，不能执行源码或命令，也不能假设已安装依赖。请严格只返回一个 JSON 对象。

给出证据、根因分类和按优先级排列的建议。任何 script_patch 只能是供用户审阅的 unified diff，不能要求本地工具自动应用。JSON 必须符合 autofusex-recovery/1 schema。"""


def _recovery_user_prompt(
    path: Path,
    framework: str,
    phase: str,
    error: BaseException,
    normalized: dict[str, Any] | None,
    traceback_text: str | None,
) -> str:
    return (
        f"脚本：{path}\n框架：{framework}\n失败阶段：{phase}\n"
        f"异常类型：{type(error).__name__}\n异常信息：{error}\n"
        f"完整 traceback：\n{traceback_text or '(未提供)'}\n"
        f"归一化结果：{json.dumps(normalized or {}, ensure_ascii=False, indent=2)}\n"
        "请判断这是入口识别、依赖/环境、输入 dtype/shape/device、编译配置、dump 采集还是工具缺陷，"
        "并返回结构化诊断与建议。"
    )
