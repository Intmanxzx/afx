from __future__ import annotations

import json
from pathlib import Path

import pytest

import autofusex.workloads.model.agent as agent
import autofusex.workloads.model.script_runner as script_runner
from autofusex.workloads.model.agent import AgentClient, AgentConfig, AgentError
from autofusex.workloads.model.adapters import CapturedInvocation
from autofusex.cli import build_parser
from autofusex.workloads.model.normalization import (
    NORMALIZATION_SCHEMA,
    RECOVERY_SCHEMA,
    normalize_script,
    recover_script_error,
    validate_normalization,
)


class _Response:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload.encode("utf-8")


def _normalization_response() -> dict:
    return {
        "schema": NORMALIZATION_SCHEMA,
        "status": "ready",
        "framework": "pytorch",
        "source": {
            "script": "agent.py",
            "project_root": None,
            "imports": ["torch"],
            "entrypoint": {
                "kind": "module_call",
                "symbol": "Model",
                "module_pattern": "Model",
                "capture_call": 0,
            },
        },
        "model": {
            "kind": "nn.Module",
            "class_name": "Model",
            "construction": {
                "strategy": "script_runtime",
                "symbol": "Model",
                "location": "agent.py:4",
                "arguments": {},
            },
            "training": "eval",
        },
        "graph": {
            "strategy": "runtime_compile",
            "forward_symbol": "forward",
            "graph_stage": "inductor_transformed_fx",
            "graph_breaks": "unknown",
            "notes": [],
        },
        "inputs": {
            "strategy": "captured_call",
            "parameters": [{"name": "x", "shape": [2], "dtype": "float32", "source": "forward call"}],
            "kwargs": [],
            "artifacts": [],
        },
        "weights": {"strategy": "inline", "artifacts": [], "load_sites": [], "required": False},
        "execution": {
            "backend": "inductor_autofuse",
            "device": "npu:0",
            "script_args": [],
            "capture_call": 0,
            "module_pattern": "Model",
            "fullgraph": False,
            "dynamic": False,
        },
        "diagnostics": {"confidence": 0.95, "assumptions": [], "missing": [], "candidates": []},
    }


def test_agent_client_parses_openai_chat_json_and_sends_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float):
        received["request"] = request
        received["timeout"] = timeout
        return _Response('{"choices":[{"message":{"content":"```json\\n{\\"ok\\":true}\\n```"}}]}')

    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    result = AgentClient(
        AgentConfig(endpoint="https://agent.invalid/v1/chat/completions", api_key="secret", model="m")
    ).complete_json(system_prompt="system", user_prompt="user")

    request = received["request"]
    assert result == {"ok": True}
    assert received["timeout"] == 90.0
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == "m"
    assert body["messages"][1]["content"] == "user"


def test_agent_client_runs_cannbot_and_collects_text_event(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class Completed:
        returncode = 0
        stdout = (
            '{"type":"step_start"}\n'
            '{"type":"text","part":{"type":"text","text":"{\\"status\\":\\"ready\\"}"}}\n'
            '{"type":"step_finish"}\n'
        )
        stderr = ""

    def fake_run(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr(agent.shutil, "which", lambda name: "/opt/cannbot/bin/cannbot")
    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    config = AgentConfig.from_env(transport="auto", model="cannbot/qwen3.8-flash", timeout=12)
    result = AgentClient(config).complete_json(system_prompt="system", user_prompt="user")

    assert config.transport == "cannbot"
    assert config.command == ("cannbot",) or config.command == ("/opt/cannbot/bin/cannbot",)
    assert result == {"status": "ready"}
    assert calls["command"][-2:] == ["cannbot/qwen3.8-flash", "system\n\nUser request:\nuser"]
    assert calls["kwargs"]["timeout"] == 12


def test_normalize_script_validates_and_overwrites_source_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "model.py"
    script.write_text("import torch\n", encoding="utf-8")
    response = _normalization_response()
    monkeypatch.setattr(AgentClient, "complete_json", lambda self, **kwargs: response)
    output = tmp_path / "out" / "agent_normalization.json"

    normalized = normalize_script(
        script=script,
        framework="pytorch",
        script_args=("--device", "npu"),
        config=AgentConfig(endpoint="https://agent.invalid"),
        output=output,
    )

    assert normalized["source"]["script"] == str(script.resolve())
    assert normalized["execution"]["script_args"] == ["--device", "npu"]
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == NORMALIZATION_SCHEMA


def test_normalization_rejects_incomplete_agent_response() -> None:
    with pytest.raises(AgentError, match="missing fields"):
        validate_normalization({"schema": NORMALIZATION_SCHEMA})


def test_normalization_accepts_factory_function_entrypoint() -> None:
    response = _normalization_response()
    response["source"]["entrypoint"]["kind"] = "function_call"
    response["inputs"]["strategy"] = "generated_random"
    validated = validate_normalization(response)
    assert validated["source"]["entrypoint"]["kind"] == "function_call"
    assert validated["inputs"]["strategy"] == "generated_random"


def test_recovery_wraps_incomplete_agent_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AgentClient, "complete_json", lambda self, **kwargs: {"schema": "wrong"})
    output = tmp_path / "agent_recovery.json"

    recovery = recover_script_error(
        script=tmp_path / "model.py",
        framework="pytorch",
        phase="framework adapter capture",
        error=RuntimeError("capture failed"),
        normalized=None,
        config=AgentConfig(endpoint="https://agent.invalid"),
        output=output,
    )

    assert recovery["schema"] == RECOVERY_SCHEMA
    assert recovery["status"] == "not_actionable"
    assert recovery["phase"] == "framework adapter capture"
    assert recovery["diagnosis"]["category"] == "agent_protocol"
    assert json.loads(output.read_text(encoding="utf-8"))["schema"] == RECOVERY_SCHEMA


def test_cli_exposes_agent_normalization_and_recovery_options() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "model", "run",
            "--script",
            "model.py",
            "--out-dir",
            "out",
            "--agent-normalize",
            "--agent-recover",
            "--agent-endpoint",
            "https://agent.invalid",
            "--agent-model",
            "m",
            "--",
            "--device",
            "npu",
        ]
    )
    assert args.agent_normalize is True
    assert args.agent_recover is True
    assert args.agent_endpoint == "https://agent.invalid"
    assert args.script_args == ["--", "--device", "npu"]


def test_run_script_uses_agent_entrypoint_hints_without_executing_agent_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "model.py"
    script.write_text("import torch\n", encoding="utf-8")
    normalized = _normalization_response()
    normalized["source"]["entrypoint"]["module_pattern"] = "Target"
    normalized["execution"]["capture_call"] = 2
    captured: dict[str, object] = {}

    def fake_normalize_script(**kwargs):
        captured["normalization"] = kwargs
        output = Path(kwargs["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(normalized), encoding="utf-8")
        return normalized

    class FakeFrontend:
        def capture(self, **kwargs):
            captured["capture"] = kwargs
            return CapturedInvocation("pytorch", object(), (), {}, "Target", {})

    class FakeBackend:
        def supports(self, framework):
            return framework == "pytorch"

        def run(self, invocation, **kwargs):
            captured["invocation_metadata"] = invocation.metadata
            return {"ok": True, "kwargs": kwargs}

    monkeypatch.setattr(script_runner, "normalize_script", fake_normalize_script)
    monkeypatch.setattr(script_runner, "get_framework_adapter", lambda name: FakeFrontend())
    monkeypatch.setattr(script_runner, "get_backend_adapter", lambda name: FakeBackend())

    result = script_runner.run_framework_script(
        script=script,
        out_dir=tmp_path / "out",
        framework="pytorch",
        agent_normalize=True,
        agent_endpoint="https://agent.invalid",
    )

    assert result["ok"] is True
    assert captured["capture"]["capture_call"] == 2
    assert captured["capture"]["module_pattern"] == "Target"
    assert captured["invocation_metadata"]["agent_normalization"]["status"] == "ready"


def test_run_script_keeps_cann_ge_entry_selection_owned_by_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "tc_case.py"
    script.write_text("from libs.tools import run_af\n", encoding="utf-8")
    normalized = _normalization_response()
    normalized["source"]["entrypoint"]["module_pattern"] = "WrongPattern"
    normalized["execution"]["capture_call"] = 3
    captured: dict[str, object] = {}

    monkeypatch.setattr(script_runner, "normalize_script", lambda **kwargs: normalized)
    monkeypatch.setattr(script_runner, "find_cann_ge_root", lambda path: tmp_path)

    class FakeFrontend:
        def capture(self, **kwargs):
            captured["capture"] = kwargs
            return CapturedInvocation("pytorch", object(), (), {}, "Target", {})

    class FakeBackend:
        def supports(self, framework):
            return framework == "pytorch"

        def run(self, invocation, **kwargs):
            return {"ok": True}

    monkeypatch.setattr(script_runner, "get_framework_adapter", lambda name: FakeFrontend())
    monkeypatch.setattr(script_runner, "get_backend_adapter", lambda name: FakeBackend())

    assert script_runner.run_framework_script(
        script=script,
        out_dir=tmp_path / "out",
        framework="pytorch",
        agent_normalize=True,
        agent_endpoint="https://agent.invalid",
    ) == {"ok": True}
    assert captured["capture"]["capture_call"] == 0
    assert captured["capture"]["module_pattern"] is None


def test_run_script_saves_agent_recovery_plan_and_preserves_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "model.py"
    script.write_text("import torch\n", encoding="utf-8")
    normalized = _normalization_response()
    recovery_calls: dict[str, object] = {}

    def fake_normalize_script(**kwargs):
        return normalized

    def fake_recover_script_error(**kwargs):
        recovery_calls.update(kwargs)
        Path(kwargs["output"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output"]).write_text("{}", encoding="utf-8")
        return {}

    class FakeFrontend:
        def capture(self, **kwargs):
            raise RuntimeError("capture failed")

    class FakeBackend:
        def supports(self, framework):
            return True

    monkeypatch.setattr(script_runner, "normalize_script", fake_normalize_script)
    monkeypatch.setattr(script_runner, "recover_script_error", fake_recover_script_error)
    monkeypatch.setattr(script_runner, "get_framework_adapter", lambda name: FakeFrontend())
    monkeypatch.setattr(script_runner, "get_backend_adapter", lambda name: FakeBackend())

    with pytest.raises(RuntimeError, match="capture failed.*Agent recovery plan"):
        script_runner.run_framework_script(
            script=script,
            out_dir=tmp_path / "out",
            framework="pytorch",
            agent_recover=True,
            agent_endpoint="https://agent.invalid",
        )

    assert recovery_calls["phase"] == "framework adapter capture"
    assert (tmp_path / "out" / "agent_recovery.json").is_file()
