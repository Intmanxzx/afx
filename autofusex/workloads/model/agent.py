from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AgentError(RuntimeError):
    """Raised when an Agent request or its JSON response is invalid."""


@dataclass(frozen=True)
class AgentConfig:
    endpoint: str | None = None
    api_key: str | None = None
    model: str = "default"
    timeout: float = 90.0
    json_mode: bool = True
    transport: str = "http"
    command: tuple[str, ...] | None = None

    @classmethod
    def from_env(
        cls,
        *,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        transport: str | None = None,
        command: str | None = None,
    ) -> "AgentConfig":
        resolved_endpoint = endpoint or os.environ.get("AUTOFUSEX_AGENT_ENDPOINT", "")
        requested_transport = (transport or os.environ.get("AUTOFUSEX_AGENT_TRANSPORT", "auto")).lower()
        if requested_transport not in {"auto", "http", "cannbot"}:
            raise AgentError(f"unsupported Agent transport: {requested_transport!r}")
        raw_command = command
        if raw_command is None:
            raw_command = os.environ.get("AUTOFUSEX_AGENT_COMMAND")
        resolved_command = _split_command(raw_command)
        if requested_transport == "auto":
            if resolved_endpoint:
                resolved_transport = "http"
            elif resolved_command:
                resolved_transport = "cannbot"
            elif shutil.which("cannbot"):
                resolved_transport = "cannbot"
                resolved_command = ("cannbot",)
            else:
                raise AgentError(
                    "Agent endpoint or cannbot command is required; pass --agent-endpoint, "
                    "--agent-command, or set AUTOFUSEX_AGENT_ENDPOINT/"
                    "AUTOFUSEX_AGENT_COMMAND"
                )
        elif requested_transport == "http":
            resolved_transport = "http"
            if not resolved_endpoint:
                raise AgentError(
                    "Agent endpoint is required for HTTP transport; pass --agent-endpoint or set "
                    "AUTOFUSEX_AGENT_ENDPOINT"
                )
        else:
            resolved_transport = "cannbot"
            if not resolved_command:
                executable = shutil.which("cannbot")
                if executable:
                    resolved_command = (executable,)
                else:
                    raise AgentError(
                        "cannbot command is required; pass --agent-command or set "
                        "AUTOFUSEX_AGENT_COMMAND"
                    )
        resolved_timeout = timeout
        if resolved_timeout is None:
            raw_timeout = os.environ.get("AUTOFUSEX_AGENT_TIMEOUT", "90")
            try:
                resolved_timeout = float(raw_timeout)
            except ValueError as exc:
                raise AgentError(f"invalid AUTOFUSEX_AGENT_TIMEOUT: {raw_timeout!r}") from exc
        if resolved_timeout <= 0:
            raise AgentError("Agent timeout must be positive")
        json_mode = os.environ.get("AUTOFUSEX_AGENT_JSON_MODE", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        return cls(
            endpoint=resolved_endpoint or None,
            api_key=os.environ.get("AUTOFUSEX_AGENT_API_KEY"),
            model=model or os.environ.get("AUTOFUSEX_AGENT_MODEL", "default"),
            timeout=resolved_timeout,
            json_mode=json_mode,
            transport=resolved_transport,
            command=resolved_command,
        )


class AgentClient:
    """Small dependency-free client for HTTP and local CANNBot agents."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def complete_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.config.transport == "cannbot":
            return self._complete_with_cannbot(system_prompt=system_prompt, user_prompt=user_prompt)
        return self._complete_with_http(system_prompt=system_prompt, user_prompt=user_prompt)

    def _complete_with_http(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.config.endpoint:
            raise AgentError("Agent HTTP endpoint is not configured")
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        if self.config.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(
            self.config.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise AgentError(f"Agent HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise AgentError(f"Agent connection failed: {exc.reason}") from exc
        except OSError as exc:
            raise AgentError(f"Agent request failed: {exc}") from exc

        try:
            response_data = json.loads(raw)
        except JSONDecodeError as exc:
            raise AgentError(f"Agent returned invalid response JSON: {exc}") from exc
        content = _response_content(response_data)
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise AgentError("Agent response does not contain JSON text")
        return _parse_json_text(content)

    def _complete_with_cannbot(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.config.command:
            raise AgentError("cannbot command is not configured")
        prompt = f"{system_prompt}\n\nUser request:\n{user_prompt}"
        command = [
            *self.config.command,
            "run",
            "--format",
            "json",
            "--model",
            self.config.model,
            prompt,
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise AgentError(f"cannbot request timed out after {self.config.timeout}s") from exc
        except OSError as exc:
            raise AgentError(f"failed to start cannbot: {exc}") from exc
        content = _cannbot_text(completed.stdout)
        if completed.returncode != 0:
            detail = (content or completed.stderr or completed.stdout).strip()[:2000]
            raise AgentError(f"cannbot exited with code {completed.returncode}: {detail}")
        if not content:
            detail = (completed.stderr or completed.stdout).strip()[:1000]
            raise AgentError(f"cannbot returned no text response: {detail}")
        return _parse_json_text(content)


def _response_content(response: Any) -> Any:
    """Extract content from direct JSON, Chat Completions, or Responses output."""
    if isinstance(response, dict) and response.get("schema"):
        return response
    if isinstance(response, dict) and isinstance(response.get("output_text"), str):
        return response["output_text"]
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, (str, dict)):
                        return content
                    if isinstance(content, list):
                        return _content_parts(content)
                if isinstance(choice.get("text"), str):
                    return choice["text"]
        output = response.get("output")
        if isinstance(output, list):
            parts: list[dict[str, Any]] = []
            for item in output:
                if isinstance(item, dict) and isinstance(item.get("content"), list):
                    parts.extend(part for part in item["content"] if isinstance(part, dict))
            return _content_parts(parts)
    return response


def _split_command(value: str | None) -> tuple[str, ...] | None:
    if not value or not value.strip():
        return None
    try:
        command = tuple(shlex.split(value))
    except ValueError as exc:
        raise AgentError(f"invalid Agent command: {exc}") from exc
    return command or None


def _cannbot_text(stdout: str) -> str:
    """Collect text events emitted by ``cannbot run --format json``."""
    text_parts: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "text":
            continue
        part = event.get("part")
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    return "\n".join(text_parts).strip() or stdout.strip()


def _content_parts(parts: list[dict[str, Any]]) -> str:
    texts = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str):
            texts.append(text)
        elif isinstance(text, dict) and isinstance(text.get("value"), str):
            texts.append(text["value"])
    return "".join(texts)


def _parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
    except JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise AgentError("Agent response does not contain a JSON object") from None
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        except JSONDecodeError as exc:
            raise AgentError(f"Agent response JSON object is invalid: {exc}") from exc
    if not isinstance(parsed, dict):
        raise AgentError("Agent response JSON must be an object")
    return parsed
