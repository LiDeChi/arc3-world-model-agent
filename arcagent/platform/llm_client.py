from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_ENV_PATHS = [
    Path.home() / ".local/share/litellm-sub2api-native/.env",
    Path("/Users/lidechi/Documents/Github/llm-layer/local/litellm-sub2api/.env"),
]


@dataclass
class LLMResult:
    ok: bool
    model: str
    content: str
    error: str | None = None
    raw: dict[str, Any] | None = None


class LiteLLMClient:
    """Tiny stdlib-only client for the local LiteLLM gateway.

    This is training/platform code, not submission-critical inference code. It
    deliberately routes through localhost and never calls provider URLs directly.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        env_path: str | Path | None = None,
        timeout: float = 45.0,
    ):
        loaded = load_litellm_env(env_path)
        port = os.environ.get("LITELLM_GATEWAY_PORT") or loaded.get("LITELLM_GATEWAY_PORT") or "41401"
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or f"http://127.0.0.1:{port}").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("LITELLM_MASTER_KEY") or loaded.get("LITELLM_MASTER_KEY") or ""
        self.timeout = timeout

    def health(self) -> LLMResult:
        try:
            with urllib.request.urlopen(self.base_url + "/health/liveliness", timeout=5) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            return LLMResult(True, "health", text)
        except Exception as exc:  # pragma: no cover - depends on local service
            return LLMResult(False, "health", "", repr(exc))

    def chat(self, messages: list[dict[str, str]], models: list[str], temperature: float = 0.2, max_tokens: int = 700) -> LLMResult:
        last_error: str | None = None
        for model in models:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            req = urllib.request.Request(
                self.base_url + "/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                message = data.get("choices", [{}])[0].get("message", {})
                content = message.get("content") or message.get("reasoning_content") or ""
                if not content and isinstance(message.get("provider_specific_fields"), dict):
                    content = message["provider_specific_fields"].get("reasoning_content", "")
                return LLMResult(True, model, content, raw=data)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {body[:1200]}"
            except Exception as exc:  # pragma: no cover - depends on local service
                last_error = repr(exc)
        return LLMResult(False, models[-1] if models else "", "", last_error)


def load_litellm_env(env_path: str | Path | None = None) -> dict[str, str]:
    paths = [Path(env_path)] if env_path else DEFAULT_ENV_PATHS
    values: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction for LLM replies that may wrap JSON in prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in LLM response")
