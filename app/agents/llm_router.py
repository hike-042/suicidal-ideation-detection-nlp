"""
Shared LLM routing layer for website agents.

Primary behavior:
- Try Anthropic first when configured
- Automatically fail over to OpenRouter when the primary hits
  quota/rate-limit/credit exhaustion errors
- Return a unified response shape: text + usage + provider/model
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    anthropic = None


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str


class LLMRouterError(RuntimeError):
    pass


def _looks_like_capacity_error(message: str) -> bool:
    lowered = (message or "").lower()
    needles = (
        "rate limit",
        "quota",
        "insufficient credits",
        "credit balance",
        "billing",
        "too many requests",
        "overloaded",
        "capacity",
        "exceeded",
        "429",
        "402",
    )
    return any(n in lowered for n in needles)


class RoutedLLMClient:
    """
    Unified LLM caller with provider failover.
    """

    def __init__(self):
        self.primary_provider = os.environ.get("LLM_PRIMARY_PROVIDER", "openrouter").strip().lower()
        self.fallback_provider = os.environ.get("LLM_FALLBACK_PROVIDER", "openrouter").strip().lower()

        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.openrouter_base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
        self.openrouter_site_url = os.environ.get("OPENROUTER_SITE_URL", "http://localhost:8000")
        self.openrouter_site_name = os.environ.get("OPENROUTER_SITE_NAME", "MindGuard")

        self.openrouter_fast_model = os.environ.get("OPENROUTER_FAST_MODEL", "openrouter/free")
        self.openrouter_smart_model = os.environ.get("OPENROUTER_SMART_MODEL", "openrouter/free")

        self._anthropic_client = None
        if anthropic is not None and self.anthropic_api_key:
            try:
                self._anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            except Exception:
                self._anthropic_client = None

    def create_message(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict],
    ) -> LLMResponse:
        providers = [self.primary_provider]
        if self.fallback_provider and self.fallback_provider not in providers:
            providers.append(self.fallback_provider)

        last_error = None
        for idx, provider in enumerate(providers):
            try:
                return self._call_provider(
                    provider=provider,
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if idx == len(providers) - 1:
                    break
                if not _looks_like_capacity_error(str(exc)):
                    break

        raise LLMRouterError(str(last_error) if last_error else "Unknown LLM routing error.")

    def _call_provider(self, *, provider: str, model: str, max_tokens: int, system: str, messages: list[dict]) -> LLMResponse:
        if provider == "anthropic":
            return self._call_anthropic(model=model, max_tokens=max_tokens, system=system, messages=messages)
        if provider == "openrouter":
            return self._call_openrouter(model=model, max_tokens=max_tokens, system=system, messages=messages)
        raise LLMRouterError(f"Unsupported provider: {provider}")

    def _call_anthropic(self, *, model: str, max_tokens: int, system: str, messages: list[dict]) -> LLMResponse:
        if self._anthropic_client is None:
            raise LLMRouterError("Anthropic client is not configured.")

        response = self._anthropic_client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return LLMResponse(
            text=response.content[0].text.strip(),
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            model=model,
            provider="anthropic",
        )

    def _call_openrouter(self, *, model: str, max_tokens: int, system: str, messages: list[dict]) -> LLMResponse:
        if not self.openrouter_api_key:
            raise LLMRouterError("OpenRouter is not configured.")

        selected_model = self._map_model_for_openrouter(model)
        payload = {
            "model": selected_model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.openrouter_base_url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": self.openrouter_site_url,
                "X-Title": self.openrouter_site_name,
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise LLMRouterError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMRouterError(f"OpenRouter network error: {exc}") from exc

        try:
            choice = body["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise LLMRouterError(f"Malformed OpenRouter response: {body}") from exc

        usage = body.get("usage", {})
        return LLMResponse(
            text=choice.strip(),
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            model=selected_model,
            provider="openrouter",
        )

    def _map_model_for_openrouter(self, model: str) -> str:
        lowered = model.lower()
        if "haiku" in lowered:
            return self.openrouter_fast_model
        if "sonnet" in lowered or "opus" in lowered:
            return self.openrouter_smart_model
        return self.openrouter_fast_model
