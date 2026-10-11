"""Hosted model backends: Claude and OpenAI, behind the same interface.

llm.BaseLLM defines one method the extraction pipeline uses -
`complete_json(system, user, schema)` - and every backend returns the same
LLMResult. Swapping Claude for GPT, or either for a local model, is a setting,
not a code change.

Neither SDK is imported until a backend is built, so the project still runs
with only the local Ollama backend installed.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from llm import BaseLLM, LLMError, LLMResult

log = logging.getLogger(__name__)

# US dollars per million tokens, (input, output). Cache reads are billed lower
# than fresh input; the saving is reported but not itemised per model here.
PRICING: Dict[str, tuple[float, float]] = {
    # Claude
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5-1": (10.00, 50.00),
    # OpenAI
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}

CLAUDE_MODELS = [
    "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-opus-4-8",
]
OPENAI_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"]

DEFAULT_CLAUDE_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1"


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  cached_input_tokens: int = 0) -> float:
    """Cost in US dollars. Returns 0.0 for a model with no published price."""
    price = PRICING.get(model)
    if not price:
        return 0.0
    fresh = max(0, input_tokens - cached_input_tokens)
    # A cache read is billed at roughly a tenth of the input rate.
    return (
        fresh * price[0] / 1_000_000.0
        + cached_input_tokens * price[0] * 0.1 / 1_000_000.0
        + output_tokens * price[1] / 1_000_000.0
    )


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #
class AnthropicLLM(BaseLLM):
    """Claude through the official Anthropic SDK, with structured outputs."""

    def __init__(self, cfg: Dict[str, Any]):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install
            raise LLMError(
                "The anthropic package is not installed. Run "
                "`pip install anthropic` to use the Claude backend."
            ) from exc

        self._anthropic = anthropic
        self.model = cfg.get("model") or DEFAULT_CLAUDE_MODEL
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.effort = str(cfg.get("effort", "low"))
        self.timeout = float(cfg.get("timeout", 600))
        self.max_attempts = int(cfg.get("max_attempts", 3))
        # Per-column extraction is a focused reading task, so the default
        # effort is low. Raise it for columns that need real reasoning.
        api_key = cfg.get("api_key") or None
        try:
            self.client = anthropic.Anthropic(api_key=api_key,
                                              timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Could not create the Anthropic client: {exc}") from exc

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      *, max_tokens: Optional[int] = None) -> LLMResult:
        last_error: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=int(max_tokens or self.max_tokens),
                    # The system prompt is identical for every paper in a run,
                    # so caching it turns the biggest fixed cost into a cache
                    # read after the first call.
                    system=[{
                        "type": "text", "text": system,
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }],
                    messages=[{"role": "user", "content": user}],
                    output_config={
                        "format": {"type": "json_schema", "schema": schema},
                        "effort": self.effort,
                    },
                )
            except self._anthropic.APIStatusError as exc:
                last_error = f"{exc.status_code}: {exc}"
                if exc.status_code in (429, 500, 502, 503, 529) and \
                        attempt < self.max_attempts:
                    time.sleep(min(60.0, 2.0 ** attempt))
                    continue
                raise LLMError(f"Claude request failed: {last_error}") from exc
            except self._anthropic.APIConnectionError as exc:
                last_error = str(exc)
                if attempt < self.max_attempts:
                    time.sleep(min(60.0, 2.0 ** attempt))
                    continue
                raise LLMError(f"Claude connection failed: {exc}") from exc

            if getattr(response, "stop_reason", "") == "refusal":
                details = getattr(response, "stop_details", None)
                raise LLMError(
                    "Claude declined this request "
                    f"({getattr(details, 'category', 'unknown')})."
                )

            text = ""
            for block in response.content:
                if getattr(block, "type", "") == "text":
                    text = block.text
                    break

            usage = response.usage
            cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            created = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            input_tokens = int(usage.input_tokens) + cached + created
            output_tokens = int(usage.output_tokens)
            return LLMResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached,
                cost_usd=estimate_cost(self.model, input_tokens,
                                       output_tokens, cached),
                model=self.model,
            )
        raise LLMError(f"Claude request failed: {last_error}")

    def warmup(self) -> None:
        try:
            self.client.models.retrieve(self.model)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"Could not reach the Claude API for model {self.model!r}: "
                f"{exc}. Check ANTHROPIC_API_KEY, or run `ant auth login`."
            ) from exc

    def describe(self) -> str:
        return f"anthropic:{self.model} (effort={self.effort})"


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAILLM(BaseLLM):
    """GPT through the official OpenAI SDK, with a strict JSON schema."""

    def __init__(self, cfg: Dict[str, Any]):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on install
            raise LLMError(
                "The openai package is not installed. Run "
                "`pip install openai` to use the OpenAI backend."
            ) from exc

        self.model = cfg.get("model") or DEFAULT_OPENAI_MODEL
        self.max_tokens = int(cfg.get("max_tokens", 4096))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.timeout = float(cfg.get("timeout", 600))
        self.max_attempts = int(cfg.get("max_attempts", 3))
        try:
            self.client = OpenAI(api_key=cfg.get("api_key") or None,
                                 timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Could not create the OpenAI client: {exc}") from exc

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      *, max_tokens: Optional[int] = None) -> LLMResult:
        last_error: Optional[str] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=int(max_tokens or self.max_tokens),
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "column_extraction",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
            except Exception as exc:  # noqa: BLE001 - SDK raises many types
                last_error = str(exc)
                if attempt < self.max_attempts:
                    time.sleep(min(60.0, 2.0 ** attempt))
                    continue
                raise LLMError(f"OpenAI request failed: {exc}") from exc

            text = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            cached = 0
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = int(getattr(details, "cached_tokens", 0) or 0)
            return LLMResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached,
                cost_usd=estimate_cost(self.model, input_tokens,
                                       output_tokens, cached),
                model=self.model,
            )
        raise LLMError(f"OpenAI request failed: {last_error}")

    def warmup(self) -> None:
        try:
            self.client.models.retrieve(self.model)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"Could not reach the OpenAI API for model {self.model!r}: "
                f"{exc}. Check OPENAI_API_KEY."
            ) from exc

    def describe(self) -> str:
        return f"openai:{self.model}"


def available_models(backend: str) -> List[str]:
    if backend == "anthropic":
        return list(CLAUDE_MODELS)
    if backend == "openai":
        return list(OPENAI_MODELS)
    return []
