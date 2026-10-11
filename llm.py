"""Local LLM interface - Ollama (HTTP) and llama.cpp (GGUF) backends.

Both backends run in structured-output mode: the JSON Schema that the caller
gives constrains decoding, so the model can only emit conforming JSON.
Hosted backends (Claude, OpenAI) are in providers.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM call fails at the transport / backend layer."""


@dataclass
class LLMResult:
    """One completion, plus what it cost.

    Every backend fills the same fields, so the runner can report cost per
    paper whether the model is local (cost 0) or hosted.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""


class BaseLLM:
    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      *, max_tokens: Optional[int] = None) -> LLMResult:
        """Return JSON conforming to `schema`, for a per-call system prompt."""
        raise NotImplementedError

    def warmup(self) -> None:
        """Load the model into memory without doing a full generation."""
        return None

    def describe(self) -> str:
        return "unknown backend"


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
class OllamaLLM(BaseLLM):
    """Talks to a local Ollama server via its /api/chat endpoint."""

    def __init__(self, cfg: Dict[str, Any]):
        self.model = cfg["model"]
        self.temperature = float(cfg.get("temperature", 0.0))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.num_predict = int(cfg.get("num_predict", 1200))
        self.num_ctx = int(cfg.get("num_ctx", 4096))

        ocfg = cfg.get("ollama", {}) or {}
        self.host = ocfg.get("host", "http://localhost:11434").rstrip("/")
        self.timeout = int(ocfg.get("timeout", 600))
        # Keep the model resident between calls; Ollama's default is 5 minutes,
        # which would unload it during a long batch.
        self.keep_alive = ocfg.get("keep_alive", "1h")

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      *, max_tokens: Optional[int] = None) -> LLMResult:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "format": schema,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": int(max_tokens or self.num_predict),
                "num_ctx": self.num_ctx,
            },
        }
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload,
                              timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        data = r.json()
        text = (data.get("message") or {}).get("content", "")
        if not text:
            raise LLMError(f"Empty response from Ollama: {data!r}")
        return LLMResult(
            text=text,
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
            cost_usd=0.0,          # a local model costs time, not money
            model=self.model,
        )

    def warmup(self) -> None:
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": "",
                      "keep_alive": self.keep_alive, "stream": False},
                timeout=self.timeout,
            )
            r.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(
                f"Ollama warmup failed: {exc}. Is `ollama serve` running and "
                f"has `ollama pull {self.model}` been done?"
            ) from exc

    def describe(self) -> str:
        return f"ollama:{self.model} @ {self.host}"


# --------------------------------------------------------------------------- #
# llama.cpp (GGUF)
# --------------------------------------------------------------------------- #
class LlamaCppLLM(BaseLLM):
    """Runs a local GGUF model through llama-cpp-python."""

    def __init__(self, cfg: Dict[str, Any]):
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "llama-cpp-python is not installed. "
                "Run `pip install llama-cpp-python` to use the llama_cpp backend."
            ) from exc

        lcfg = cfg.get("llama_cpp", {}) or {}
        model_path = lcfg.get("model_path")
        if not model_path:
            raise LLMError("gguf_path is required for the llama_cpp backend")

        self.model_path = model_path
        self.temperature = float(cfg.get("temperature", 0.0))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.num_predict = int(cfg.get("num_predict", 1200))

        log.info("Loading GGUF model from %s", model_path)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=int(lcfg.get("n_ctx", 4096)),
            n_gpu_layers=int(lcfg.get("n_gpu_layers", 0)),
            n_threads=int(lcfg.get("n_threads", 8)),
            verbose=False,
        )

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      *, max_tokens: Optional[int] = None) -> LLMResult:
        try:
            out = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=int(max_tokens or self.num_predict),
                response_format={"type": "json_object", "schema": schema},
            )
        except Exception as exc:
            raise LLMError(f"llama.cpp generation failed: {exc}") from exc
        try:
            text = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed llama.cpp response: {out!r}") from exc
        usage = out.get("usage", {}) or {}
        return LLMResult(
            text=text,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            cost_usd=0.0,
            model=self.model_path,
        )

    def describe(self) -> str:
        return f"llama_cpp:{self.model_path}"


# --------------------------------------------------------------------------- #
# Factory / diagnostics
# --------------------------------------------------------------------------- #
def build_llm(cfg: Dict[str, Any]) -> BaseLLM:
    backend = (cfg.get("backend") or "ollama").lower()
    if backend == "ollama":
        return OllamaLLM(cfg)
    if backend in {"llama_cpp", "llamacpp", "llama.cpp"}:
        return LlamaCppLLM(cfg)
    if backend in {"anthropic", "claude"}:
        # Imported here so the project runs with only the local backend
        # installed.
        from providers import AnthropicLLM

        return AnthropicLLM(cfg)
    if backend in {"openai", "gpt"}:
        from providers import OpenAILLM

        return OpenAILLM(cfg)
    raise LLMError(f"Unknown backend: {backend!r}")


BACKENDS = ["ollama", "anthropic", "openai", "llama_cpp"]


def health_check(llm: BaseLLM) -> Optional[str]:
    """Load the model and confirm it responds. None if OK, else an error string.

    Uses warmup rather than a real generation so startup stays cheap.
    """
    try:
        llm.warmup()
        return None
    except LLMError as exc:
        return str(exc)
    except Exception as exc:  # pragma: no cover
        return f"unexpected: {exc}"


def list_ollama_models(host: str = "http://localhost:11434") -> list[str]:
    """Return locally available Ollama model tags; empty list if unreachable."""
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except (requests.RequestException, ValueError, KeyError):
        return []
