"""Text embeddings behind one small interface, with swappable providers.

One model serves two jobs: routing a user-defined column to a paper section
(grouping.py) and semantic retrieval over chunks (index.py).

Providers
---------
* ``sentence_transformers`` (default): runs locally and needs no API key.
* ``openai``: the OpenAI embeddings API. Needs ``OPENAI_API_KEY``.

To add a provider, subclass ``Embedder``, implement ``_load`` and ``_encode``,
and call ``register_provider("name", YourClass)``. Nothing else changes.

The model loads lazily, on the first text that is not in the cache. A run over
papers that are already embedded therefore never loads the model.

If the model cannot be loaded the pipeline still runs. Retrieval falls back to
BM25 alone and column routing falls back to the cue lists.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import threading
from typing import Callable, Dict, List, Optional, Sequence, Type

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_PROVIDER = "sentence_transformers"
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Encoded to fingerprint the loaded weights. If the same namespace name ever
# maps to different weights, the fingerprint differs and the cache refuses to
# mix the two sets of vectors.
_PROBE_TEXT = "Rodents were trapped at 12 sites and screened by RT-PCR."


class EmbeddingModelChanged(RuntimeError):
    """The model behind a namespace is not the model that wrote it."""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-").lower() or "model"


class Embedder:
    """Base class. Thread-safe, lazy, and counts every encode call."""

    provider = "base"

    def __init__(self, model_name: str = DEFAULT_MODEL, revision: str = "",
                 **options: object):
        self.model_name = model_name
        self.revision = (revision or "").strip()
        self.options = options
        self.dim = 0
        self.error: Optional[str] = None
        self.encode_calls = 0      # number of calls that reached the model
        self.texts_encoded = 0     # number of texts sent to the model
        self._loaded = False
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    # ----------------------------------------------------------------- #
    @property
    def namespace(self) -> str:
        """Cache namespace: provider, model and revision.

        A different model, or a different revision of the same model, gets a
        different namespace. Vectors from two models are never mixed.
        """
        return (f"{_slug(self.provider)}__{_slug(self.model_name)}"
                f"__{_slug(self.revision) if self.revision else 'default'}")

    def installed(self) -> bool:
        """Cheap check that the provider can be used at all. No model load."""
        return True

    @property
    def available(self) -> bool:
        """False only when the provider is missing or the load has failed."""
        if self.error is not None:
            return False
        if self._loaded:
            return True
        return self.installed()

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        """Load the model once. Returns True when it is ready."""
        if self._loaded:
            return True
        if self.error is not None:
            return False
        with self._load_lock:
            if self._loaded:
                return True
            try:
                self._load()
                self._loaded = True
            except Exception as exc:  # noqa: BLE001 - loaders raise many types
                self.error = (
                    f"Could not load embedding model {self.provider}:"
                    f"{self.model_name} ({exc}). Semantic retrieval is off; "
                    "BM25 keyword search still runs.")
                log.warning(self.error)
        return self._loaded

    def _load(self) -> None:
        raise NotImplementedError

    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        raise NotImplementedError

    # ----------------------------------------------------------------- #
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalized float32 vectors, shape (len(texts), dim).

        Returns a (len(texts), 0) array when the model is unavailable, so
        callers can test `.size` instead of branching on a None.
        """
        items: List[str] = [t if t else " " for t in texts]
        if not items:
            return np.zeros((0, max(self.dim, 1)), dtype=np.float32)
        if not self.load():
            return np.zeros((len(items), 0), dtype=np.float32)
        with self._encode_lock:
            self.encode_calls += 1
            self.texts_encoded += len(items)
            vectors = self._encode(items, batch_size)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)

    def fingerprint(self) -> str:
        """Short hash of the loaded weights, from one probe sentence."""
        if not self.load():
            return ""
        vector = self._encode([_PROBE_TEXT], 1)
        rounded = np.round(np.asarray(vector, dtype=np.float32), 3)
        return hashlib.sha1(rounded.tobytes()).hexdigest()[:12]

    def describe(self) -> str:
        if self.error:
            return f"unavailable: {self.error}"
        state = f"dim {self.dim}" if self._loaded else "not loaded yet"
        return f"{self.provider}:{self.model_name} ({state})"


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #
class SentenceTransformersEmbedder(Embedder):
    """A local sentence-transformers model. No API key, no network after the
    first download of the weights."""

    provider = "sentence_transformers"

    def installed(self) -> bool:
        if importlib.util.find_spec("sentence_transformers") is None:
            self.error = ("sentence-transformers is not installed. Semantic "
                          "retrieval is off; BM25 keyword search still runs.")
            return False
        return True

    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        if self.revision:
            kwargs["revision"] = self.revision
        device = self.options.get("device")
        if device:
            kwargs["device"] = str(device)
        self._model = SentenceTransformer(self.model_name, **kwargs)
        get_dim = (getattr(self._model, "get_embedding_dimension", None)
                   or self._model.get_sentence_embedding_dimension)
        self.dim = int(get_dim())

    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        return self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=True)


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings API, for example text-embedding-3-small."""

    provider = "openai"

    def installed(self) -> bool:
        if importlib.util.find_spec("openai") is None:
            self.error = "The openai package is not installed (pip install openai)."
            return False
        if not (self.options.get("api_key") or os.environ.get("OPENAI_API_KEY")):
            self.error = "OPENAI_API_KEY is not set."
            return False
        return True

    def _load(self) -> None:
        from openai import OpenAI

        api_key = self.options.get("api_key") or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=str(api_key) if api_key else None)
        probe = self._client.embeddings.create(model=self.model_name,
                                               input=["dimension probe"])
        self.dim = len(probe.data[0].embedding)

    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        out: List[List[float]] = []
        step = max(1, min(batch_size * 8, 512))
        for start in range(0, len(texts), step):
            response = self._client.embeddings.create(
                model=self.model_name, input=texts[start:start + step])
            out.extend(item.embedding for item in response.data)
        return np.asarray(out, dtype=np.float32)


_PROVIDERS: Dict[str, Type[Embedder]] = {}


def register_provider(name: str, factory: Type[Embedder]) -> None:
    """Make a provider available under `name` in config and the CLI."""
    _PROVIDERS[name] = factory


def providers() -> List[str]:
    return sorted(_PROVIDERS)


register_provider("sentence_transformers", SentenceTransformersEmbedder)
register_provider("openai", OpenAIEmbedder)


def build_embedder(provider: str = DEFAULT_PROVIDER,
                   model_name: str = DEFAULT_MODEL,
                   revision: str = "", **options: object) -> Embedder:
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(f"Unknown embedding provider {provider!r}. "
                         f"Known: {', '.join(providers())}")
    return factory(model_name, revision, **options)


# --------------------------------------------------------------------------- #
# Process-wide instance
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()
_INSTANCES: Dict[tuple, Embedder] = {}


def get_embedder(model_name: str = DEFAULT_MODEL,
                 provider: str = DEFAULT_PROVIDER,
                 revision: str = "") -> Embedder:
    """Return the shared embedder for this provider and model.

    Loading a model twice wastes a few hundred megabytes, so the app and the
    pipeline share one instance per (provider, model, revision).
    """
    key = (provider, model_name, revision)
    with _LOCK:
        if key not in _INSTANCES:
            _INSTANCES[key] = build_embedder(provider, model_name, revision)
        return _INSTANCES[key]


def embedder_from_settings(settings: Dict[str, object],
                           factory: Optional[Callable[..., Embedder]] = None,
                           ) -> Embedder:
    """The shared embedder that a settings dictionary asks for."""
    getter = factory or get_embedder
    return getter(
        str(settings.get("embedding_model") or DEFAULT_MODEL),
        provider=str(settings.get("embedding_provider") or DEFAULT_PROVIDER),
        revision=str(settings.get("embedding_revision") or ""),
    )


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one normalized query against normalized rows."""
    if query.size == 0 or matrix.size == 0:
        return np.zeros((matrix.shape[0],), dtype=np.float32)
    return matrix.astype(np.float32) @ query.astype(np.float32).ravel()
