"""Wiring for the column-extraction pipeline, shared by the app and the CLI.

Everything the two front ends need in one place: where files live, how
settings are read, how a run is created or resumed, and how a run is executed
and exported.

Settings come from four places. A later one wins:

1. DEFAULT_SETTINGS below
2. config.yaml (every section is merged into one flat dictionary)
3. environment variables, also read from a .env file (see ENV_SETTINGS)
4. explicit overrides from the CLI or the app
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml

import export
import grouping
import runner as runner_module
import validate
from cache import PaperCache
from columns import ColumnSpec
from embedder import Embedder, embedder_from_settings
from fetch import Fetcher, PaperRef
from grouping import ColumnGroup
from llm import BaseLLM, build_llm
from runstore import RunInfo, RunRegistry, RunStore

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
DEFAULT_CACHE_DIR = "~/.paper-extractor/cache"
DEFAULT_RUNS_DIR = "runs"

DEFAULT_SETTINGS: Dict[str, Any] = {
    # Paths
    "cache_dir": DEFAULT_CACHE_DIR,
    "runs_dir": DEFAULT_RUNS_DIR,
    # Model
    "backend": "ollama",
    "model": "qwen2.5:7b",
    "host": "http://localhost:11434",
    "temperature": 0.0,
    "num_ctx": 8192,
    "num_predict": 2048,
    "keep_alive": "1h",
    "effort": "low",
    "api_key": "",
    "gguf_path": "",
    "n_gpu_layers": 0,
    # Embeddings
    "embedding_provider": "sentence_transformers",
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_revision": "",
    # Chunking (a change writes to a new cache namespace)
    "chunk_target_tokens": 400,
    "chunk_overlap_tokens": 80,
    "chunk_min_tokens": 20,
    # Retrieval and validation
    "top_k": 12,
    "token_budget": 2600,
    "confidence_threshold": validate.DEFAULT_CONFIDENCE_THRESHOLD,
    "max_retries": 1,
    "max_columns_per_call": grouping.MAX_COLUMNS_PER_CALL,
    # Execution
    "fetch_workers": 6,
    "extract_workers": 2,
    "batch_size": 200,
    "miss_retry_days": 7.0,
    # NCBI
    "ncbi_email": "",
    "ncbi_api_key": "",
    # Output
    "include_identifiers": True,
    "allow_network": True,
}

# Environment variable -> setting. Secrets belong here or in .env, never in
# config.yaml.
ENV_SETTINGS: Dict[str, str] = {
    "PAPER_CACHE_DIR": "cache_dir",
    "PAPER_RUNS_DIR": "runs_dir",
    "NCBI_EMAIL": "ncbi_email",
    "NCBI_API_KEY": "ncbi_api_key",
    "LLM_BACKEND": "backend",
    "LLM_MODEL": "model",
    "OLLAMA_HOST": "host",
    "EMBEDDING_PROVIDER": "embedding_provider",
    "EMBEDDING_MODEL": "embedding_model",
    "EMBEDDING_REVISION": "embedding_revision",
}

# Settings that must never be written to a run folder.
SECRET_SETTINGS = ("api_key", "ncbi_api_key")


def load_dotenv_file(path: Optional[Path] = None) -> None:
    """Read .env into the environment. A variable already set wins."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(path or PROJECT_DIR / ".env", override=False)


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """config.yaml as one flat dictionary. Sections only group the keys."""
    path = Path(path or CONFIG_PATH)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def merge_settings(overrides: Optional[Dict[str, Any]] = None,
                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Defaults, then config.yaml, then the environment, then overrides."""
    settings = dict(DEFAULT_SETTINGS)
    settings.update({k: v for k, v in (config or {}).items() if v is not None})
    for variable, key in ENV_SETTINGS.items():
        value = os.environ.get(variable, "").strip()
        if value:
            settings[key] = value
    settings.update({k: v for k, v in (overrides or {}).items()
                     if v is not None})
    return settings


def load_settings(overrides: Optional[Dict[str, Any]] = None,
                  config_path: Optional[Path] = None) -> Dict[str, Any]:
    """.env, config.yaml and overrides, merged."""
    load_dotenv_file()
    return merge_settings(overrides, load_config(config_path))


def stored_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Settings safe to write into a run folder."""
    return {k: v for k, v in settings.items() if k not in SECRET_SETTINGS}


def _resolve(value: Any, default: str) -> Path:
    path = Path(str(value or default)).expanduser()
    return path if path.is_absolute() else (PROJECT_DIR / path)


@dataclass
class Paths:
    cache: Path
    runs: Path

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "Paths":
        return cls(cache=_resolve(settings.get("cache_dir"), DEFAULT_CACHE_DIR),
                   runs=_resolve(settings.get("runs_dir"), DEFAULT_RUNS_DIR))

    def ensure(self) -> "Paths":
        for path in (self.cache, self.runs):
            path.mkdir(parents=True, exist_ok=True)
        return self


def make_embedder(settings: Dict[str, Any]) -> Embedder:
    return embedder_from_settings(settings)


def make_paper_cache(paths: Paths, settings: Dict[str, Any],
                     embedder: Optional[Embedder] = None) -> PaperCache:
    return PaperCache(
        paths.cache, embedder or make_embedder(settings),
        target_tokens=int(settings.get("chunk_target_tokens", 400)),
        overlap_tokens=int(settings.get("chunk_overlap_tokens", 80)),
        min_chunk_tokens=int(settings.get("chunk_min_tokens", 20)),
        miss_retry_days=float(settings.get("miss_retry_days", 7.0)),
    )


def make_llm(settings: Dict[str, Any]) -> BaseLLM:
    backend = settings.get("backend", "ollama")
    return build_llm({
        "backend": backend,
        "model": settings.get("model"),
        "temperature": settings.get("temperature", 0.0),
        "num_predict": settings.get("num_predict", 2048),
        "max_tokens": settings.get("num_predict", 2048),
        "num_ctx": settings.get("num_ctx", 8192),
        "effort": settings.get("effort", "low"),
        "api_key": settings.get("api_key") or None,
        "max_attempts": int(settings.get("max_retries", 1)) + 2,
        "ollama": {
            "host": settings.get("host", "http://localhost:11434"),
            "timeout": int(settings.get("ollama_timeout", 900)),
            "keep_alive": settings.get("keep_alive", "1h"),
        },
        "llama_cpp": {
            "model_path": settings.get("gguf_path", ""),
            "n_ctx": settings.get("num_ctx", 8192),
            "n_gpu_layers": int(settings.get("n_gpu_layers", 0)),
            "n_threads": 8,
        },
    })


def make_fetcher(paper_cache: PaperCache, settings: Dict[str, Any]) -> Fetcher:
    return Fetcher(
        paper_cache.raw, paper_cache.ids,
        email=settings.get("ncbi_email", ""),
        api_key=settings.get("ncbi_api_key", ""),
        allow_network=bool(settings.get("allow_network", True)),
    )


# Settings that decide which group a column lands in, and therefore the keys
# the run store is checkpointed against. They are pinned to the run.
GROUPING_SETTINGS = ("max_columns_per_call", "embedding_provider",
                     "embedding_model", "embedding_revision")


def plan(specs: Sequence[ColumnSpec], settings: Dict[str, Any],
         embedder: Optional[Any] = None,
         ) -> Tuple[List[ColumnGroup], List[grouping.Routing]]:
    """Group the user's columns for this run.

    Pass the paper cache's embedder so that the column vectors are cached too.
    """
    embedder = embedder or make_embedder(settings)
    return grouping.plan_groups(
        specs,
        max_per_call=int(settings.get("max_columns_per_call",
                                      grouping.MAX_COLUMNS_PER_CALL)),
        router=grouping.ColumnRouter(embedder),
    )


def settings_for_run(info: RunInfo, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Pin the grouping settings to the ones the run was created with.

    Resume matches on (paper, group key). Regrouping the same columns
    differently would rename every group, so stored work would look missing and
    be redone under new keys. Everything else - workers, model, thresholds -
    stays free to change between passes.
    """
    merged = dict(settings)
    for key in GROUPING_SETTINGS:
        stored = (info.settings or {}).get(key)
        if stored is None:
            continue
        if merged.get(key) != stored:
            log.info("Using %s=%r from run %s instead of %r, so resume still "
                     "matches.", key, stored, info.run_id, merged.get(key))
        merged[key] = stored
    return merged


def prepare_run(
    specs: Sequence[ColumnSpec],
    settings: Dict[str, Any],
    paths: Paths,
    *,
    label: str = "",
    reuse_run_id: Optional[str] = None,
) -> Tuple[RunInfo, RunStore]:
    """Create a run, or reopen one whose column set matches exactly."""
    registry = RunRegistry(paths.runs)
    if reuse_run_id:
        info = registry.load(reuse_run_id)
        if info is None:
            raise ValueError(f"No run named {reuse_run_id!r}.")
        from columns import column_set_fingerprint

        if info.fingerprint != column_set_fingerprint(list(specs)):
            raise ValueError(
                f"Run {reuse_run_id!r} was made with a different column set. "
                "Start a new run, or load that run's columns first."
            )
    else:
        info = registry.create(specs, stored_settings(settings), label=label)
    return info, RunStore(registry.run_dir(info.run_id))


def execute_run(
    refs: Sequence[PaperRef],
    info: RunInfo,
    store: RunStore,
    settings: Dict[str, Any],
    paths: Paths,
    *,
    llm: Optional[BaseLLM] = None,
    groups: Optional[Sequence[ColumnGroup]] = None,
    on_progress: Optional[runner_module.ProgressFn] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    resume: bool = True,
    embedder: Optional[Embedder] = None,
) -> Dict[str, Any]:
    """Run the extraction and return the run's totals, with cache counts."""
    paths.ensure()
    specs = list(info.columns)
    paper_cache = make_paper_cache(paths, settings, embedder)
    try:
        if groups is None:
            groups, _ = plan(specs, settings, paper_cache.embedder)
        runner = runner_module.ExtractionRunner(
            llm=llm or make_llm(settings),
            fetcher=make_fetcher(paper_cache, settings),
            paper_store=paper_cache,
            run_store=store,
            groups=groups,
            columns=specs,
            config=runner_module.RunnerConfig(
                fetch_workers=int(settings.get("fetch_workers", 6)),
                extract_workers=int(settings.get("extract_workers", 2)),
                batch_size=int(settings.get("batch_size", 200)),
                top_k=int(settings.get("top_k", 12)),
                token_budget=int(settings.get("token_budget", 2600)),
                confidence_threshold=float(settings.get(
                    "confidence_threshold",
                    validate.DEFAULT_CONFIDENCE_THRESHOLD)),
                max_retries=int(settings.get("max_retries", 1)),
                resume=resume,
            ),
        )
        stats = runner.run(refs, on_progress=on_progress,
                           should_stop=should_stop)
    finally:
        paper_cache.flush()
        cache_report = paper_cache.stats.snapshot()
        cache_report["cache_dir"] = str(paths.cache)
        cache_report["embedding_model_loaded"] = paper_cache.base_embedder.loaded
        cache_report["namespaces"] = {"chunks": paper_cache.chunk_ns,
                                      "embeddings": paper_cache.embed_ns}
        paper_cache.record_run(info.run_id)
        log.info("Run %s %s", info.run_id, paper_cache.stats.summary_line())
        paper_cache.close()
    stats["cache"] = cache_report
    store.set_meta("last_run_stats", stats)
    store.set_meta("cache_stats", cache_report)
    return stats


def export_results(info: RunInfo, store: RunStore, paths: Paths,
                   *, stats: Optional[Dict[str, Any]] = None,
                   include_identifiers: bool = True) -> Dict[str, Any]:
    run_dir = RunRegistry(paths.runs).run_dir(info.run_id)
    return export.export_run(run_dir, info, store,
                             include_identifiers=include_identifiers,
                             stats=stats)
