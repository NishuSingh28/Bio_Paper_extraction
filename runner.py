"""The batch runner: concurrent, resumable, and the same code at any scale.

Two pools, because the two stages block on different things. Fetching waits on
a remote server and must respect its rate limit; extraction waits on a model.
Running them in one pool would leave whichever is slower idle.

Work is processed in batches so memory stays flat: one batch of references is
resolved, ingested and extracted while the next batch waits on disk. A run over
100,000 papers holds no more in memory than a run over 100.

Every finished paper is written to the run store before the next one starts, so
stopping the run - by button, by Ctrl-C, or by the machine dying - costs at most
the papers in flight.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import extract_columns
import validate
from cache import PaperCache
from columns import ColumnSpec
from embedder import EmbeddingModelChanged
from fetch import Fetcher, PaperRef
from grouping import ColumnGroup
from index import PaperIndex
from llm import BaseLLM
from runstore import RunStore

log = logging.getLogger(__name__)

ProgressFn = Callable[[Dict[str, Any], Dict[str, Any]], None]


@dataclass
class RunnerConfig:
    """Everything that changes between a 100-paper run and a 100,000-paper one."""

    fetch_workers: int = 6
    extract_workers: int = 2
    batch_size: int = 200
    top_k: int = extract_columns.DEFAULT_TOP_K
    token_budget: int = extract_columns.DEFAULT_TOKEN_BUDGET
    confidence_threshold: float = validate.DEFAULT_CONFIDENCE_THRESHOLD
    max_retries: int = 1
    resume: bool = True


@dataclass
class RunProgress:
    """Live totals. Read by the UI while the run is going."""

    total: int = 0
    done: int = 0
    skipped: int = 0
    failed: int = 0
    values_flagged: int = 0
    values_total: int = 0
    papers_flagged: int = 0
    full_text: int = 0
    abstract_only: int = 0
    no_text: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    paper_seconds: float = 0.0
    started_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, paper: Dict[str, Any]) -> None:
        with self._lock:
            self.done += 1
            self.values_total += len(paper.get("values", {}))
            flagged = int(paper.get("flagged_values", 0))
            self.values_flagged += flagged
            if flagged:
                self.papers_flagged += 1
            self.llm_calls += int(paper.get("llm_calls", 0))
            self.input_tokens += int(paper.get("input_tokens", 0))
            self.output_tokens += int(paper.get("output_tokens", 0))
            self.cost_usd += float(paper.get("cost_usd", 0.0))
            self.paper_seconds += float(paper.get("elapsed_s", 0.0))
            source = paper.get("source_type", "")
            if source == "full_text":
                self.full_text += 1
            elif source == "abstract_only":
                self.abstract_only += 1
            else:
                self.no_text += 1
            if paper.get("errors"):
                self.failed += 1

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    @property
    def seconds_per_paper(self) -> float:
        return self.elapsed_s / max(1, self.done)

    @property
    def processing_seconds_per_paper(self) -> float:
        """Time actually spent on a paper, summed over workers.

        Wall-clock time divided by papers is wrong twice over: it shrinks with
        concurrency and it grows if the machine sleeps mid-run. This number
        does neither, so it is the one to size a batch with.
        """
        return self.paper_seconds / max(1, self.done)

    @property
    def cost_per_paper(self) -> float:
        return self.cost_usd / max(1, self.done)

    @property
    def eta_s(self) -> float:
        remaining = max(0, self.total - self.done - self.skipped)
        return remaining * self.seconds_per_paper

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total": self.total, "done": self.done, "skipped": self.skipped,
                "failed": self.failed, "values_total": self.values_total,
                "values_flagged": self.values_flagged,
                "papers_flagged": self.papers_flagged,
                "full_text": self.full_text,
                "abstract_only": self.abstract_only, "no_text": self.no_text,
                "llm_calls": self.llm_calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cost_usd": round(self.cost_usd, 4),
                "elapsed_s": round(self.elapsed_s, 1),
                "seconds_per_paper": round(self.seconds_per_paper, 2),
                "processing_seconds_per_paper": round(
                    self.processing_seconds_per_paper, 2),
                "cost_per_paper": round(self.cost_per_paper, 5),
                "eta_s": round(self.eta_s, 0),
            }


def _batches(items: Sequence[Any], size: int) -> List[Sequence[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


class ExtractionRunner:
    """Runs one column set over one list of papers."""

    def __init__(
        self,
        *,
        llm: BaseLLM,
        fetcher: Fetcher,
        paper_store: PaperCache,
        run_store: RunStore,
        groups: Sequence[ColumnGroup],
        columns: Sequence[ColumnSpec],
        config: Optional[RunnerConfig] = None,
    ):
        self.llm = llm
        self.fetcher = fetcher
        self.paper_store = paper_store
        self.run_store = run_store
        self.groups = list(groups)
        self.columns = list(columns)
        self.config = config or RunnerConfig()
        self.group_keys = [g.key for g in self.groups]
        self.progress = RunProgress()

    # ----------------------------------------------------------------- #
    def _ingest(self, ref: PaperRef) -> Optional[PaperIndex]:
        """Fetch and index one paper. Returns None only on an unexpected error."""
        try:
            return self.paper_store.index_for(ref, self.fetcher)
        except EmbeddingModelChanged:
            raise  # every later paper would fail the same way; stop the run
        except Exception as exc:  # noqa: BLE001 - one bad paper must not stop a run
            log.warning("Ingest failed for %s: %s", ref.label, exc)
            return None

    def _extract(self, ref: PaperRef, paper_index: PaperIndex,
                 skip_groups: Sequence[str]) -> Dict[str, Any]:
        """Extract one paper, skipping groups a previous run already finished."""
        document = paper_index.document
        pending = [g for g in self.groups if g.key not in set(skip_groups)]

        record = extract_columns.extract_paper(
            self.llm, document, paper_index, pending,
            top_k=self.config.top_k,
            token_budget=self.config.token_budget,
            confidence_threshold=self.config.confidence_threshold,
            max_retries=self.config.max_retries,
        )

        for stats in record["calls"]:
            group_key = stats["group"]
            self.run_store.save_group(
                ref.key, group_key,
                record["values_by_group"].get(group_key, {}), stats)

        if skip_groups:
            # Merge in what an earlier run already produced, so the record the
            # caller sees is the whole paper rather than this run's part of it.
            stored = self.run_store.load_paper(ref.key) or {}
            merged = dict(stored.get("values", {}))
            merged.update(record["values"])
            record["values"] = merged
            record["flagged_values"] = sum(
                1 for v in merged.values() if v.get("needs_review"))

        meta = {k: v for k, v in record.items()
                if k not in ("values", "values_by_group", "calls")}
        self.run_store.save_paper(
            ref.key, meta, status="error" if record.get("errors") else "ok")
        return record

    def _failed_record(self, ref: PaperRef, reason: str) -> Dict[str, Any]:
        """A paper that could not be ingested still gets a full row.

        Skipping it would break the promise that every submitted paper produces
        a value for every column.
        """
        values = {
            spec.key: validate.missing_value(spec, reason)
            for spec in self.columns
        }
        record = {
            "paper_key": ref.key, "article_id": ref.extra.get("article_id", "")
            or ref.label, "ids": ref.ids, "title": ref.title, "year": "",
            "journal": "", "source_type": "no_text", "source_api": "",
            "fetch_notes": reason, "chunk_count": 0, "values": values,
            "calls": [], "flagged_values": len(values), "llm_calls": 0,
            "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
            "elapsed_s": 0.0, "errors": [reason],
        }
        meta = {k: v for k, v in record.items()
                if k not in ("values", "values_by_group", "calls")}
        self.run_store.save_group(ref.key, "__failed__", values,
                                  {"group": "__failed__", "error": reason,
                                   "llm_call": False, "elapsed_s": 0.0,
                                   "input_tokens": 0, "output_tokens": 0,
                                   "cached_input_tokens": 0, "cost_usd": 0.0,
                                   "attempts": 0, "chunks": []})
        self.run_store.save_paper(ref.key, meta, status="error")
        return record

    # ----------------------------------------------------------------- #
    def run(
        self,
        refs: Sequence[PaperRef],
        *,
        on_progress: Optional[ProgressFn] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Process every reference. Returns the run's totals."""
        refs = list(refs)
        self.progress = RunProgress(total=len(refs))

        done_map: Dict[str, set] = (
            self.run_store.done_groups() if self.config.resume else {})
        wanted = set(self.group_keys)
        stopped = False

        pending: List[PaperRef] = []
        for ref in refs:
            done = done_map.get(ref.key, set())
            if wanted and wanted.issubset(done):
                self.progress.skipped += 1
                continue
            pending.append(ref)

        log.info("Run: %d papers, %d already complete, %d to process",
                 len(refs), self.progress.skipped, len(pending))

        def wants_stop() -> bool:
            return should_stop is not None and should_stop()

        # Downloading is far faster than extracting, so without a cap every
        # paper in the batch would be queued on the extract pool within a
        # minute. A pool that is shut down waits for everything queued, which
        # would make Stop and Ctrl-C wait for the whole run. This keeps the
        # queue short, so stopping takes effect in about one paper's time.
        slots = threading.Semaphore(max(2, self.config.extract_workers * 2))

        def extract_and_release(ref: PaperRef, paper_index: PaperIndex,
                                skip: Sequence[str]) -> Dict[str, Any]:
            try:
                return self._extract(ref, paper_index, skip)
            finally:
                slots.release()

        fetch_pool = ThreadPoolExecutor(max_workers=self.config.fetch_workers,
                                        thread_name_prefix="fetch")
        extract_pool = ThreadPoolExecutor(
            max_workers=self.config.extract_workers,
            thread_name_prefix="extract")
        interrupted = False

        def collect(future: Future, ref: PaperRef) -> None:
            try:
                record = future.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("Extraction failed for %s: %s", ref.label, exc)
                record = self._failed_record(ref, f"extraction failed: {exc}")
            self.progress.record(record)
            if on_progress is not None:
                on_progress(record, self.progress.snapshot())

        try:
            for batch in _batches(pending, self.config.batch_size):
                if stopped:
                    break
                # One converter call for the whole batch instead of one per
                # paper. At 100,000 papers that is the difference between
                # minutes and hours. Papers already in the cache need none.
                try:
                    unknown = self.paper_store.unknown(batch)
                    if unknown:
                        self.fetcher.resolve_batch(unknown)
                    # One vector read for the whole batch, not one per paper.
                    self.paper_store.prefetch(batch)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Batch preparation failed: %s", exc)

                ingest_futures: Dict[Future, PaperRef] = {
                    fetch_pool.submit(self._ingest, ref): ref for ref in batch
                }
                extract_futures: Dict[Future, PaperRef] = {}

                for future in as_completed(ingest_futures):
                    ref = ingest_futures[future]
                    if wants_stop():
                        stopped = True
                        break
                    paper_index = future.result()
                    if paper_index is None:
                        record = self._failed_record(
                            ref, "the paper could not be fetched")
                        self.progress.record(record)
                        if on_progress is not None:
                            on_progress(record, self.progress.snapshot())
                        continue

                    # Wait for room on the extract queue, checking for a stop
                    # while waiting.
                    while not slots.acquire(timeout=0.5):
                        if wants_stop():
                            stopped = True
                            break
                    if stopped:
                        break

                    skip = sorted(done_map.get(ref.key, set()) & wanted)
                    extract_futures[extract_pool.submit(
                        extract_and_release, ref, paper_index, skip)] = ref

                    # Record whatever has finished, so progress and the Stop
                    # button stay current while the batch is still feeding.
                    for finished in [f for f in extract_futures if f.done()]:
                        collect(finished, extract_futures.pop(finished))

                if stopped:
                    # Abandon what is still queued and collect only what has
                    # already finished. Every completed paper is in the run
                    # store, so nothing is lost by walking away from the queue.
                    for future, queued in extract_futures.items():
                        if future.done():
                            collect(future, queued)
                    break

                for future in as_completed(extract_futures):
                    collect(future, extract_futures[future])
                    if wants_stop():
                        stopped = True
                # Commit this batch's new vectors, so a crash in a later
                # batch does not cost this batch's embedding work.
                self.paper_store.flush()
                if stopped:
                    break
        except KeyboardInterrupt:
            stopped = interrupted = True
            log.warning("Interrupted.")
        finally:
            # cancel_futures drops what has not started. Work in flight still
            # finishes and is saved, so a stop costs at most the papers running.
            fetch_pool.shutdown(wait=not stopped, cancel_futures=stopped)
            extract_pool.shutdown(wait=not stopped, cancel_futures=stopped)

        if stopped:
            log.warning("Stopped with %d papers saved. Rerun to continue where "
                        "this left off.",
                        self.progress.done + self.progress.skipped)

        snapshot = self.progress.snapshot()
        snapshot["stopped"] = stopped
        snapshot["interrupted"] = interrupted
        snapshot["papers_in_store"] = self.run_store.paper_count()
        self.run_store.set_meta("last_run_stats", snapshot)
        return snapshot
