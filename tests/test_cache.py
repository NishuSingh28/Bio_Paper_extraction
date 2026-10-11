"""The paper cache: each paper is fetched, parsed, chunked and embedded once.

No network and no real model. FakeNCBI serves the XML, FakeEmbedder makes
the vectors and counts its calls.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import cache as cache_module
import column_pipeline
import fetch
from columns import build_column_set
from conftest import FakeEmbedder, FakeLLM, FakeNCBI, paper_links

PAPERS = 6
COLUMNS_A = [{"name": "sample size", "type": "integer"},
             {"name": "detection method"}]
COLUMNS_B = [{"name": "country of sampling"}, {"name": "funding source"},
             {"name": "title"}]
CACHED_LAYERS = ("parsed", "chunks", "bm25", "embeddings")


def run_once(tmp_path: Path, columns: List[Dict[str, Any]],
             embedder: Optional[FakeEmbedder] = None,
             links: Optional[List[str]] = None,
             **overrides: Any) -> Dict[str, Any]:
    settings = column_pipeline.merge_settings({
        "cache_dir": str(tmp_path / "cache"),
        "runs_dir": str(tmp_path / "runs"),
        "embedding_provider": "fake", "embedding_model": "fake-a",
        "fetch_workers": 3, "extract_workers": 2, "batch_size": 4,
        **overrides,
    })
    embedder = embedder or FakeEmbedder(settings["embedding_model"])
    paths = column_pipeline.Paths.from_settings(settings).ensure()
    info, store = column_pipeline.prepare_run(build_column_set(columns),
                                              settings, paths)
    refs = fetch.refs_from_lines(links or paper_links(PAPERS))
    try:
        stats = column_pipeline.execute_run(
            refs, info, store, settings, paths, llm=FakeLLM(),
            embedder=embedder)
        column_pipeline.export_results(info, store, paths, stats=stats)
    finally:
        store.close()
    stats["embedder"] = embedder
    return stats


def layer(stats: Dict[str, Any], name: str) -> Dict[str, int]:
    return stats["cache"]["layers"][name]


# --------------------------------------------------------------------------- #
def test_second_run_makes_zero_embedding_calls(tmp_path: Path,
                                               fake_ncbi: FakeNCBI) -> None:
    first = run_once(tmp_path, COLUMNS_A)
    assert first["done"] == PAPERS
    assert first["cache"]["embedding_calls"] == PAPERS
    assert first["cache"]["network_requests"] > 0
    for name in CACHED_LAYERS:
        assert layer(first, name)["misses"] == PAPERS

    requests_before = fake_ncbi.requests
    fresh_embedder = FakeEmbedder("fake-a")   # as if in a new process
    second = run_once(tmp_path, COLUMNS_A, embedder=fresh_embedder)

    assert second["done"] == PAPERS
    assert second["cache"]["embedding_calls"] == 0
    assert second["cache"]["chunks_embedded"] == 0
    assert second["cache"]["query_encode_calls"] == 0
    assert fresh_embedder.encode_calls == 0
    assert fresh_embedder.load_count == 0, "the model is never even loaded"
    assert second["cache"]["network_requests"] == 0
    assert fake_ncbi.requests == requests_before
    for name in CACHED_LAYERS:
        assert layer(second, name) == {"hits": PAPERS, "misses": 0,
                                       "hit_rate": 1.0}


def test_changing_columns_reuses_the_cache(tmp_path: Path,
                                           fake_ncbi: FakeNCBI) -> None:
    run_once(tmp_path, COLUMNS_A)
    requests_before = fake_ncbi.requests

    other = run_once(tmp_path, COLUMNS_B)

    assert other["done"] == PAPERS
    assert other["cache"]["embedding_calls"] == 0, "no chunk is re-embedded"
    assert fake_ncbi.requests == requests_before, "no paper is re-fetched"
    for name in CACHED_LAYERS:
        assert layer(other, name)["hits"] == PAPERS
        assert layer(other, name)["misses"] == 0
    # Only the new column texts are embedded, once each, not once per paper.
    assert other["cache"]["query_encode_calls"] <= 3


def test_changing_embedding_model_creates_new_namespace(
        tmp_path: Path, fake_ncbi: FakeNCBI) -> None:
    first = run_once(tmp_path, COLUMNS_A)
    old_ns = first["cache"]["namespaces"]["embeddings"]
    requests_before = fake_ncbi.requests

    second = run_once(tmp_path, COLUMNS_A, embedding_model="fake-b")
    new_ns = second["cache"]["namespaces"]["embeddings"]

    assert new_ns != old_ns
    assert layer(second, "embeddings") == {"hits": 0, "misses": PAPERS,
                                           "hit_rate": 0.0}
    assert second["cache"]["embedding_calls"] == PAPERS
    # Text layers are shared across embedding models.
    for name in ("parsed", "chunks", "bm25"):
        assert layer(second, name)["hits"] == PAPERS
    assert fake_ncbi.requests == requests_before

    root = tmp_path / "cache"
    chunk_ns = first["cache"]["namespaces"]["chunks"]
    old_dir = root / "vectors" / f"{chunk_ns}__{old_ns}"
    new_dir = root / "vectors" / f"{chunk_ns}__{new_ns}"
    assert old_dir.exists() and new_dir.exists()

    old_store = cache_module.VectorStore(old_dir, old_dir.name,
                                         cache_module.Manifest(root / "manifest.sqlite"))
    new_store = cache_module.VectorStore(new_dir, new_dir.name,
                                         cache_module.Manifest(root / "manifest.sqlite"))
    old_vectors = old_store.read_many(list(old_store.stored_keys()))
    new_vectors = new_store.read_many(list(new_store.stored_keys()))
    assert {m.shape[1] for _, m in old_vectors.values()} == {8}
    assert {m.shape[1] for _, m in new_vectors.values()} == {12}

    # The first model's vectors are still there and still valid.
    back = run_once(tmp_path, COLUMNS_A, embedding_model="fake-a")
    assert back["cache"]["embedding_calls"] == 0


def test_changing_chunk_size_creates_new_chunk_namespace(
        tmp_path: Path, fake_ncbi: FakeNCBI) -> None:
    first = run_once(tmp_path, COLUMNS_A)
    requests_before = fake_ncbi.requests
    second = run_once(tmp_path, COLUMNS_A, chunk_target_tokens=150)
    assert (second["cache"]["namespaces"]["chunks"]
            != first["cache"]["namespaces"]["chunks"])
    assert layer(second, "parsed")["hits"] == PAPERS
    assert layer(second, "chunks")["misses"] == PAPERS
    assert layer(second, "embeddings")["misses"] == PAPERS
    assert fake_ncbi.requests == requests_before


def test_changed_weights_under_the_same_name_are_refused(
        tmp_path: Path, fake_ncbi: FakeNCBI) -> None:
    run_once(tmp_path, COLUMNS_A)
    # New papers force a model load; the weights differ from the namespace's.
    with pytest.raises(cache_module.EmbeddingModelChanged):
        run_once(tmp_path, COLUMNS_A,
                 embedder=FakeEmbedder("fake-a", salt="retrained"),
                 links=paper_links(PAPERS + 2))


# --------------------------------------------------------------------------- #
# Interrupted writes
# --------------------------------------------------------------------------- #
def test_interrupted_file_write_keeps_the_old_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "entry.bin"
    cache_module.atomic_write_bytes(target, b"complete old entry")

    def crash(*args: Any, **kwargs: Any) -> None:
        raise OSError("power cut during rename")

    monkeypatch.setattr(cache_module.os, "replace", crash)
    with pytest.raises(OSError):
        cache_module.atomic_write_bytes(target, b"half-written new entry")

    assert target.read_bytes() == b"complete old entry"
    assert not list(tmp_path.glob(".*.tmp")), "no temporary file is left"


def test_crash_between_vector_commit_and_manifest(
        tmp_path: Path, fake_ncbi: FakeNCBI,
        monkeypatch: pytest.MonkeyPatch) -> None:
    original = cache_module.Manifest.transaction
    state = {"crashed": False}

    def crash_on_embeddings(self, statements):  # noqa: ANN001
        statements = list(statements)
        if not state["crashed"] and any(
                "'embeddings'" in str(p) or (len(p) > 1 and p[1] == "embeddings")
                for _, p in statements):
            state["crashed"] = True
            raise RuntimeError("killed after the LanceDB commit")
        return original(self, statements)

    monkeypatch.setattr(cache_module.Manifest, "transaction",
                        crash_on_embeddings)
    with pytest.raises(RuntimeError):
        run_once(tmp_path, COLUMNS_A)
    monkeypatch.setattr(cache_module.Manifest, "transaction", original)

    root = tmp_path / "cache"
    report = cache_module.verify_cache(root, repair=False)
    assert report.removed, "vectors with no manifest row are found"

    # The next run treats them as misses and re-embeds; no duplicates remain.
    rerun = run_once(tmp_path, COLUMNS_A)
    assert rerun["done"] == PAPERS
    assert layer(rerun, "embeddings")["misses"] > 0
    assert cache_module.verify_cache(root).ok
    final = run_once(tmp_path, COLUMNS_A)
    assert final["cache"]["embedding_calls"] == 0


def test_interrupted_run_leaves_a_valid_cache(tmp_path: Path,
                                              fake_ncbi: FakeNCBI) -> None:
    run_once(tmp_path, COLUMNS_A, links=paper_links(3))
    fake_ncbi.fail = True   # the next network request "crashes" the run
    stats = run_once(tmp_path, COLUMNS_A, links=paper_links(PAPERS))
    assert stats["interrupted"] or stats["stopped"]
    fake_ncbi.fail = False

    root = tmp_path / "cache"
    assert cache_module.verify_cache(root, repair=False).ok

    resumed = run_once(tmp_path, COLUMNS_A, links=paper_links(PAPERS))
    assert resumed["done"] == PAPERS
    assert layer(resumed, "embeddings")["hits"] >= 3
    assert cache_module.verify_cache(root, repair=False).ok


def test_verify_repairs_and_quarantines_bad_entries(
        tmp_path: Path, fake_ncbi: FakeNCBI) -> None:
    run_once(tmp_path, COLUMNS_A)
    root = tmp_path / "cache"

    chunk_file = next((root / "chunks").rglob("*.parquet"))
    chunk_file.write_bytes(chunk_file.read_bytes()[:20])        # torn write
    raw_file = next((root / "raw").rglob("*.xml.gz"))
    raw_file.write_bytes(b"not gzip at all")                     # bad sync
    (root / "parsed" / ".stray.json.gz.1234abcd.tmp").write_bytes(b"x")

    found = cache_module.verify_cache(root, repair=False)
    assert not found.ok
    assert any(chunk_file.stem in item for item in found.quarantined)

    repaired = cache_module.verify_cache(root, repair=True)
    assert any(chunk_file.stem in item for item in repaired.repaired)
    assert any(raw_file.name.split(".")[0] in item
               for item in repaired.quarantined)
    assert list((root / "quarantine").rglob(raw_file.name))
    assert cache_module.verify_cache(root, repair=False).ok

    # The repaired chunks are used again; the paper is not re-embedded.
    after = run_once(tmp_path, COLUMNS_A)
    assert after["cache"]["embedding_calls"] == 0


def test_a_torn_cache_file_is_rebuilt_not_trusted(
        tmp_path: Path, fake_ncbi: FakeNCBI) -> None:
    run_once(tmp_path, COLUMNS_A)
    chunk_file = next((tmp_path / "cache" / "chunks").rglob("*.parquet"))
    chunk_file.write_bytes(b"PAR1 torn")
    after = run_once(tmp_path, COLUMNS_A)
    assert after["done"] == PAPERS
    assert layer(after, "chunks")["misses"] == 1
    assert layer(after, "embeddings")["hits"] == PAPERS


# --------------------------------------------------------------------------- #
# Maintenance
# --------------------------------------------------------------------------- #
def test_reset_one_layer_keeps_the_layers_below(tmp_path: Path,
                                                fake_ncbi: FakeNCBI) -> None:
    run_once(tmp_path, COLUMNS_A)
    root = tmp_path / "cache"
    cache_module.reset_cache(root, ["embeddings"])
    assert not (root / "vectors").exists()
    requests_before = fake_ncbi.requests
    after = run_once(tmp_path, COLUMNS_A)
    assert layer(after, "parsed")["hits"] == PAPERS
    assert layer(after, "embeddings")["misses"] == PAPERS
    assert fake_ncbi.requests == requests_before


def test_import_raw_from_old_layout(tmp_path: Path,
                                    fake_ncbi: FakeNCBI) -> None:
    from conftest import pmc_xml

    old = tmp_path / "old_raw" / "ab"
    old.mkdir(parents=True)
    (old / "pmc_PMC1000001.xml.gz").write_bytes(
        gzip.compress(pmc_xml(1).encode()))
    (old / "pmc_PMC1000002.xml.gz").write_bytes(gzip.compress(b""))
    counts = cache_module.import_raw(tmp_path / "cache", tmp_path / "old_raw")
    assert counts["imported"] == 1 and counts["skipped_empty"] == 1

    stats = run_once(tmp_path, COLUMNS_A, links=["PMC1000001"])
    assert layer(stats, "raw")["hits"] == 1
    assert stats["cache"]["network_requests"] == 0


def test_cache_dir_is_configurable_by_environment(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER_CACHE_DIR", str(tmp_path / "synced" / "cache"))
    settings = column_pipeline.merge_settings()
    paths = column_pipeline.Paths.from_settings(settings)
    assert paths.cache == tmp_path / "synced" / "cache"


def test_run_outputs_are_separate_from_the_cache(tmp_path: Path,
                                                 fake_ncbi: FakeNCBI) -> None:
    run_once(tmp_path, COLUMNS_A)
    cache_files = {p.name for p in (tmp_path / "cache").rglob("*")}
    assert not any(name.endswith(".xlsx") for name in cache_files)
    assert any((tmp_path / "runs").rglob("extraction.xlsx"))
    assert not any((tmp_path / "runs").rglob("*.parquet"))
    assert os.path.commonpath([tmp_path / "cache", tmp_path / "runs"]) \
        == str(tmp_path)
