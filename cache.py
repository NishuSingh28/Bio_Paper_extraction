"""The persistent local paper cache: fetch, parse, chunk and embed each paper once.

The cache is per paper, not per run. A change to the columns never causes a new
download, a new parse, a new chunking or a new embedding.

Layers, each checked before any work is done::

    raw         original XML from PMC / PubMed / Europe PMC, per source id
    parsed      sections ("blocks"), metadata and source_type, per paper
    chunks      Parquet, one file per paper, per chunker namespace
    bm25        BM25 statistics, per paper, per chunker namespace
    embeddings  LanceDB table, one per (chunker namespace, embedding namespace)

A chunk or vector entry is valid only when its stored content hash equals the
hash of the paper's parsed text. The chunker namespace holds the chunker version
and parameters; the embedding namespace holds the provider, model and revision.
A change to any of them writes to a new namespace. Vectors from two models are
never mixed.

Every file is written to a temporary name, flushed, and renamed. The manifest
row is committed after the rename. An interrupted run therefore leaves either
a complete entry or a cache miss, never a corrupt hit. `verify` finds what a
crash or a sync tool damaged, and repairs or quarantines it.

Layout::

    <cache_dir>/
      manifest.sqlite
      raw/<shard>/<source>_<id>.xml.gz
      parsed/<shard>/<paper_key>.json.gz
      chunks/<chunk_ns>/<shard>/<paper_key>.parquet
      bm25/<chunk_ns>/<shard>/<paper_key>.json.gz
      vectors/<chunk_ns>__<embed_ns>/vectors.lance
      quarantine/<timestamp>/...
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import chunking
from embedder import Embedder, EmbeddingModelChanged
from index import BM25, PaperIndex

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
LAYERS = ("raw", "parsed", "chunks", "bm25", "embeddings")
# The counters a run reports. id_map and query_vectors are lookups that would
# otherwise cost a network call or a model call.
STAT_LAYERS = ("id_map", "raw", "parsed", "chunks", "bm25", "embeddings",
               "query_vectors")
PARSED_NAMESPACE = "parsed-v1"
DEFAULT_MISS_RETRY_DAYS = 7.0
VECTOR_FLUSH_PAPERS = 32

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS papers (
    paper_key    TEXT PRIMARY KEY,
    ids_json     TEXT NOT NULL,
    title        TEXT,
    source_type  TEXT,
    content_hash TEXT,
    first_seen   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS aliases (
    ref_key    TEXT PRIMARY KEY,
    paper_key  TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS id_map (
    source_id   TEXT PRIMARY KEY,
    pmcid       TEXT,
    pmid        TEXT,
    doi         TEXT,
    resolved_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS raw_files (
    name       TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    path       TEXT,
    checksum   TEXT,
    bytes      INTEGER,
    error      TEXT,
    fetched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS layers (
    paper_key    TEXT NOT NULL,
    layer        TEXT NOT NULL,
    namespace    TEXT NOT NULL,
    status       TEXT NOT NULL,
    content_hash TEXT,
    checksum     TEXT,
    n_items      INTEGER,
    bytes        INTEGER,
    path         TEXT,
    error        TEXT,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (paper_key, layer, namespace)
);
CREATE INDEX IF NOT EXISTS layers_by_namespace ON layers (layer, namespace);
CREATE TABLE IF NOT EXISTS namespaces (
    namespace   TEXT PRIMARY KEY,
    layer       TEXT NOT NULL,
    params_json TEXT NOT NULL,
    fingerprint TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS query_vectors (
    namespace  TEXT NOT NULL,
    text_hash  TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (namespace, text_hash)
);
CREATE TABLE IF NOT EXISTS run_cache_stats (
    run_id     TEXT NOT NULL,
    layer      TEXT NOT NULL,
    hits       INTEGER NOT NULL,
    misses     INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (run_id, layer)
);
"""


# --------------------------------------------------------------------------- #
# Atomic file writes
# --------------------------------------------------------------------------- #
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(directory: Path) -> None:
    # A rename is durable only once the directory entry is on disk. Some
    # platforms cannot open a directory; the rename is still atomic there.
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> str:
    """Write `data` to `path` so a reader sees the old file or the new one.

    Returns the SHA-256 of the data. The temporary file is in the same
    directory, so the rename never crosses a file system.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    _fsync_dir(path.parent)
    return sha256_bytes(data)


def _gzip_json(obj: Any) -> bytes:
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    # mtime=0 makes the bytes, and so the checksum, reproducible.
    return gzip.compress(raw, compresslevel=6, mtime=0)


def _read_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _shard(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:2]


CHUNK_SCHEMA = pa.schema([
    ("id", pa.string()), ("chunk_id", pa.string()), ("section", pa.string()),
    ("title", pa.string()), ("kind", pa.string()), ("text", pa.string()),
    ("tokens", pa.int32()), ("block", pa.int32()),
    ("char_start", pa.int32()), ("char_end", pa.int32()),
])


def _chunks_parquet(chunks: List[Dict[str, Any]]) -> bytes:
    """One paper's chunks as Parquet bytes, ready for an atomic write."""
    table = pa.Table.from_pylist(
        [{name: c.get(name) for name in CHUNK_SCHEMA.names} for c in chunks],
        schema=CHUNK_SCHEMA)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Hit / miss counters
# --------------------------------------------------------------------------- #
@dataclass
class CacheStats:
    """Hit and miss counts for one run, per layer."""

    hits: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in STAT_LAYERS})
    misses: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in STAT_LAYERS})
    embedding_calls: int = 0      # model calls that embedded paper chunks
    chunks_embedded: int = 0
    query_encode_calls: int = 0   # model calls that embedded queries or columns
    network_requests: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def hit(self, layer: str, n: int = 1) -> None:
        with self._lock:
            self.hits[layer] = self.hits.get(layer, 0) + n

    def miss(self, layer: str, n: int = 1) -> None:
        with self._lock:
            self.misses[layer] = self.misses.get(layer, 0) + n

    def add(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            layers = {}
            for layer in STAT_LAYERS:
                hits, misses = self.hits.get(layer, 0), self.misses.get(layer, 0)
                total = hits + misses
                layers[layer] = {
                    "hits": hits, "misses": misses,
                    "hit_rate": round(hits / total, 4) if total else None,
                }
            return {
                "layers": layers,
                "embedding_calls": self.embedding_calls,
                "chunks_embedded": self.chunks_embedded,
                "query_encode_calls": self.query_encode_calls,
                "network_requests": self.network_requests,
            }

    def summary_line(self) -> str:
        snap = self.snapshot()
        parts = [f"{layer} {v['hits']}/{v['hits'] + v['misses']}"
                 for layer, v in snap["layers"].items()
                 if v["hits"] + v["misses"]]
        return (f"cache hits: {', '.join(parts) or 'none'}; "
                f"embedding calls: {snap['embedding_calls']} "
                f"({snap['chunks_embedded']} chunks); query encode calls: "
                f"{snap['query_encode_calls']}; network requests: "
                f"{snap['network_requests']}")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
class Manifest:
    """SQLite index of every cache entry: status, versions, times, checksums."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False,
                                     timeout=60.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.executescript(_SCHEMA)
            stored = self.get_meta("schema_version")
            if stored is None:
                self.set_meta("schema_version", str(SCHEMA_VERSION))
            elif int(stored) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Cache manifest {self.path} has schema {stored}; this "
                    f"version of the app needs {SCHEMA_VERSION}. Use a new "
                    "cache directory or run `python cli.py cache reset`.")

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def transaction(self, statements: Iterable[Tuple[str, Sequence[Any]]]) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for sql, params in statements:
                    self._conn.execute(sql, params)
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    # ----------------------------------------------------------------- #
    def get_meta(self, key: str) -> Optional[str]:
        row = self.one("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                     (key, value))

    # ----------------------------------------------------------------- #
    def layer(self, paper_key: str, layer: str,
              namespace: str) -> Optional[sqlite3.Row]:
        return self.one(
            "SELECT * FROM layers WHERE paper_key = ? AND layer = ? "
            "AND namespace = ?", (paper_key, layer, namespace))

    @staticmethod
    def layer_upsert(paper_key: str, layer: str, namespace: str, *,
                     status: str = "ok", content_hash: str = "",
                     checksum: str = "", n_items: int = 0, size: int = 0,
                     path: str = "", error: str = "",
                     ) -> Tuple[str, Sequence[Any]]:
        return (
            "INSERT OR REPLACE INTO layers (paper_key, layer, namespace, "
            "status, content_hash, checksum, n_items, bytes, path, error, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (paper_key, layer, namespace, status, content_hash, checksum,
             n_items, size, path, error, time.time()))

    def ensure_namespace(self, namespace: str, layer: str,
                         params: Dict[str, Any]) -> None:
        self.execute(
            "INSERT OR IGNORE INTO namespaces (namespace, layer, params_json, "
            "fingerprint, created_at) VALUES (?, ?, ?, NULL, ?)",
            (namespace, layer, json.dumps(params, sort_keys=True), time.time()))

    def check_fingerprint(self, namespace: str, fingerprint: str) -> None:
        """Record the model fingerprint, or refuse a model that changed."""
        if not fingerprint:
            return
        row = self.one("SELECT fingerprint FROM namespaces WHERE namespace = ?",
                       (namespace,))
        if row is None:
            return
        if row["fingerprint"] in (None, ""):
            self.execute("UPDATE namespaces SET fingerprint = ? "
                         "WHERE namespace = ?", (fingerprint, namespace))
            return
        if row["fingerprint"] != fingerprint:
            raise EmbeddingModelChanged(
                f"The embedding model loaded for namespace {namespace!r} is "
                "not the model that wrote it (the weights changed). Set "
                "`embedding_revision` in config.yaml to a new value, so the "
                "new vectors go to a new namespace.")


# --------------------------------------------------------------------------- #
# Raw layer and identifier map, used by fetch.Fetcher
# --------------------------------------------------------------------------- #
class RawStore:
    """Original downloads, keyed by source name such as `pmc_PMC123456`.

    `get` returns the text on a hit, "" for a recent known miss (the source
    has no such record; do not ask again yet), and None when the caller must
    download. A miss is asked again after `miss_retry_days`, because a paper
    can become open access later.
    """

    def __init__(self, root: Path, manifest: Manifest,
                 stats: Optional[CacheStats] = None,
                 miss_retry_days: float = DEFAULT_MISS_RETRY_DAYS):
        self.root = Path(root)
        self.manifest = manifest
        self.stats = stats or CacheStats()
        self.miss_retry_s = float(miss_retry_days) * 86400.0

    def path_for(self, name: str) -> Path:
        return self.root / _shard(name) / f"{name}.xml.gz"

    def get(self, name: str) -> Optional[str]:
        row = self.manifest.one("SELECT * FROM raw_files WHERE name = ?", (name,))
        if row is not None and row["status"] == "missing":
            if time.time() - float(row["fetched_at"]) < self.miss_retry_s:
                self.stats.hit("raw")
                return ""
            self.stats.miss("raw")
            return None
        path = self.path_for(name)
        if row is None and not path.exists():
            self.stats.miss("raw")
            return None
        try:
            data = path.read_bytes()
            text = gzip.decompress(data).decode("utf-8")
        except (OSError, EOFError, UnicodeDecodeError, gzip.BadGzipFile) as exc:
            log.warning("Unreadable raw cache entry %s (%s); downloading again.",
                        path, exc)
            self.stats.miss("raw")
            return None
        if row is None:
            # A file from an older cache layout. Register it.
            self._record_ok(name, path, sha256_bytes(data), len(data))
        self.stats.hit("raw")
        return text

    def put(self, name: str, text: str) -> None:
        data = gzip.compress(text.encode("utf-8"), compresslevel=6, mtime=0)
        path = self.path_for(name)
        checksum = atomic_write_bytes(path, data)
        self._record_ok(name, path, checksum, len(data))

    def put_miss(self, name: str, error: str = "") -> None:
        self.manifest.execute(
            "INSERT OR REPLACE INTO raw_files (name, status, path, checksum, "
            "bytes, error, fetched_at) VALUES (?, 'missing', '', '', 0, ?, ?)",
            (name, error, time.time()))

    def _record_ok(self, name: str, path: Path, checksum: str, size: int) -> None:
        self.manifest.execute(
            "INSERT OR REPLACE INTO raw_files (name, status, path, checksum, "
            "bytes, error, fetched_at) VALUES (?, 'ok', ?, ?, ?, '', ?)",
            (name, str(path.relative_to(self.root.parent)), checksum, size,
             time.time()))


class IdMapStore:
    """Persisted PMID/DOI -> PMCID resolution, so a rerun asks NCBI nothing."""

    def __init__(self, manifest: Manifest, stats: Optional[CacheStats] = None,
                 miss_retry_days: float = DEFAULT_MISS_RETRY_DAYS):
        self.manifest = manifest
        self.stats = stats or CacheStats()
        self.miss_retry_s = float(miss_retry_days) * 86400.0

    def get(self, source_id: str) -> Optional[Dict[str, str]]:
        """The stored mapping, {} for a recent "no PMC record", or None."""
        row = self.manifest.one("SELECT * FROM id_map WHERE source_id = ?",
                                (source_id,))
        if row is None:
            self.stats.miss("id_map")
            return None
        mapping = {k: row[k] for k in ("pmcid", "pmid", "doi") if row[k]}
        if (not mapping.get("pmcid")
                and time.time() - float(row["resolved_at"]) > self.miss_retry_s):
            self.stats.miss("id_map")
            return None
        self.stats.hit("id_map")
        return mapping

    def put(self, source_id: str, mapping: Dict[str, str]) -> None:
        self.manifest.execute(
            "INSERT OR REPLACE INTO id_map (source_id, pmcid, pmid, doi, "
            "resolved_at) VALUES (?, ?, ?, ?, ?)",
            (source_id, mapping.get("pmcid", ""), mapping.get("pmid", ""),
             mapping.get("doi", ""), time.time()))


# --------------------------------------------------------------------------- #
# Vector layer
# --------------------------------------------------------------------------- #
class VectorStore:
    """Chunk vectors in one LanceDB table per namespace pair.

    Rows are (chunk_id, paper_key, content_hash, row, vector). Writes are
    buffered and committed in batches, because each LanceDB commit makes a new
    data fragment. The manifest marks a paper's vectors `ok` only after the
    commit, so a crash before the commit leaves a cache miss, not a bad hit.
    """

    TABLE = "vectors"

    def __init__(self, root: Path, namespace: str, manifest: Manifest,
                 flush_every: int = VECTOR_FLUSH_PAPERS):
        self.root = Path(root)
        self.namespace = namespace
        self.manifest = manifest
        self.flush_every = max(1, int(flush_every))
        self._lock = threading.RLock()
        self._db = None
        self._table = None
        self._pending: Dict[str, Tuple[str, List[str], np.ndarray]] = {}
        self._prefetched: Dict[str, Tuple[List[str], np.ndarray]] = {}

    # ----------------------------------------------------------------- #
    def _connect(self):
        if self._db is None:
            import lancedb

            self.root.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.root))
        return self._db

    def _open(self, create_dim: int = 0):
        if self._table is not None:
            return self._table
        db = self._connect()
        try:
            self._table = db.open_table(self.TABLE)
        except (FileNotFoundError, ValueError):
            if not create_dim:
                return None
            schema = pa.schema([
                ("chunk_id", pa.string()), ("paper_key", pa.string()),
                ("content_hash", pa.string()), ("row", pa.int32()),
                ("vector", pa.list_(pa.float32(), int(create_dim))),
            ])
            self._table = db.create_table(self.TABLE, schema=schema)
        return self._table

    @staticmethod
    def _quote(values: Iterable[str]) -> str:
        return ", ".join("'" + v.replace("'", "''") + "'" for v in values)

    # ----------------------------------------------------------------- #
    def prefetch(self, paper_keys: Sequence[str]) -> None:
        """Read the vectors of many papers in one query."""
        keys = [k for k in dict.fromkeys(paper_keys)
                if k not in self._prefetched and k not in self._pending]
        if not keys:
            return
        with self._lock:
            table = self._open()
            if table is None:
                return
            for start in range(0, len(keys), 500):
                part = keys[start:start + 500]
                found = (table.search()
                         .where(f"paper_key IN ({self._quote(part)})")
                         .select(["chunk_id", "paper_key", "row", "vector"])
                         .limit(None).to_arrow())
                self._prefetched.update(self._group(found))

    @staticmethod
    def _group(found: pa.Table) -> Dict[str, Tuple[List[str], np.ndarray]]:
        if found.num_rows == 0:
            return {}
        keys = found.column("paper_key").to_pylist()
        rows = found.column("row").to_pylist()
        chunk_ids = found.column("chunk_id").to_pylist()
        vectors = found.column("vector")
        dim = vectors.type.list_size
        matrix = (vectors.combine_chunks().flatten().to_numpy(zero_copy_only=False)
                  .astype(np.float32).reshape(-1, dim))
        grouped: Dict[str, List[Tuple[int, str, int]]] = {}
        for i, (key, row, chunk_id) in enumerate(zip(keys, rows, chunk_ids)):
            grouped.setdefault(key, []).append((row, chunk_id, i))
        out: Dict[str, Tuple[List[str], np.ndarray]] = {}
        for key, items in grouped.items():
            items.sort()
            out[key] = ([c for _, c, _ in items],
                        matrix[[i for _, _, i in items]])
        return out

    def get(self, paper_key: str) -> Optional[Tuple[List[str], np.ndarray]]:
        with self._lock:
            if paper_key in self._pending:
                _, chunk_ids, matrix = self._pending[paper_key]
                return chunk_ids, matrix
            if paper_key not in self._prefetched:
                table = self._open()
                if table is None:
                    return None
                found = (table.search()
                         .where(f"paper_key = {self._quote([paper_key])}")
                         .select(["chunk_id", "paper_key", "row", "vector"])
                         .limit(None).to_arrow())
                self._prefetched.update(self._group(found))
            return self._prefetched.pop(paper_key, None)

    def put(self, paper_key: str, content_hash: str, chunk_ids: List[str],
            matrix: np.ndarray) -> None:
        with self._lock:
            self._pending[paper_key] = (content_hash, list(chunk_ids),
                                        np.asarray(matrix, dtype=np.float32))
            if len(self._pending) >= self.flush_every:
                self.flush()

    def flush(self) -> int:
        """Commit buffered vectors, then mark them `ok` in the manifest."""
        with self._lock:
            if not self._pending:
                return 0
            pending, self._pending = self._pending, {}
            dim = next(iter(pending.values()))[2].shape[1]
            table = self._open(create_dim=dim)
            table_dim = table.schema.field("vector").type.list_size
            if table_dim != dim:
                raise EmbeddingModelChanged(
                    f"Vector namespace {self.namespace!r} holds {table_dim}-"
                    f"dimensional vectors, but the model gives {dim}. Set "
                    "`embedding_revision` to a new value.")
            keys = list(pending)
            # Old rows for these papers are stale (a crash after a commit and
            # before the manifest row, or changed text). Remove them first.
            table.delete(f"paper_key IN ({self._quote(keys)})")
            chunk_ids: List[str] = []
            paper_keys: List[str] = []
            hashes: List[str] = []
            rows: List[int] = []
            blocks: List[np.ndarray] = []
            for key, (digest, ids, matrix) in pending.items():
                chunk_ids.extend(ids)
                paper_keys.extend([key] * len(ids))
                hashes.extend([digest] * len(ids))
                rows.extend(range(len(ids)))
                blocks.append(matrix)
            flat = np.concatenate(blocks).astype(np.float32)
            batch = pa.table({
                "chunk_id": chunk_ids, "paper_key": paper_keys,
                "content_hash": hashes,
                "row": pa.array(rows, type=pa.int32()),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(flat.ravel()), dim),
            }, schema=table.schema)
            table.add(batch)
            statements = []
            for key, (digest, ids, matrix) in pending.items():
                statements.append(Manifest.layer_upsert(
                    key, "embeddings", self.namespace, content_hash=digest,
                    checksum=sha256_bytes(matrix.astype(np.float32).tobytes()),
                    n_items=len(ids), size=int(matrix.nbytes),
                    path=str(self.root.name)))
            self.manifest.transaction(statements)
            return len(pending)

    def clear_prefetched(self) -> None:
        """Drop prefetched vectors that no paper asked for."""
        with self._lock:
            self._prefetched.clear()

    def delete(self, paper_keys: Sequence[str]) -> None:
        with self._lock:
            for key in paper_keys:
                self._pending.pop(key, None)
                self._prefetched.pop(key, None)
            table = self._open()
            if table is not None and paper_keys:
                table.delete(f"paper_key IN ({self._quote(paper_keys)})")

    def stored_keys(self) -> set:
        """Every paper_key that has rows in the table. Reads one column."""
        with self._lock:
            table = self._open()
            if table is None:
                return set()
            found = table.search().select(["paper_key"]).limit(None).to_arrow()
            return set(found.column("paper_key").to_pylist())

    def read_many(self, paper_keys: Sequence[str]
                  ) -> Dict[str, Tuple[List[str], np.ndarray]]:
        """Vectors for the given papers, read in batches, not buffered."""
        out: Dict[str, Tuple[List[str], np.ndarray]] = {}
        with self._lock:
            table = self._open()
            if table is None:
                return out
            keys = list(paper_keys)
            for start in range(0, len(keys), 500):
                part = keys[start:start + 500]
                found = (table.search()
                         .where(f"paper_key IN ({self._quote(part)})")
                         .select(["chunk_id", "paper_key", "row", "vector"])
                         .limit(None).to_arrow())
                out.update(self._group(found))
        return out

    def compact(self) -> None:
        with self._lock:
            self.flush()
            table = self._open()
            if table is None:
                return
            try:
                from lancedb.index import BTree

                table.create_index("paper_key", config=BTree(), replace=True)
            except Exception as exc:  # noqa: BLE001 - index is an optimisation
                log.warning("Could not index %s: %s", self.namespace, exc)
            table.optimize()


# --------------------------------------------------------------------------- #
# Query vectors
# --------------------------------------------------------------------------- #
class CachedQueryEmbedder:
    """Embeds short texts (queries, column descriptions) through a cache.

    The same column is searched in every paper, so its vector is computed once
    and then read from memory or from the manifest. A rerun with the same
    columns therefore loads no model at all.
    """

    def __init__(self, base: Embedder, manifest: Manifest, stats: CacheStats,
                 before_model_call: Callable[[], None]):
        self.base = base
        self.manifest = manifest
        self.stats = stats
        self._before_model_call = before_model_call
        self._memory: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    # The attributes callers read on an embedder.
    @property
    def model_name(self) -> str:
        return self.base.model_name

    @property
    def namespace(self) -> str:
        return self.base.namespace

    @property
    def available(self) -> bool:
        return self.base.available

    @property
    def error(self) -> Optional[str]:
        return self.base.error

    @property
    def dim(self) -> int:
        return self.base.dim

    def describe(self) -> str:
        return self.base.describe()

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        items = [t if t else " " for t in texts]
        if not items:
            return np.zeros((0, max(self.base.dim, 1)), dtype=np.float32)
        hashes = [self._hash(t) for t in items]
        found: Dict[str, np.ndarray] = {}
        with self._lock:
            for digest in hashes:
                if digest in self._memory:
                    found[digest] = self._memory[digest]
        wanted = [h for h in dict.fromkeys(hashes) if h not in found]
        if wanted:
            placeholders = ",".join("?" * len(wanted))
            for row in self.manifest.query(
                    f"SELECT text_hash, dim, vector FROM query_vectors WHERE "
                    f"namespace = ? AND text_hash IN ({placeholders})",
                    (self.namespace, *wanted)):
                found[row["text_hash"]] = np.frombuffer(
                    row["vector"], dtype=np.float32).reshape(int(row["dim"]))
        missing = [h for h in dict.fromkeys(hashes) if h not in found]
        self.stats.hit("query_vectors", len(set(hashes)) - len(missing))
        if missing:
            self.stats.miss("query_vectors", len(missing))
            texts_by_hash = {h: t for h, t in zip(hashes, items)}
            self._before_model_call()
            vectors = self.base.encode([texts_by_hash[h] for h in missing],
                                       batch_size)
            self.stats.add("query_encode_calls")
            if not vectors.size:
                return np.zeros((len(items), 0), dtype=np.float32)
            now = time.time()
            statements = []
            for digest, vector in zip(missing, vectors):
                found[digest] = vector
                statements.append((
                    "INSERT OR REPLACE INTO query_vectors (namespace, "
                    "text_hash, dim, vector, created_at) VALUES (?, ?, ?, ?, ?)",
                    (self.namespace, digest, int(vector.shape[0]),
                     vector.astype(np.float32).tobytes(), now)))
            self.manifest.transaction(statements)
        dims = {v.shape[0] for v in found.values()}
        if len(dims) != 1:
            return np.zeros((len(items), 0), dtype=np.float32)
        with self._lock:
            self._memory.update(found)
        return np.stack([found[h] for h in hashes]).astype(np.float32)


# --------------------------------------------------------------------------- #
# The paper cache
# --------------------------------------------------------------------------- #
class PaperCache:
    """Every layer of the cache, behind one `index_for(ref, fetcher)` call."""

    def __init__(self, root: Path, embedder: Embedder, *,
                 target_tokens: int = chunking.TARGET_TOKENS,
                 overlap_tokens: int = chunking.OVERLAP_TOKENS,
                 min_chunk_tokens: int = chunking.MIN_CHUNK_TOKENS,
                 miss_retry_days: float = DEFAULT_MISS_RETRY_DAYS,
                 vector_flush_papers: int = VECTOR_FLUSH_PAPERS,
                 stats: Optional[CacheStats] = None):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(self.root / "manifest.sqlite")
        self.stats = stats or CacheStats()
        self.base_embedder = embedder
        self.chunk_params = {"target_tokens": int(target_tokens),
                             "overlap_tokens": int(overlap_tokens),
                             "min_chunk_tokens": int(min_chunk_tokens)}
        self.chunk_ns = chunking.chunker_namespace(**self.chunk_params)
        self.embed_ns = embedder.namespace
        self.vector_ns = f"{self.chunk_ns}__{self.embed_ns}"
        self.raw = RawStore(self.root / "raw", self.manifest, self.stats,
                            miss_retry_days)
        self.ids = IdMapStore(self.manifest, self.stats, miss_retry_days)
        self.vectors = VectorStore(self.root / "vectors" / self.vector_ns,
                                   self.vector_ns, self.manifest,
                                   vector_flush_papers)
        self.embedder = CachedQueryEmbedder(embedder, self.manifest,
                                            self.stats, self._model_ready)
        self._fingerprint_checked = False
        self._fingerprint_lock = threading.Lock()
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

        self.manifest.ensure_namespace(PARSED_NAMESPACE, "parsed", {})
        self.manifest.ensure_namespace(
            self.chunk_ns, "chunks",
            {"chunker_version": chunking.CHUNKER_VERSION, **self.chunk_params})
        self.manifest.ensure_namespace(
            self.embed_ns, "embeddings",
            {"provider": embedder.provider, "model": embedder.model_name,
             "revision": embedder.revision})

    def close(self) -> None:
        self.flush()
        self.manifest.close()

    def __enter__(self) -> "PaperCache":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----------------------------------------------------------------- #
    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    def _model_ready(self) -> None:
        """Before the first model call: check the weights match the namespace."""
        if self._fingerprint_checked:
            return
        with self._fingerprint_lock:
            if self._fingerprint_checked:
                return
            if self.base_embedder.load():
                self.manifest.check_fingerprint(
                    self.embed_ns, self.base_embedder.fingerprint())
            self._fingerprint_checked = True

    def parsed_path(self, paper_key: str) -> Path:
        return self.root / "parsed" / _shard(paper_key) / f"{paper_key}.json.gz"

    def chunks_path(self, paper_key: str, namespace: Optional[str] = None) -> Path:
        return (self.root / "chunks" / (namespace or self.chunk_ns)
                / _shard(paper_key) / f"{paper_key}.parquet")

    def bm25_path(self, paper_key: str, namespace: Optional[str] = None) -> Path:
        return (self.root / "bm25" / (namespace or self.chunk_ns)
                / _shard(paper_key) / f"{paper_key}.json.gz")

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    # ----------------------------------------------------------------- #
    def paper_key_for(self, ref_key: str) -> Optional[str]:
        row = self.manifest.one("SELECT paper_key FROM aliases WHERE ref_key = ?",
                                (ref_key,))
        return row["paper_key"] if row else None

    def unknown(self, refs: Sequence[Any]) -> List[Any]:
        """The references the cache has never parsed. Only these need
        identifier resolution."""
        return [ref for ref in refs if self.paper_key_for(ref.key) is None]

    def prefetch(self, refs: Sequence[Any]) -> None:
        keys = [k for k in (self.paper_key_for(r.key) for r in refs) if k]
        if keys:
            self.vectors.prefetch(keys)

    def flush(self) -> None:
        """End of a batch: commit new vectors, drop unused prefetched ones."""
        self.vectors.flush()
        self.vectors.clear_prefetched()

    # ----------------------------------------------------------------- #
    # Parsed layer
    # ----------------------------------------------------------------- #
    def _load_parsed(self, paper_key: str) -> Optional[Dict[str, Any]]:
        row = self.manifest.layer(paper_key, "parsed", PARSED_NAMESPACE)
        if row is None or row["status"] != "ok":
            return None
        try:
            document = _read_gzip_json(self.root / row["path"])
        except (OSError, EOFError, ValueError, gzip.BadGzipFile) as exc:
            log.warning("Unreadable parsed entry for %s (%s); rebuilding.",
                        paper_key, exc)
            return None
        if document.get("content_hash") != row["content_hash"]:
            return None
        return document

    def _save_parsed(self, paper_key: str, document: Dict[str, Any]) -> None:
        path = self.parsed_path(paper_key)
        data = _gzip_json(document)
        checksum = atomic_write_bytes(path, data)
        now = time.time()
        self.manifest.transaction([
            Manifest.layer_upsert(
                paper_key, "parsed", PARSED_NAMESPACE,
                content_hash=document["content_hash"], checksum=checksum,
                n_items=len(document.get("blocks", [])), size=len(data),
                path=self._relative(path)),
            ("INSERT INTO papers (paper_key, ids_json, title, source_type, "
             "content_hash, first_seen, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
             "ON CONFLICT(paper_key) DO UPDATE SET ids_json = excluded.ids_json, "
             "title = excluded.title, source_type = excluded.source_type, "
             "content_hash = excluded.content_hash, updated_at = excluded.updated_at",
             (paper_key, json.dumps(document.get("ids", {}), sort_keys=True),
              document.get("title", ""), document.get("source_type", ""),
              document["content_hash"], now, now)),
        ])

    def _alias(self, ref_key: str, paper_key: str) -> None:
        self.manifest.transaction([
            ("INSERT OR REPLACE INTO aliases (ref_key, paper_key, updated_at) "
             "VALUES (?, ?, ?)", (ref_key, paper_key, time.time())),
            ("INSERT OR IGNORE INTO aliases (ref_key, paper_key, updated_at) "
             "VALUES (?, ?, ?)", (paper_key, paper_key, time.time())),
        ])

    # ----------------------------------------------------------------- #
    # Chunk and BM25 layers
    # ----------------------------------------------------------------- #
    def _load_chunks(self, paper_key: str,
                     digest: str) -> Optional[List[Dict[str, Any]]]:
        row = self.manifest.layer(paper_key, "chunks", self.chunk_ns)
        if row is None or row["status"] != "ok" or row["content_hash"] != digest:
            return None
        try:
            table = pq.read_table(self.root / row["path"])
        except (OSError, pa.ArrowException) as exc:
            log.warning("Unreadable chunks for %s (%s); rebuilding.",
                        paper_key, exc)
            return None
        chunks = table.to_pylist()
        if len(chunks) != int(row["n_items"]):
            return None
        return chunks

    def _save_chunks(self, paper_key: str, digest: str,
                     chunks: List[Dict[str, Any]]) -> None:
        data = _chunks_parquet(chunks)
        path = self.chunks_path(paper_key)
        checksum = atomic_write_bytes(path, data)
        self.manifest.transaction([Manifest.layer_upsert(
            paper_key, "chunks", self.chunk_ns, content_hash=digest,
            checksum=checksum, n_items=len(chunks), size=len(data),
            path=self._relative(path))])

    def _load_bm25(self, paper_key: str, digest: str,
                   n_chunks: int) -> Optional[BM25]:
        row = self.manifest.layer(paper_key, "bm25", self.chunk_ns)
        if row is None or row["status"] != "ok" or row["content_hash"] != digest:
            return None
        try:
            index = BM25.from_state(_read_gzip_json(self.root / row["path"]))
        except (OSError, EOFError, ValueError, KeyError, TypeError,
                gzip.BadGzipFile) as exc:
            log.warning("Unreadable BM25 index for %s (%s); rebuilding.",
                        paper_key, exc)
            return None
        return index if index.doc_count == n_chunks else None

    def _save_bm25(self, paper_key: str, digest: str, index: BM25) -> None:
        data = _gzip_json(index.to_state())
        path = self.bm25_path(paper_key)
        checksum = atomic_write_bytes(path, data)
        self.manifest.transaction([Manifest.layer_upsert(
            paper_key, "bm25", self.chunk_ns, content_hash=digest,
            checksum=checksum, n_items=index.doc_count, size=len(data),
            path=self._relative(path))])

    # ----------------------------------------------------------------- #
    # Embedding layer
    # ----------------------------------------------------------------- #
    def _load_vectors(self, paper_key: str, digest: str,
                      chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        row = self.manifest.layer(paper_key, "embeddings", self.vector_ns)
        if row is None or row["status"] != "ok" or row["content_hash"] != digest:
            return None
        stored = self.vectors.get(paper_key)
        if stored is None:
            return None
        chunk_ids, matrix = stored
        if chunk_ids != [c["chunk_id"] for c in chunks]:
            return None
        return matrix

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        if not chunks or not self.base_embedder.available:
            return np.zeros((0, 0), dtype=np.float32)
        self._model_ready()
        texts = [
            # The section name is part of what a chunk is about, so it is
            # embedded with the text rather than left to the boost alone.
            f"{c.get('section', '')}: {c.get('title', '')}\n{c['text']}"
            for c in chunks
        ]
        vectors = self.base_embedder.encode(texts)
        if vectors.size:
            self.stats.add("embedding_calls")
            self.stats.add("chunks_embedded", len(chunks))
        return vectors

    # ----------------------------------------------------------------- #
    def index_for(self, ref: Any, fetcher: Any) -> PaperIndex:
        """Return a search index for one paper, building only missing layers."""
        with self._lock_for(ref.key):
            document = None
            paper_key = self.paper_key_for(ref.key)
            if paper_key:
                document = self._load_parsed(paper_key)
            if document is not None:
                self.stats.hit("parsed")
            else:
                self.stats.miss("parsed")
                document = fetcher.fetch(ref)
                paper_key = paper_key or _canonical_key(ref, document)
                document["content_hash"] = chunking.content_hash(document)
                document["cache_key"] = paper_key
                # "no_text" usually means a network failure or a paper that
                # is not yet indexed. It is not stored, so it is retried.
                if document.get("source_type") != "no_text":
                    self._save_parsed(paper_key, _cacheable(document))
                    self._alias(ref.key, paper_key)
            digest = document["content_hash"]

            chunks = self._load_chunks(paper_key, digest)
            if chunks is not None:
                self.stats.hit("chunks")
            else:
                self.stats.miss("chunks")
                chunks = chunking.chunk_document(document, paper_key,
                                                 **self.chunk_params)
                if document.get("source_type") != "no_text":
                    self._save_chunks(paper_key, digest, chunks)

            bm25 = self._load_bm25(paper_key, digest, len(chunks))
            if bm25 is not None:
                self.stats.hit("bm25")
            else:
                self.stats.miss("bm25")
                bm25 = BM25.for_chunks(chunks)
                if document.get("source_type") != "no_text":
                    self._save_bm25(paper_key, digest, bm25)

            vectors = None
            if chunks:
                vectors = self._load_vectors(paper_key, digest, chunks)
                if vectors is not None:
                    self.stats.hit("embeddings")
                else:
                    self.stats.miss("embeddings")
                    vectors = self.embed_chunks(chunks)
                    # Vectors are stored only when the model gave some, so a
                    # run without a model leaves a miss to fill in later.
                    if (vectors.size and document.get("source_type")
                            != "no_text"):
                        self.vectors.put(paper_key, digest,
                                         [c["chunk_id"] for c in chunks],
                                         vectors)

            run_document = dict(document)
            run_document["chunks"] = chunks
            return PaperIndex(_for_run(run_document, ref), vectors,
                              self.embedder, bm25)

    # ----------------------------------------------------------------- #
    def record_run(self, run_id: str) -> None:
        snap = self.stats.snapshot()
        now = time.time()
        self.manifest.transaction([
            ("INSERT OR REPLACE INTO run_cache_stats (run_id, layer, hits, "
             "misses, updated_at) VALUES (?, ?, ?, ?, ?)",
             (run_id, layer, v["hits"], v["misses"], now))
            for layer, v in snap["layers"].items()
        ])


def _canonical_key(ref: Any, document: Dict[str, Any]) -> str:
    """The cache key: the strongest identifier the paper turned out to have.

    A paper given once as a PMID and once as a PMCID resolves to the same
    key, so it is cached once.
    """
    from fetch import canonical_key

    ids = {**getattr(ref, "ids", {}),
           **{k: v for k, v in (document.get("ids") or {}).items() if v}}
    if any(ids.get(k) for k in ("pmcid", "pmid", "doi")):
        return canonical_key(ids)
    return ref.key


# Fields that describe one run's view of a paper, not the paper itself.
_RUN_FIELDS = ("paper_key", "article_id", "chunks", "chunk_count", "has_text")


def _cacheable(document: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in document.items() if k not in _RUN_FIELDS}


def _for_run(document: Dict[str, Any], ref: Any) -> Dict[str, Any]:
    document["paper_key"] = ref.key
    extra = getattr(ref, "extra", {}) or {}
    document["article_id"] = extra.get("article_id", "") or ref.label
    document["chunk_count"] = len(document.get("chunks", []))
    document["has_text"] = bool(document.get("chunks"))
    if not document["has_text"]:
        document["source_type"] = "no_text"
    return document


# --------------------------------------------------------------------------- #
# Maintenance: stats, verify, reset, compact, import
# --------------------------------------------------------------------------- #
def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cache_stats(root: Path) -> Dict[str, Any]:
    """What the cache holds, per layer and namespace."""
    root = Path(root).expanduser()
    manifest = Manifest(root / "manifest.sqlite")
    try:
        layers = [dict(r) for r in manifest.query(
            "SELECT layer, namespace, status, COUNT(*) AS papers, "
            "SUM(n_items) AS items, SUM(bytes) AS bytes FROM layers "
            "GROUP BY layer, namespace, status ORDER BY layer, namespace")]
        raw = [dict(r) for r in manifest.query(
            "SELECT status, COUNT(*) AS files, SUM(bytes) AS bytes "
            "FROM raw_files GROUP BY status")]
        papers = manifest.one("SELECT COUNT(*) AS n FROM papers")["n"]
        id_map = manifest.one("SELECT COUNT(*) AS n FROM id_map")["n"]
        queries = manifest.one("SELECT COUNT(*) AS n FROM query_vectors")["n"]
        namespaces = [dict(r) for r in manifest.query(
            "SELECT namespace, layer, params_json, fingerprint FROM namespaces "
            "ORDER BY layer, namespace")]
    finally:
        manifest.close()
    return {
        "cache_dir": str(root), "papers": papers, "raw": raw, "layers": layers,
        "id_map_entries": id_map, "query_vectors": queries,
        "namespaces": namespaces, "bytes_on_disk": _dir_size(root),
    }


@dataclass
class VerifyReport:
    checked: Dict[str, int] = field(default_factory=dict)
    repaired: List[str] = field(default_factory=list)
    quarantined: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.repaired or self.quarantined or self.removed)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "checked": self.checked,
                "repaired": self.repaired, "quarantined": self.quarantined,
                "removed": self.removed}


def verify_cache(root: Path, *, repair: bool = True) -> VerifyReport:
    """Check every entry against its checksum and its upstream layer.

    With `repair`, a bad derived entry (chunks, BM25) is rebuilt from the layer
    below it, and a bad source entry (raw, parsed) or a bad vector set is moved
    to `quarantine/` and dropped from the manifest, so the next run rebuilds
    it. Without `repair`, problems are only reported.
    """
    root = Path(root).expanduser()
    manifest = Manifest(root / "manifest.sqlite")
    report = VerifyReport()
    quarantine = root / "quarantine" / time.strftime("%Y%m%d-%H%M%S")

    def move_to_quarantine(path: Path) -> None:
        if not path.exists():
            return
        target = quarantine / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))

    def count(name: str) -> None:
        report.checked[name] = report.checked.get(name, 0) + 1

    try:
        # Temporary files left by a crash mid-write.
        for tmp in list(root.rglob(".*.tmp")):
            if quarantine in tmp.parents:
                continue
            report.removed.append(str(tmp.relative_to(root)))
            if repair:
                tmp.unlink(missing_ok=True)

        # -- raw ---------------------------------------------------------- #
        for row in manifest.query("SELECT * FROM raw_files WHERE status = 'ok'"):
            count("raw")
            path = root / row["path"]
            problem = ""
            if not path.exists():
                problem = "file missing"
            elif sha256_file(path) != row["checksum"]:
                problem = "checksum mismatch"
            else:
                try:
                    gzip.decompress(path.read_bytes())
                except (OSError, EOFError, gzip.BadGzipFile):
                    problem = "not valid gzip"
            if problem:
                report.quarantined.append(f"raw/{row['name']}: {problem}")
                if repair:
                    move_to_quarantine(path)
                    manifest.execute("DELETE FROM raw_files WHERE name = ?",
                                     (row["name"],))

        # -- parsed ------------------------------------------------------- #
        for row in manifest.query(
                "SELECT * FROM layers WHERE layer = 'parsed'"):
            count("parsed")
            path = root / row["path"]
            problem = ""
            if not path.exists():
                problem = "file missing"
            elif sha256_file(path) != row["checksum"]:
                problem = "checksum mismatch"
            else:
                try:
                    document = _read_gzip_json(path)
                    if document.get("content_hash") != row["content_hash"]:
                        problem = "content hash mismatch"
                except (OSError, EOFError, ValueError, gzip.BadGzipFile):
                    problem = "unreadable"
            if problem:
                report.quarantined.append(
                    f"parsed/{row['paper_key']}: {problem}")
                if repair:
                    move_to_quarantine(path)
                    manifest.transaction([
                        ("DELETE FROM layers WHERE paper_key = ?",
                         (row["paper_key"],)),
                        ("DELETE FROM aliases WHERE paper_key = ?",
                         (row["paper_key"],)),
                    ])

        parsed_ok = {r["paper_key"]: r for r in manifest.query(
            "SELECT * FROM layers WHERE layer = 'parsed' AND status = 'ok'")}

        # -- chunks and BM25: rebuildable --------------------------------- #
        chunk_ns_params = {r["namespace"]: json.loads(r["params_json"])
                           for r in manifest.query(
                               "SELECT * FROM namespaces WHERE layer = 'chunks'")}
        for layer in ("chunks", "bm25"):
            for row in manifest.query(
                    "SELECT * FROM layers WHERE layer = ?", (layer,)):
                count(layer)
                path = root / row["path"]
                problem = ""
                if not path.exists():
                    problem = "file missing"
                elif sha256_file(path) != row["checksum"]:
                    problem = "checksum mismatch"
                elif row["paper_key"] not in parsed_ok:
                    problem = "no parsed entry"
                elif (parsed_ok[row["paper_key"]]["content_hash"]
                      != row["content_hash"]):
                    problem = "stale content hash"
                if not problem:
                    continue
                label = f"{layer}/{row['namespace']}/{row['paper_key']}: {problem}"
                if not repair:
                    report.quarantined.append(label)
                    continue
                parsed = parsed_ok.get(row["paper_key"])
                params = chunk_ns_params.get(row["namespace"])
                if parsed is not None and params is not None:
                    document = _read_gzip_json(root / parsed["path"])
                    chunk_params = {k: params[k] for k in (
                        "target_tokens", "overlap_tokens", "min_chunk_tokens")}
                    chunks = chunking.chunk_document(
                        document, row["paper_key"], **chunk_params)
                    if layer == "chunks":
                        _rewrite_chunks(root, manifest, row, chunks,
                                        document["content_hash"])
                    else:
                        _rewrite_bm25(root, manifest, row,
                                      BM25.for_chunks(chunks),
                                      document["content_hash"])
                    report.repaired.append(label)
                else:
                    move_to_quarantine(path)
                    manifest.execute(
                        "DELETE FROM layers WHERE paper_key = ? AND layer = ? "
                        "AND namespace = ?",
                        (row["paper_key"], layer, row["namespace"]))
                    report.quarantined.append(label)

        # -- embeddings --------------------------------------------------- #
        vectors_root = root / "vectors"
        namespaces = sorted(p.name for p in vectors_root.iterdir()
                            if p.is_dir()) if vectors_root.exists() else []
        manifest_ns = {r["namespace"] for r in manifest.query(
            "SELECT DISTINCT namespace FROM layers WHERE layer = 'embeddings'")}
        for namespace in sorted(set(namespaces) | manifest_ns):
            store = VectorStore(vectors_root / namespace, namespace, manifest)
            chunk_ns = namespace.split("__", 1)[0]
            rows = {r["paper_key"]: r for r in manifest.query(
                "SELECT * FROM layers WHERE layer = 'embeddings' AND "
                "namespace = ?", (namespace,))}
            try:
                stored_keys = store.stored_keys()
            except Exception as exc:  # noqa: BLE001 - a broken dataset
                report.quarantined.append(
                    f"vectors/{namespace}: unreadable ({exc})")
                if repair:
                    move_to_quarantine(vectors_root / namespace)
                    manifest.execute(
                        "DELETE FROM layers WHERE layer = 'embeddings' AND "
                        "namespace = ?", (namespace,))
                continue
            bad: List[str] = []
            keys = sorted(rows)
            for start in range(0, len(keys), 500):
                part = keys[start:start + 500]
                stored = store.read_many(part)
                for paper_key in part:
                    row = rows[paper_key]
                    count("embeddings")
                    problem = ""
                    chunk_row = manifest.layer(paper_key, "chunks", chunk_ns)
                    if paper_key not in stored:
                        problem = "vectors missing"
                    elif len(stored[paper_key][0]) != int(row["n_items"]):
                        problem = "vector count mismatch"
                    elif sha256_bytes(stored[paper_key][1].astype(np.float32)
                                      .tobytes()) != row["checksum"]:
                        problem = "checksum mismatch"
                    elif chunk_row is not None and (
                            int(chunk_row["n_items"]) != int(row["n_items"])
                            or chunk_row["content_hash"] != row["content_hash"]):
                        problem = "does not match the chunks"
                    if problem:
                        bad.append(paper_key)
                        report.quarantined.append(
                            f"vectors/{namespace}/{paper_key}: {problem}")
            orphans = sorted(stored_keys - set(rows))
            for paper_key in orphans:
                report.removed.append(
                    f"vectors/{namespace}/{paper_key}: no manifest entry")
            if repair and (bad or orphans):
                store.delete(bad + orphans)
                manifest.transaction([
                    ("DELETE FROM layers WHERE paper_key = ? AND layer = "
                     "'embeddings' AND namespace = ?", (key, namespace))
                    for key in bad
                ])
    finally:
        manifest.close()
    return report


def _rewrite_chunks(root: Path, manifest: Manifest, row: sqlite3.Row,
                    chunks: List[Dict[str, Any]], digest: str) -> None:
    data = _chunks_parquet(chunks)
    checksum = atomic_write_bytes(root / row["path"], data)
    manifest.transaction([Manifest.layer_upsert(
        row["paper_key"], "chunks", row["namespace"], content_hash=digest,
        checksum=checksum, n_items=len(chunks), size=len(data),
        path=row["path"])])


def _rewrite_bm25(root: Path, manifest: Manifest, row: sqlite3.Row,
                  index: BM25, digest: str) -> None:
    data = _gzip_json(index.to_state())
    checksum = atomic_write_bytes(root / row["path"], data)
    manifest.transaction([Manifest.layer_upsert(
        row["paper_key"], "bm25", row["namespace"], content_hash=digest,
        checksum=checksum, n_items=index.doc_count, size=len(data),
        path=row["path"])])


def reset_cache(root: Path, layers: Optional[Sequence[str]] = None) -> List[str]:
    """Delete the whole cache, or only some layers. Returns what was removed.

    Removing a layer also removes every layer built on it, so the cache never
    holds vectors for chunks it no longer has.
    """
    root = Path(root).expanduser()
    if not layers:
        if root.exists():
            shutil.rmtree(root)
        return [str(root)]
    order = list(LAYERS)
    unknown = [l for l in layers if l not in order]
    if unknown:
        raise ValueError(f"Unknown layer(s): {unknown}. Known: {order}")
    first = min(order.index(l) for l in layers)
    doomed = order[first:]
    removed: List[str] = []
    manifest = Manifest(root / "manifest.sqlite")
    try:
        folders = {"raw": "raw", "parsed": "parsed", "chunks": "chunks",
                   "bm25": "bm25", "embeddings": "vectors"}
        for layer in doomed:
            folder = root / folders[layer]
            if folder.exists():
                shutil.rmtree(folder)
                removed.append(str(folder))
            if layer == "raw":
                manifest.execute("DELETE FROM raw_files")
                manifest.execute("DELETE FROM id_map")
            else:
                manifest.execute("DELETE FROM layers WHERE layer = ?", (layer,))
            if layer == "parsed":
                manifest.execute("DELETE FROM aliases")
                manifest.execute("DELETE FROM papers")
            if layer == "embeddings":
                manifest.execute("DELETE FROM query_vectors")
                manifest.execute("UPDATE namespaces SET fingerprint = NULL "
                                 "WHERE layer = 'embeddings'")
    finally:
        manifest.close()
    return removed


def compact_cache(root: Path) -> List[str]:
    """Merge LanceDB fragments, refresh the paper_key index, vacuum SQLite."""
    root = Path(root).expanduser()
    manifest = Manifest(root / "manifest.sqlite")
    done: List[str] = []
    try:
        vectors_root = root / "vectors"
        if vectors_root.exists():
            for folder in sorted(p for p in vectors_root.iterdir() if p.is_dir()):
                VectorStore(folder, folder.name, manifest).compact()
                done.append(folder.name)
        manifest.execute("VACUUM")
    finally:
        manifest.close()
    return done


def import_raw(root: Path, source_dir: Path) -> Dict[str, int]:
    """Copy raw downloads from an older cache layout into this cache.

    An empty file was the old way to record "this source has no record"; such
    files are skipped, so those sources are asked again.
    """
    root = Path(root).expanduser()
    manifest = Manifest(root / "manifest.sqlite")
    store = RawStore(root / "raw", manifest)
    counts = {"imported": 0, "already_present": 0, "skipped_empty": 0,
              "unreadable": 0}
    try:
        for path in sorted(Path(source_dir).expanduser().rglob("*.xml.gz")):
            name = path.name[:-len(".xml.gz")]
            if manifest.one("SELECT 1 FROM raw_files WHERE name = ? AND "
                            "status = 'ok'", (name,)):
                counts["already_present"] += 1
                continue
            try:
                text = gzip.decompress(path.read_bytes()).decode("utf-8")
            except (OSError, EOFError, UnicodeDecodeError, gzip.BadGzipFile):
                counts["unreadable"] += 1
                continue
            if not text.strip():
                counts["skipped_empty"] += 1
                continue
            store.put(name, text)
            counts["imported"] += 1
    finally:
        manifest.close()
    return counts
