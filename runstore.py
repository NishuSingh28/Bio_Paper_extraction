"""Per-run storage: the column set, the results, and the resume state.

A run owns a directory. Inside it are the column set the user defined, and a
SQLite database holding one row per (paper, column group). Two consequences
matter:

* Runs cannot mix. A different column set is a different run directory, so a
  checkpoint from one run can never be read into another.
* Resume is exact and cheap. Restarting asks the database which (paper, group)
  pairs are already done. At 100,000 papers that is one indexed query, not a
  large JSON file loaded into memory.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from columns import ColumnSpec, column_set_fingerprint

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_key   TEXT PRIMARY KEY,
    article_id  TEXT,
    title       TEXT,
    source_type TEXT,
    status      TEXT,
    meta_json   TEXT,
    updated_at  REAL
);
CREATE TABLE IF NOT EXISTS groups (
    paper_key   TEXT NOT NULL,
    group_key   TEXT NOT NULL,
    values_json TEXT NOT NULL,
    stats_json  TEXT NOT NULL,
    updated_at  REAL,
    PRIMARY KEY (paper_key, group_key)
);
CREATE INDEX IF NOT EXISTS groups_by_paper ON groups (paper_key);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class RunInfo:
    run_id: str
    created_at: float
    label: str
    columns: List[ColumnSpec]
    settings: Dict[str, Any]
    fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "created_at_text": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)),
            "label": self.label,
            "fingerprint": self.fingerprint,
            "columns": [c.to_dict() for c in self.columns],
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunInfo":
        columns = [ColumnSpec.from_dict(c) for c in data.get("columns", [])]
        return cls(
            run_id=data["run_id"],
            created_at=float(data.get("created_at", 0.0)),
            label=data.get("label", ""),
            columns=columns,
            settings=data.get("settings", {}),
            fingerprint=data.get("fingerprint") or column_set_fingerprint(columns),
        )


def make_run_id(columns: Sequence[ColumnSpec], label: str = "") -> str:
    """A run id that carries its column set, so two runs cannot collide."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    fingerprint = column_set_fingerprint(list(columns))
    suffix = f"-{label}" if label else ""
    # Two runs started in the same second with the same columns must still
    # get two folders; otherwise the second silently resumes the first.
    return f"{stamp}-{fingerprint}-{uuid.uuid4().hex[:4]}{suffix}"


class RunStore:
    """The results database for one run."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.run_dir / "results.sqlite"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                     timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL lets a reader (the UI) see progress while writers work.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)))
            self._conn.commit()

    # ----------------------------------------------------------------- #
    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "RunStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----------------------------------------------------------------- #
    def save_group(self, paper_key: str, group_key: str,
                   values: Dict[str, Any], stats: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO groups "
                "(paper_key, group_key, values_json, stats_json, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (paper_key, group_key, json.dumps(values, ensure_ascii=False),
                 json.dumps(stats, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    def save_paper(self, paper_key: str, meta: Dict[str, Any],
                   status: str = "ok") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO papers "
                "(paper_key, article_id, title, source_type, status, "
                " meta_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (paper_key, str(meta.get("article_id", "")),
                 str(meta.get("title", "")), str(meta.get("source_type", "")),
                 status, json.dumps(meta, ensure_ascii=False), time.time()),
            )
            self._conn.commit()

    # ----------------------------------------------------------------- #
    def done_groups(self) -> Dict[str, Set[str]]:
        """Every (paper, group) pair already stored."""
        out: Dict[str, Set[str]] = {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT paper_key, group_key FROM groups").fetchall()
        for row in rows:
            out.setdefault(row["paper_key"], set()).add(row["group_key"])
        return out

    def completed_papers(self, group_keys: Sequence[str]) -> Set[str]:
        """Papers whose every group is stored, so they need no further work."""
        wanted = set(group_keys)
        if not wanted:
            return set()
        return {
            key for key, done in self.done_groups().items()
            if wanted.issubset(done)
        }

    def paper_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM papers").fetchone()
        return int(row["n"])

    def load_paper(self, paper_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            paper = self._conn.execute(
                "SELECT * FROM papers WHERE paper_key = ?",
                (paper_key,)).fetchone()
            group_rows = self._conn.execute(
                "SELECT * FROM groups WHERE paper_key = ?",
                (paper_key,)).fetchall()
        if paper is None and not group_rows:
            return None
        meta = json.loads(paper["meta_json"]) if paper else {}
        values: Dict[str, Any] = {}
        calls: List[Dict[str, Any]] = []
        for row in group_rows:
            values.update(json.loads(row["values_json"]))
            calls.append(json.loads(row["stats_json"]))
        meta["values"] = values
        meta["calls"] = calls
        return meta

    def iter_papers(self, batch_size: int = 500) -> Iterator[Dict[str, Any]]:
        """Stream every stored paper. Memory stays flat at any run size.

        Groups are read a batch of papers at a time rather than one paper at a
        time: exporting 100,000 papers is then a few hundred queries instead of
        a few hundred thousand.
        """
        offset = 0
        while True:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT paper_key, meta_json FROM papers ORDER BY rowid "
                    "LIMIT ? OFFSET ?", (batch_size, offset)).fetchall()
                if not rows:
                    return
                keys = [row["paper_key"] for row in rows]
                placeholders = ",".join("?" * len(keys))
                group_rows = self._conn.execute(
                    f"SELECT paper_key, values_json, stats_json FROM groups "
                    f"WHERE paper_key IN ({placeholders})", keys).fetchall()

            grouped: Dict[str, List[sqlite3.Row]] = {}
            for group_row in group_rows:
                grouped.setdefault(group_row["paper_key"], []).append(group_row)

            for row in rows:
                try:
                    record = json.loads(row["meta_json"])
                except (ValueError, TypeError):
                    record = {}
                values: Dict[str, Any] = {}
                calls: List[Dict[str, Any]] = []
                for group_row in grouped.get(row["paper_key"], []):
                    values.update(json.loads(group_row["values_json"]))
                    calls.append(json.loads(group_row["stats_json"]))
                record["values"] = values
                record["calls"] = calls
                yield record
            offset += len(rows)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            papers = self._conn.execute(
                "SELECT COUNT(*) AS n, "
                "SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors "
                "FROM papers").fetchone()
            groups = self._conn.execute(
                "SELECT COUNT(*) AS n FROM groups").fetchone()
        return {
            "papers_done": int(papers["n"] or 0),
            "papers_with_errors": int(papers["errors"] or 0),
            "group_calls_done": int(groups["n"] or 0),
        }

    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)))
            self._conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except ValueError:
            return row["value"]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class RunRegistry:
    """The runs directory. Creating, finding and describing runs."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def create(self, columns: Sequence[ColumnSpec], settings: Dict[str, Any],
               label: str = "") -> RunInfo:
        columns = list(columns)
        info = RunInfo(
            run_id=make_run_id(columns, label),
            created_at=time.time(),
            label=label,
            columns=columns,
            settings=dict(settings),
            fingerprint=column_set_fingerprint(columns),
        )
        directory = self.run_dir(info.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.json").write_text(
            json.dumps(info.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8")
        return info

    def load(self, run_id: str) -> Optional[RunInfo]:
        path = self.run_dir(run_id) / "run.json"
        if not path.exists():
            return None
        try:
            return RunInfo.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError) as exc:
            log.warning("Unreadable run description %s: %s", path, exc)
            return None

    def find_matching(self, columns: Sequence[ColumnSpec]) -> List[RunInfo]:
        """Runs whose column set is identical, newest first.

        This is what makes resume safe: a run is only offered for resume when
        the columns match exactly.
        """
        fingerprint = column_set_fingerprint(list(columns))
        matches = [info for info in self.list_runs()
                   if info.fingerprint == fingerprint]
        return matches

    def list_runs(self) -> List[RunInfo]:
        runs: List[RunInfo] = []
        for directory in sorted(self.root.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            info = self.load(directory.name)
            if info is not None:
                runs.append(info)
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs

    def describe(self, info: RunInfo) -> Dict[str, Any]:
        store_path = self.run_dir(info.run_id) / "results.sqlite"
        done = 0
        if store_path.exists():
            try:
                with sqlite3.connect(store_path) as connection:
                    done = int(connection.execute(
                        "SELECT COUNT(*) FROM papers").fetchone()[0])
            except sqlite3.Error:
                done = 0
        return {
            "Run": info.run_id,
            "Created": time.strftime("%Y-%m-%d %H:%M",
                                     time.localtime(info.created_at)),
            "Columns": len(info.columns),
            "Papers done": done,
            "Label": info.label,
        }
