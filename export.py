"""Turn a run's stored results into the workbook the user downloads.

Five tables:

* Extraction    - one row per paper, holding exactly the columns the user
                  defined, in the order they defined them.
* Evidence      - one row per (paper, column): the value, the sentence that
                  supports it, the section it came from, and the confidence.
* Review queue  - the subset of Evidence a human has to look at.
* Papers        - what happened to each paper on the way in.
* Run info      - the column set and the settings, so the run is reproducible.

Excel holds 1,048,576 rows. At 100,000 papers the Extraction sheet fits and the
Evidence sheet does not, so Evidence and the review queue are always written
beside the workbook as CSV (and Parquet when pyarrow is present), and are folded
into the workbook only when they fit.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from columns import ColumnSpec
from runstore import RunInfo, RunStore

log = logging.getLogger(__name__)

EXCEL_ROW_LIMIT = 1_048_575  # one row is the header
REVIEW_PREVIEW_CAP = 200_000

IDENTIFIER_COLUMNS = ["Paper ID", "Title", "Source type"]
TRAILING_COLUMNS = ["Values flagged", "Review flags"]

EVIDENCE_COLUMNS = [
    "Paper ID", "Title", "Column", "Value", "Evidence span", "Source section",
    "Confidence", "Model confidence", "Quote check", "Validation",
    "Validation error", "Flags",
]

PAPERS_COLUMNS = [
    "Paper ID", "Title", "Year", "Journal", "PMID", "PMCID", "DOI", "URL",
    "Source type", "Source", "Chunks", "Fetch notes", "LLM calls",
    "Input tokens", "Output tokens", "Cost (USD)", "Seconds",
    "Values flagged", "Errors",
]


def _paper_id(record: Dict[str, Any]) -> str:
    ids = record.get("ids") or {}
    return (str(record.get("article_id") or "").strip()
            or ids.get("pmid") or ids.get("pmcid") or ids.get("doi")
            or ids.get("url") or record.get("paper_key", ""))


def extraction_row(record: Dict[str, Any], specs: Sequence[ColumnSpec],
                   include_identifiers: bool = True) -> Dict[str, Any]:
    """One paper as one row, with exactly the user's columns."""
    values = record.get("values") or {}
    row: Dict[str, Any] = {}
    if include_identifiers:
        row["Paper ID"] = _paper_id(record)
        row["Title"] = record.get("title", "")
        row["Source type"] = record.get("source_type", "")

    flags: List[str] = []
    for spec in specs:
        entry = values.get(spec.key) or {}
        row[spec.name] = entry.get("value", "")
        for flag in entry.get("flags", []) or []:
            flags.append(f"{spec.name}: {flag}")

    if include_identifiers:
        row["Values flagged"] = int(record.get("flagged_values", 0))
        row["Review flags"] = "; ".join(flags)
    return row


def evidence_rows(record: Dict[str, Any],
                  specs: Sequence[ColumnSpec]) -> List[Dict[str, Any]]:
    values = record.get("values") or {}
    paper_id = _paper_id(record)
    title = record.get("title", "")
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        entry = values.get(spec.key) or {}
        rows.append({
            "Paper ID": paper_id,
            "Title": title,
            "Column": spec.name,
            "Value": entry.get("value", ""),
            "Evidence span": entry.get("evidence_span", ""),
            "Source section": entry.get("source_section", ""),
            "Confidence": entry.get("confidence", 0.0),
            "Model confidence": entry.get("model_confidence", 0.0),
            "Quote check": entry.get("quote_check", ""),
            "Validation": entry.get("validation", ""),
            "Validation error": entry.get("validation_error", ""),
            "Flags": "; ".join(entry.get("flags", []) or []),
            "_needs_review": bool(entry.get("needs_review")),
        })
    return rows


def paper_row(record: Dict[str, Any]) -> Dict[str, Any]:
    ids = record.get("ids") or {}
    return {
        "Paper ID": _paper_id(record),
        "Title": record.get("title", ""),
        "Year": record.get("year", ""),
        "Journal": record.get("journal", ""),
        "PMID": ids.get("pmid", ""),
        "PMCID": ids.get("pmcid", ""),
        "DOI": ids.get("doi", ""),
        "URL": ids.get("url", ""),
        "Source type": record.get("source_type", ""),
        "Source": record.get("source_api", ""),
        "Chunks": record.get("chunk_count", 0),
        "Fetch notes": record.get("fetch_notes", ""),
        "LLM calls": record.get("llm_calls", 0),
        "Input tokens": record.get("input_tokens", 0),
        "Output tokens": record.get("output_tokens", 0),
        "Cost (USD)": round(float(record.get("cost_usd", 0.0)), 6),
        "Seconds": round(float(record.get("elapsed_s", 0.0)), 2),
        "Values flagged": record.get("flagged_values", 0),
        "Errors": "; ".join(record.get("errors", []) or []),
    }


def run_info_rows(info: RunInfo, stats: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = [
        {"Setting": "Run ID", "Value": info.run_id},
        {"Setting": "Created", "Value": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(info.created_at))},
        {"Setting": "Column set fingerprint", "Value": info.fingerprint},
        {"Setting": "Columns", "Value": str(len(info.columns))},
    ]
    for key, value in (info.settings or {}).items():
        rows.append({"Setting": key, "Value": str(value)})
    for key, value in (stats or {}).items():
        if key == "cache" and isinstance(value, dict):
            for layer, counts in (value.get("layers") or {}).items():
                rows.append({"Setting": f"cache.{layer}",
                             "Value": f"{counts.get('hits', 0)} hits, "
                                      f"{counts.get('misses', 0)} misses"})
            for name in ("embedding_calls", "chunks_embedded",
                         "query_encode_calls", "network_requests"):
                rows.append({"Setting": f"cache.{name}",
                             "Value": str(value.get(name, 0))})
            for layer, namespace in (value.get("namespaces") or {}).items():
                rows.append({"Setting": f"cache.namespace.{layer}",
                             "Value": str(namespace)})
            continue
        rows.append({"Setting": f"result.{key}", "Value": str(value)})
    for spec in info.columns:
        rows.append({
            "Setting": f"column.{spec.name}",
            "Value": json.dumps(
                {k: v for k, v in spec.to_dict().items()
                 if v not in ("", [], None)},
                ensure_ascii=False),
        })
    return rows


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def build_tables(
    info: RunInfo,
    store: RunStore,
    *,
    include_identifiers: bool = True,
    stats: Optional[Dict[str, Any]] = None,
    evidence_csv: Optional[Path] = None,
    review_csv: Optional[Path] = None,
    max_rows_in_memory: int = EXCEL_ROW_LIMIT,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Stream the run store into tables. Returns (tables, a report)."""
    specs = list(info.columns)
    extraction: List[Dict[str, Any]] = []
    papers: List[Dict[str, Any]] = []
    evidence_buffer: List[Dict[str, Any]] = []
    review_buffer: List[Dict[str, Any]] = []
    evidence_count = 0
    review_count = 0

    evidence_writer = review_writer = None
    evidence_handle = review_handle = None
    try:
        if evidence_csv is not None:
            evidence_csv.parent.mkdir(parents=True, exist_ok=True)
            evidence_handle = evidence_csv.open("w", newline="",
                                                encoding="utf-8")
            evidence_writer = csv.DictWriter(evidence_handle,
                                             fieldnames=EVIDENCE_COLUMNS)
            evidence_writer.writeheader()
        if review_csv is not None:
            review_csv.parent.mkdir(parents=True, exist_ok=True)
            review_handle = review_csv.open("w", newline="", encoding="utf-8")
            review_writer = csv.DictWriter(review_handle,
                                           fieldnames=EVIDENCE_COLUMNS)
            review_writer.writeheader()

        for record in store.iter_papers():
            extraction.append(
                extraction_row(record, specs, include_identifiers))
            papers.append(paper_row(record))
            for row in evidence_rows(record, specs):
                needs_review = row.pop("_needs_review")
                evidence_count += 1
                if evidence_writer is not None:
                    evidence_writer.writerow(row)
                if len(evidence_buffer) < max_rows_in_memory:
                    evidence_buffer.append(row)
                if needs_review:
                    review_count += 1
                    if review_writer is not None:
                        review_writer.writerow(row)
                    if len(review_buffer) < REVIEW_PREVIEW_CAP:
                        review_buffer.append(row)
    finally:
        for handle in (evidence_handle, review_handle):
            if handle is not None:
                handle.close()

    extraction_columns = (
        (IDENTIFIER_COLUMNS if include_identifiers else [])
        + [s.name for s in specs]
        + (TRAILING_COLUMNS if include_identifiers else [])
    )
    tables = {
        "Extraction": pd.DataFrame(extraction, columns=extraction_columns),
        "Evidence": pd.DataFrame(evidence_buffer, columns=EVIDENCE_COLUMNS),
        "Review queue": pd.DataFrame(review_buffer, columns=EVIDENCE_COLUMNS),
        "Papers": pd.DataFrame(papers, columns=PAPERS_COLUMNS),
        "Run info": pd.DataFrame(run_info_rows(info, stats or {})),
    }
    report = {
        "papers": len(extraction),
        "evidence_rows": evidence_count,
        "review_rows": review_count,
        "evidence_truncated_in_workbook": evidence_count > len(evidence_buffer),
        "review_truncated_in_workbook": review_count > len(review_buffer),
    }
    return tables, report


def write_workbook(path: Path, tables: Dict[str, pd.DataFrame]) -> Path:
    """Write the workbook atomically, skipping any sheet too tall for Excel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        for name, frame in tables.items():
            if len(frame) > EXCEL_ROW_LIMIT:
                log.warning(
                    "Sheet %r has %d rows, over the Excel limit. It is in the "
                    "CSV beside the workbook instead.", name, len(frame))
                continue
            frame.to_excel(writer, index=False, sheet_name=name[:31])
            sheet = writer.sheets[name[:31]]
            for position, column in enumerate(frame.columns, start=1):
                width = min(60, max(12, len(str(column)) + 2))
                if str(column) in {"Title", "Evidence span", "Review flags",
                                   "Value"}:
                    width = 55
                sheet.column_dimensions[
                    sheet.cell(row=1, column=position).column_letter
                ].width = width
            sheet.freeze_panes = "A2"
    tmp.replace(path)
    return path


def workbook_bytes(tables: Dict[str, pd.DataFrame]) -> bytes:
    import io

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in tables.items():
            if len(frame) > EXCEL_ROW_LIMIT:
                continue
            frame.to_excel(writer, index=False, sheet_name=name[:31])
    return buffer.getvalue()


def write_parquet(path: Path, frame: pd.DataFrame) -> Optional[Path]:
    """Parquet for a table too large for a spreadsheet. None if unavailable."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return path
    except Exception as exc:  # noqa: BLE001 - parquet is a convenience, not a must
        log.warning("Could not write %s: %s", path, exc)
        return None


def export_run(
    run_dir: Path,
    info: RunInfo,
    store: RunStore,
    *,
    include_identifiers: bool = True,
    stats: Optional[Dict[str, Any]] = None,
    basename: str = "extraction",
) -> Dict[str, Any]:
    """Write every output file for a run and report what was written."""
    run_dir = Path(run_dir)
    evidence_csv = run_dir / f"{basename}_evidence.csv"
    review_csv = run_dir / f"{basename}_review_queue.csv"

    tables, report = build_tables(
        info, store,
        include_identifiers=include_identifiers,
        stats=stats,
        evidence_csv=evidence_csv,
        review_csv=review_csv,
    )

    workbook = write_workbook(run_dir / f"{basename}.xlsx", tables)
    extraction_csv = run_dir / f"{basename}.csv"
    tables["Extraction"].to_csv(extraction_csv, index=False)

    written = {
        "workbook": str(workbook),
        "extraction_csv": str(extraction_csv),
        "evidence_csv": str(evidence_csv),
        "review_csv": str(review_csv),
    }
    if report["evidence_rows"] > EXCEL_ROW_LIMIT:
        parquet = write_parquet(run_dir / f"{basename}_evidence.parquet",
                                tables["Evidence"])
        if parquet:
            written["evidence_parquet"] = str(parquet)

    report["files"] = written
    report["tables"] = {name: len(frame) for name, frame in tables.items()}
    return report
