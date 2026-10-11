"""Reading a workbook of papers, such as a screening export.

Column names are matched against candidate lists, so an export from a
screening tool (CADIMA, Rayyan, Covidence, EndNote) loads without editing code.
To teach the reader a new header name, add it to COLUMN_CANDIDATES.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from normalize import clean_text

log = logging.getLogger(__name__)

# Canonical field -> header names seen in the wild, in priority order. Matching
# is case- and whitespace-insensitive.
COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "Article ID": [
        "article id", "paper id", "record id", "reference id", "study id",
        "refid", "id",
    ],
    "Title": ["title", "article title", "primary title", "t1", "ti"],
    "Abstract": ["abstract", "abstract note", "abstract text", "n2", "summary"],
    "Year": [
        "publication year", "year", "primary date", "publication date",
        "date", "py", "y1",
    ],
    "Journal": [
        "periodical full name", "journal", "journal name", "source title",
        "periodical name - user abbreviation 2", "jf", "j2",
    ],
    "DOI": ["doi"],
    "PMID": ["user definable 1", "pmid", "pubmed id", "pubmed"],
    "URL": ["web/url", "url", "link", "link to full-text", "ur", "l2"],
}

REQUIRED_FIELDS = ["Title", "Abstract"]
OUTPUT_COLUMNS = ["Article ID", "Title", "Year", "Journal", "DOI", "PMID",
                  "URL", "Abstract"]

_PMID_RE = re.compile(r"(\d{4,9})")
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


def _norm_header(name: object) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def detect_columns(columns: List[str]) -> Dict[str, List[str]]:
    """Map each canonical field to the source columns that supply it.

    Several source columns can feed one field (CADIMA splits publication year
    across 'publication year' and 'primary date'); they are coalesced on read.
    """
    normalized = {col: _norm_header(col) for col in columns}
    mapping: Dict[str, List[str]] = {}
    for field, candidates in COLUMN_CANDIDATES.items():
        matches: List[str] = []
        for candidate in candidates:
            for col, norm in normalized.items():
                if norm == candidate and col not in matches:
                    matches.append(col)
        if matches:
            mapping[field] = matches
    return mapping


def _coalesce(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """First non-empty value across cols, row by row."""
    out = pd.Series([""] * len(df), index=df.index, dtype=object)
    for col in cols:
        vals = df[col].map(clean_text)
        out = out.where(out != "", vals)
    return out


def list_sheets(source: Any) -> List[str]:
    return pd.ExcelFile(source, engine="openpyxl").sheet_names


def read_papers(
    source: Any,
    sheet: Any = 0,
    overrides: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load a workbook into a canonical frame plus a report of what happened.

    Returns a DataFrame with the columns in OUTPUT_COLUMNS, and a report dict
    suitable for showing the user.
    """
    df = pd.read_excel(source, sheet_name=sheet, dtype=str, engine="openpyxl")
    df = df.where(pd.notna(df), "")
    report: Dict[str, Any] = {"rows_in_sheet": len(df), "sheet": sheet}

    mapping = detect_columns(list(df.columns))
    for field, col in (overrides or {}).items():
        if col and col in df.columns:
            mapping[field] = [col]

    missing = [f for f in REQUIRED_FIELDS if f not in mapping]
    if missing:
        raise ValueError(
            f"Could not find a column for: {missing}. "
            f"Columns in this sheet: {list(df.columns)[:25]}"
        )
    report["column_mapping"] = {k: v for k, v in mapping.items()}

    out = pd.DataFrame(index=df.index)
    for field in ["Article ID", "Title", "Abstract", "Journal", "DOI", "URL"]:
        out[field] = _coalesce(df, mapping[field]) if field in mapping else ""

    # CADIMA writes a second header row holding reviewer names; every content
    # column in it is blank, so it drops out cleanly here.
    content = out[["Article ID", "Title", "Abstract"]]
    blank_rows = (content == "").all(axis=1)
    report["blank_rows_dropped"] = int(blank_rows.sum())
    out, df = out[~blank_rows].copy(), df[~blank_rows].copy()

    # Year: keep the first 4-digit year found in any date-ish column.
    if "Year" in mapping:
        raw_year = _coalesce(df, mapping["Year"])
        out["Year"] = raw_year.map(
            lambda s: (_YEAR_RE.search(s).group(1) if _YEAR_RE.search(s) else "")
        )
    else:
        out["Year"] = ""

    # PMID: CADIMA stores it as "34203238[pmid]".
    if "PMID" in mapping:
        raw_pmid = _coalesce(df, mapping["PMID"])
        out["PMID"] = raw_pmid.map(
            lambda s: (_PMID_RE.search(s).group(1) if _PMID_RE.search(s) else "")
        )
    else:
        out["PMID"] = ""

    out = out.reset_index(drop=True)

    # Give every row a stable, unique identifier - it is the resume key.
    ids = out["Article ID"].astype(str).map(clean_text)
    blank_ids = ids == ""
    report["ids_synthesized"] = int(blank_ids.sum())
    ids = ids.mask(blank_ids, pd.Series([f"row-{i + 1}" for i in out.index],
                                        index=out.index))
    duplicated = ids.duplicated(keep=False)
    report["duplicate_ids"] = int(duplicated.sum())
    if duplicated.any():
        counter: Dict[str, int] = {}
        fixed = []
        for value, is_dup in zip(ids, duplicated):
            if is_dup:
                counter[value] = counter.get(value, 0) + 1
                fixed.append(f"{value}#{counter[value]}")
            else:
                fixed.append(value)
        ids = pd.Series(fixed, index=out.index)
    out["Article ID"] = ids

    out = out[OUTPUT_COLUMNS]

    report["rows_loaded"] = len(out)
    report["abstracts_present"] = int((out["Abstract"].str.len() >= 40).sum())
    report["abstracts_unusable"] = int(len(out) - report["abstracts_present"])
    return out, report
