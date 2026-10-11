"""Route user-defined columns into groups, one group per LLM call.

The whole point is that this works for columns nobody anticipated. Nothing here
keys on a column name. A column is routed by two generic signals:

1. cue-term overlap against a vocabulary that describes paper *sections*
2. cosine similarity between the column's text and a paragraph describing each
   section

Some columns need no model at all: a title, a DOI, a funding statement are
already in the source record or in the PMC XML. Those are answered directly.

The section a group is routed to is a retrieval *prior*, never a filter
(see index.py). A mis-routed column still reaches its evidence, so a routing
mistake costs rank, not recall.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import sections as S
from columns import ColumnSpec
from embedder import Embedder, get_embedder

log = logging.getLogger(__name__)

MAX_COLUMNS_PER_CALL = 8
MIN_COLUMNS_PER_CALL = 3


# --------------------------------------------------------------------------- #
# Metadata columns - answered without a model call
# --------------------------------------------------------------------------- #
def _first_nonempty(*values: Any) -> str:
    for value in values:
        if isinstance(value, (list, tuple)):
            joined = "; ".join(str(v).strip() for v in value if str(v).strip())
            if joined:
                return joined
        elif value not in (None, ""):
            text = str(value).strip()
            if text:
                return text
    return ""


# Canonical metadata field -> (aliases the user might type, resolver).
# Matching is exact on the normalized column name. It is deliberately strict:
# a false metadata match means a column never reaches the model at all.
MetaResolver = Callable[[Dict[str, Any]], str]

METADATA_FIELDS: Dict[str, Tuple[List[str], MetaResolver]] = {
    "title": (
        ["title", "paper title", "article title", "study title"],
        lambda d: _first_nonempty(d.get("title")),
    ),
    "year": (
        ["year", "publication year", "year published", "pub year",
         "year of publication"],
        lambda d: _first_nonempty(d.get("year")),
    ),
    "journal": (
        ["journal", "journal name", "source", "source title", "periodical"],
        lambda d: _first_nonempty(d.get("journal")),
    ),
    "doi": (["doi"], lambda d: _first_nonempty((d.get("ids") or {}).get("doi"))),
    "pmid": (
        ["pmid", "pubmed id", "pubmed identifier"],
        lambda d: _first_nonempty((d.get("ids") or {}).get("pmid")),
    ),
    "pmcid": (
        ["pmcid", "pmc id"],
        lambda d: _first_nonempty((d.get("ids") or {}).get("pmcid")),
    ),
    "url": (
        ["url", "link", "paper url", "paper link", "web address"],
        lambda d: _first_nonempty((d.get("ids") or {}).get("url")),
    ),
    "authors": (
        ["authors", "author list", "author names", "all authors"],
        lambda d: _first_nonempty(d.get("authors")),
    ),
    "first_author": (
        ["first author", "lead author", "corresponding author"],
        lambda d: _first_nonempty((d.get("authors") or [""])[0]),
    ),
    "article_type": (
        ["article type", "publication type", "paper type", "document type"],
        lambda d: _first_nonempty(d.get("article_type")),
    ),
    "language": (["language"], lambda d: _first_nonempty(d.get("language"))),
    "volume": (["volume"], lambda d: _first_nonempty(d.get("volume"))),
    "issue": (["issue"], lambda d: _first_nonempty(d.get("issue"))),
    "pages": (
        ["pages", "page range", "page numbers"],
        lambda d: _first_nonempty(d.get("pages")),
    ),
    "keywords": (
        ["keywords", "key words", "index terms"],
        lambda d: _first_nonempty(d.get("keywords")),
    ),
    "funding_source": (
        ["funding", "funding source", "funder", "funding body",
         "funding statement", "grant", "grant source", "financial support",
         "sponsor", "funding agency"],
        lambda d: _first_nonempty(d.get("funding_statement"),
                                  d.get("award_sources")),
    ),
    "conflict_of_interest": (
        ["conflict of interest", "competing interests", "coi",
         "conflicts of interest", "declaration of interest"],
        lambda d: _first_nonempty(d.get("conflict_statement")),
    ),
    "ethics_approval": (
        ["ethics", "ethics approval", "ethical approval", "irb approval",
         "ethics statement", "iacuc"],
        lambda d: _first_nonempty(d.get("ethics_statement")),
    ),
    "data_availability": (
        ["data availability", "data availability statement",
         "data sharing", "accession numbers"],
        lambda d: _first_nonempty(d.get("data_availability")),
    ),
    "affiliation": (
        ["affiliation", "affiliations", "institution", "institutions"],
        lambda d: _first_nonempty(d.get("affiliations")),
    ),
    "abstract": (
        ["abstract", "abstract text", "summary text"],
        lambda d: _first_nonempty(d.get("abstract")),
    ),
}

_ALIAS_TO_FIELD: Dict[str, str] = {}
for _field_name, (_aliases, _) in METADATA_FIELDS.items():
    for _alias in _aliases:
        _ALIAS_TO_FIELD[_alias] = _field_name

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NORM_RE.sub(" ", str(text or "").lower()).strip()


def metadata_field_for(spec: ColumnSpec) -> Optional[str]:
    """Return the metadata field this column asks for, or None."""
    if spec.section_hint == "metadata":
        return _ALIAS_TO_FIELD.get(_norm(spec.name))
    if spec.section_hint != "auto":
        return None
    return _ALIAS_TO_FIELD.get(_norm(spec.name))


def resolve_metadata(field_name: str, doc: Dict[str, Any]) -> str:
    """Pull one metadata field straight out of an ingested paper."""
    entry = METADATA_FIELDS.get(field_name)
    if not entry:
        return ""
    try:
        return entry[1](doc) or ""
    except Exception as exc:  # noqa: BLE001 - a malformed doc must not stop a run
        log.debug("Metadata resolver %s failed: %s", field_name, exc)
        return ""


# --------------------------------------------------------------------------- #
# Affinity scoring
# --------------------------------------------------------------------------- #
def _cue_scores(text: str) -> Dict[str, float]:
    """Weighted cue-term overlap per group. A longer phrase counts for more."""
    lowered = f" {_norm(text)} "
    scores: Dict[str, float] = {}
    for group, cues in S.GROUP_CUES.items():
        total = 0.0
        for cue in cues:
            needle = _norm(cue)
            # Match whole words only. A bare substring test lets a one-letter
            # cue hit any word that contains it, which silently mis-routes a
            # column ("n" inside "kind").
            if needle and f" {needle} " in lowered:
                total += 1.0 + 0.5 * needle.count(" ")
        scores[group] = total
    return scores


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    values = list(scores.values())
    if not values:
        return scores
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


@dataclass
class Routing:
    """Why one column landed where it did. Shown in the UI."""

    key: str
    name: str
    group: str
    score: float
    margin: float
    method: str  # "hint" | "metadata" | "cues+embedding" | "cues" | "default"
    scores: Dict[str, float] = field(default_factory=dict)


class ColumnRouter:
    """Scores columns against section prototypes. Embeds prototypes once."""

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder if embedder is not None else get_embedder()
        self.proto_matrix = np.zeros((0, 0), dtype=np.float32)
        self.proto_groups: List[str] = []
        if self.embedder.available:
            self.proto_groups = list(S.GROUP_PROTOTYPES)
            self.proto_matrix = self.embedder.encode(
                [S.GROUP_PROTOTYPES[g] for g in self.proto_groups]
            )

    def score(self, spec: ColumnSpec) -> Routing:
        text = spec.retrieval_text()
        cue_raw = _cue_scores(text)
        cue_norm = _normalize(cue_raw)

        emb_norm: Dict[str, float] = {}
        if self.proto_matrix.size:
            vector = self.embedder.encode([text])[0]
            # An empty vector means the model failed to load after the
            # prototypes came from the cache; fall back to the cues.
            if vector.size == self.proto_matrix.shape[1]:
                cosines = self.proto_matrix @ vector
                emb_norm = _normalize(
                    {g: float(c) for g, c in zip(self.proto_groups, cosines)}
                )

        if emb_norm:
            combined = {
                g: 0.55 * emb_norm.get(g, 0.0) + 0.45 * cue_norm.get(g, 0.0)
                for g in S.LLM_GROUPS
            }
            method = "cues+embedding"
        else:
            combined = {g: cue_norm.get(g, 0.0) for g in S.LLM_GROUPS}
            method = "cues"

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        top_group, top_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0

        # Every signal flat means neither the cues nor the prototypes
        # recognised the column. "General" retrieves from the abstract and the
        # summary sections, which is the right place to look when in doubt.
        if top_score <= 1e-9 and max(cue_raw.values(), default=0.0) <= 0.0:
            top_group, method = S.GROUP_GENERAL, "default"

        return Routing(
            key=spec.key, name=spec.name, group=top_group,
            score=round(float(top_score), 4),
            margin=round(float(top_score - second), 4),
            method=method,
            scores={g: round(float(v), 4) for g, v in combined.items()},
        )


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #
@dataclass
class ColumnGroup:
    """One LLM call: the columns it covers and the sections it prefers."""

    key: str
    group_type: str
    columns: List[ColumnSpec]
    sections: List[str]

    @property
    def label(self) -> str:
        base = S.GROUP_LABELS.get(self.group_type, self.group_type)
        suffix = self.key[len(self.group_type):]
        return f"{base} ({suffix.lstrip('_')})" if suffix else base

    @property
    def is_metadata(self) -> bool:
        return self.group_type == S.GROUP_METADATA

    def retrieval_query(self) -> str:
        return " ".join(c.retrieval_text() for c in self.columns)

    def keyword_terms(self) -> List[str]:
        """Terms for BM25. Column names carry the exact words that matter."""
        terms: List[str] = []
        for column in self.columns:
            terms.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/%]*",
                                    column.name))
            for value in column.allowed_values[:12]:
                terms.extend(re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/%]*", value))
            if column.unit:
                terms.append(column.unit)
        return terms


def _order_by_similarity(vectors: np.ndarray) -> List[int]:
    """Chain the columns so neighbours are the most alike.

    Start from the item furthest from the centroid, then repeatedly append the
    nearest unused item. Splitting this order into consecutive buckets keeps
    related columns in the same LLM call.
    """
    count = vectors.shape[0]
    if count <= 1 or vectors.shape[1] == 0:
        return list(range(count))
    centroid = vectors.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    start = int(np.argmin(vectors @ centroid))

    order = [start]
    remaining = set(range(count)) - {start}
    while remaining:
        last = vectors[order[-1]]
        nearest = max(remaining, key=lambda i: float(vectors[i] @ last))
        order.append(nearest)
        remaining.discard(nearest)
    return order


def _split_group(specs: List[ColumnSpec], vectors: np.ndarray,
                 max_size: int) -> List[List[ColumnSpec]]:
    """Split an oversized group into balanced, semantically coherent buckets."""
    if len(specs) <= max_size:
        return [specs]
    order = _order_by_similarity(vectors)
    ordered = [specs[i] for i in order]
    bucket_count = math.ceil(len(ordered) / max_size)
    size = math.ceil(len(ordered) / bucket_count)
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def _merge_small_groups(groups: List[ColumnGroup], router: ColumnRouter,
                        max_size: int, min_size: int) -> List[ColumnGroup]:
    """Fold a tiny group into its nearest neighbour, to save an LLM call.

    The merged group keeps both section preferences, so the absorbed columns do
    not lose their retrieval prior.
    """
    if len(groups) < 2:
        return groups

    proto_index = {g: i for i, g in enumerate(router.proto_groups)}
    have_protos = router.proto_matrix.size > 0

    def similarity(a: str, b: str) -> float:
        if not have_protos or a not in proto_index or b not in proto_index:
            return 0.0
        return float(router.proto_matrix[proto_index[a]]
                     @ router.proto_matrix[proto_index[b]])

    changed = True
    while changed:
        changed = False
        groups.sort(key=lambda g: len(g.columns))
        for i, small in enumerate(groups):
            if len(small.columns) >= min_size:
                break
            candidates = [
                (similarity(small.group_type, other.group_type), j)
                for j, other in enumerate(groups)
                if j != i and len(other.columns) + len(small.columns) <= max_size
            ]
            if not candidates:
                continue
            _, j = max(candidates, key=lambda t: t[0])
            target = groups[j]
            target.columns.extend(small.columns)
            merged_sections = list(target.sections)
            for section in small.sections:
                if section not in merged_sections:
                    merged_sections.append(section)
            target.sections = merged_sections
            groups.pop(i)
            changed = True
            break
    return groups


def plan_groups(
    specs: Sequence[ColumnSpec],
    *,
    max_per_call: int = MAX_COLUMNS_PER_CALL,
    min_per_call: int = MIN_COLUMNS_PER_CALL,
    router: Optional[ColumnRouter] = None,
) -> Tuple[List[ColumnGroup], List[Routing]]:
    """Group columns for extraction. Returns (groups, per-column routing)."""
    specs = list(specs)
    if not specs:
        return [], []

    router = router or ColumnRouter()
    routings: List[Routing] = []
    metadata_specs: List[ColumnSpec] = []
    by_group: Dict[str, List[ColumnSpec]] = {g: [] for g in S.LLM_GROUPS}

    for spec in specs:
        meta_field = metadata_field_for(spec)
        if meta_field is not None:
            metadata_specs.append(spec)
            routings.append(Routing(
                key=spec.key, name=spec.name, group=S.GROUP_METADATA,
                score=1.0, margin=1.0, method="metadata",
                scores={S.GROUP_METADATA: 1.0},
            ))
            continue

        if spec.section_hint not in ("auto", "metadata"):
            by_group[spec.section_hint].append(spec)
            routings.append(Routing(
                key=spec.key, name=spec.name, group=spec.section_hint,
                score=1.0, margin=1.0, method="hint",
                scores={spec.section_hint: 1.0},
            ))
            continue

        routing = router.score(spec)
        by_group[routing.group].append(spec)
        routings.append(routing)

    groups: List[ColumnGroup] = []
    if metadata_specs:
        groups.append(ColumnGroup(
            key=S.GROUP_METADATA, group_type=S.GROUP_METADATA,
            columns=metadata_specs,
            sections=S.GROUP_SECTIONS[S.GROUP_METADATA],
        ))

    llm_groups: List[ColumnGroup] = []
    for group_type in S.LLM_GROUPS:
        members = by_group.get(group_type) or []
        if not members:
            continue
        vectors = np.zeros((len(members), 0), dtype=np.float32)
        if router.embedder.available and len(members) > max_per_call:
            vectors = router.embedder.encode(
                [m.retrieval_text() for m in members])
        for index, bucket in enumerate(
                _split_group(members, vectors, max_per_call), start=1):
            key = group_type if index == 1 else f"{group_type}_{index}"
            llm_groups.append(ColumnGroup(
                key=key, group_type=group_type, columns=bucket,
                sections=list(S.GROUP_SECTIONS[group_type]),
            ))

    llm_groups = _merge_small_groups(llm_groups, router, max_per_call,
                                     min_per_call)
    # Keep a stable, readable order for the UI and the logs.
    order = {g: i for i, g in enumerate(S.LLM_GROUPS)}
    llm_groups.sort(key=lambda g: (order.get(g.group_type, 99), g.key))
    groups.extend(llm_groups)
    return groups, routings


def describe_plan(groups: Sequence[ColumnGroup]) -> List[Dict[str, Any]]:
    """A table of the plan, for the UI."""
    return [
        {
            "Group": group.label,
            "LLM call": "no" if group.is_metadata else "yes",
            "Columns": len(group.columns),
            "Column names": ", ".join(c.name for c in group.columns),
            "Sections searched first": ", ".join(group.sections[:3]),
        }
        for group in groups
    ]
