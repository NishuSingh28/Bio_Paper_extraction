"""Paper link -> full text, with an abstract fallback that never skips a paper.

The user hands over links in whatever shape they have them: PMIDs, DOIs,
PubMed URLs, PMC URLs, bare identifiers. This module turns any of those into a
parsed document with labelled blocks (chunking.py), and records how it got
there. Chunking and embedding happen in the paper cache (cache.py).

Every paper ends up tagged `full_text` or `abstract_only`. A paper is never
dropped for want of full text, because a run must produce a value for every
column of every paper the user submitted.

Raw XML and identifier lookups are kept in the paper cache. A second run over
the same papers makes no network request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

import requests

import chunking
from normalize import clean_text

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

TOOL_NAME = "column_extractor"
DEFAULT_TIMEOUT = 45
IDCONV_BATCH = 180


# --------------------------------------------------------------------------- #
# Identifier parsing
# --------------------------------------------------------------------------- #
_PMCID_RE = re.compile(r"\bPMC(\d{5,9})\b", re.I)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>,;]+)", re.I)
_PMID_RE = re.compile(r"\b(\d{7,8})\b")
_PMID_LABEL_RE = re.compile(r"pmid[:\s/=]*(\d{1,9})", re.I)
_PUBMED_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{1,9})", re.I)
_URL_RE = re.compile(r"https?://\S+", re.I)


def parse_reference(text: str) -> Dict[str, str]:
    """Pull whatever identifiers a line of user input contains."""
    raw = clean_text(text)
    ids: Dict[str, str] = {}
    if not raw:
        return ids

    match = _PMCID_RE.search(raw)
    if match:
        ids["pmcid"] = f"PMC{match.group(1)}"

    match = _PUBMED_URL_RE.search(raw)
    if match:
        ids["pmid"] = match.group(1)
    else:
        match = _PMID_LABEL_RE.search(raw)
        if match:
            ids["pmid"] = match.group(1)

    match = _DOI_RE.search(raw)
    if match:
        # A DOI copied out of a URL often keeps a trailing bracket or period.
        ids["doi"] = match.group(1).rstrip(").,;'\"").lower()

    match = _URL_RE.search(raw)
    if match:
        ids["url"] = match.group(0).rstrip(").,;'\"")

    if not ids:
        # A bare number on its own line is a PMID.
        match = _PMID_RE.fullmatch(raw.strip())
        if match:
            ids["pmid"] = match.group(1)
    elif "pmid" not in ids and "pmcid" not in ids and "doi" not in ids:
        match = _PMID_RE.search(raw)
        if match and "url" in ids and match.group(1) in ids["url"]:
            pass  # a number inside an unrelated URL is not a PMID
    return ids


def canonical_key(ids: Dict[str, str], fallback: str = "") -> str:
    """Stable per-paper key. The same paper always hashes the same way."""
    for field_name in ("pmcid", "pmid", "doi", "url"):
        value = clean_text(ids.get(field_name, "")).lower()
        if value:
            return hashlib.sha1(
                f"{field_name}:{value}".encode("utf-8")).hexdigest()[:16]
    return hashlib.sha1(
        f"raw:{clean_text(fallback).lower()}".encode("utf-8")).hexdigest()[:16]


@dataclass
class PaperRef:
    """One paper the user asked for, before it is fetched."""

    raw: str = ""
    ids: Dict[str, str] = field(default_factory=dict)
    title: str = ""
    abstract: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    key: str = ""

    def __post_init__(self) -> None:
        self.ids = {k: clean_text(v) for k, v in (self.ids or {}).items()
                    if clean_text(v)}
        if not self.key:
            self.key = canonical_key(self.ids, self.raw or self.title)

    @property
    def label(self) -> str:
        return (self.ids.get("pmid") or self.ids.get("pmcid")
                or self.ids.get("doi") or self.ids.get("url")
                or self.title[:60] or self.key)


def refs_from_lines(lines: Iterable[str]) -> List[PaperRef]:
    """Build references from pasted or uploaded links, dropping duplicates."""
    refs: List[PaperRef] = []
    seen: set[str] = set()
    for line in lines:
        text = clean_text(line)
        if not text or text.startswith("#"):
            continue
        ids = parse_reference(text)
        ref = PaperRef(raw=text, ids=ids)
        if ref.key in seen:
            continue
        seen.add(ref.key)
        refs.append(ref)
    return refs


def refs_from_records(records: Sequence[Dict[str, Any]]) -> List[PaperRef]:
    """Build references from workbook rows produced by excel_io.read_papers."""
    refs: List[PaperRef] = []
    seen: set[str] = set()
    for record in records:
        ids: Dict[str, str] = {}
        for source, target in (("PMID", "pmid"), ("DOI", "doi"),
                               ("URL", "url")):
            value = clean_text(record.get(source, ""))
            if value:
                ids[target] = value.lower() if target == "doi" else value
        ref = PaperRef(
            raw=str(record.get("Article ID", "")),
            ids=ids,
            title=clean_text(record.get("Title", "")),
            abstract=clean_text(record.get("Abstract", "")),
            extra={
                "article_id": str(record.get("Article ID", "")),
                "year": clean_text(record.get("Year", "")),
                "journal": clean_text(record.get("Journal", "")),
            },
        )
        if not ids:
            # No identifier at all: key on the workbook row so resume still
            # works, and the abstract in the sheet is all we will have.
            ref.key = canonical_key({}, f"{record.get('Article ID')}|{ref.title}")
        if ref.key in seen:
            continue
        seen.add(ref.key)
        refs.append(ref)
    return refs


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Token bucket. NCBI allows 3 requests per second without an API key."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / max(0.1, per_second)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                delay = self._next - now
            else:
                delay = 0.0
                self._next = now
            self._next += self.interval
        if delay > 0:
            time.sleep(delay)


# --------------------------------------------------------------------------- #
# Fetcher
# --------------------------------------------------------------------------- #
class RawCache(Protocol):
    """Where downloads are kept. cache.RawStore is the implementation."""

    stats: Any

    def get(self, name: str) -> Optional[str]: ...
    def put(self, name: str, text: str) -> None: ...
    def put_miss(self, name: str, error: str = "") -> None: ...


class IdCache(Protocol):
    """Where identifier lookups are kept. cache.IdMapStore is the one."""

    def get(self, source_id: str) -> Optional[Dict[str, str]]: ...
    def put(self, source_id: str, mapping: Dict[str, str]) -> None: ...


class Fetcher:
    """Resolves identifiers and downloads papers, through the paper cache."""

    def __init__(
        self,
        raw_cache: RawCache,
        id_cache: Optional[IdCache] = None,
        *,
        email: str = "",
        api_key: str = "",
        requests_per_second: Optional[float] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_attempts: int = 3,
        allow_network: bool = True,
    ):
        self.raw_cache = raw_cache
        self.id_cache = id_cache
        self.email = email.strip()
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.allow_network = allow_network
        default_rate = 9.0 if self.api_key else 2.8
        self.limiter = RateLimiter(requests_per_second or default_rate)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = f"{TOOL_NAME}/1.0"
        self._resolved: Dict[str, Dict[str, str]] = {}
        self._resolve_lock = threading.Lock()

    # ----------------------------------------------------------------- #
    def _params(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        params = {"tool": TOOL_NAME}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        params.update(extra)
        return params

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None,
             *, attempts: Optional[int] = None) -> Optional[str]:
        """GET with rate limiting and exponential backoff. None on failure."""
        if not self.allow_network:
            return None
        last_error: Optional[str] = None
        max_attempts = attempts if attempts is not None else self.max_attempts
        for attempt in range(1, max_attempts + 1):
            self.limiter.wait()
            stats = getattr(self.raw_cache, "stats", None)
            if stats is not None:
                stats.add("network_requests")
            try:
                response = self._session.get(url, params=params,
                                             timeout=self.timeout)
                if response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    raise requests.RequestException(last_error)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                last_error = str(exc)
                if attempt < max_attempts:
                    # Jitter keeps a wide pool of workers from retrying in step.
                    delay = min(30.0, 1.5 ** attempt) * (0.7 + 0.6 * (attempt % 3) / 2)
                    time.sleep(delay)
        log.debug("GET %s failed after %d attempts: %s", url, max_attempts,
                  last_error)
        return None

    # ----------------------------------------------------------------- #
    def _cached(self, name: str) -> Optional[str]:
        return self.raw_cache.get(name)

    def _store(self, name: str, text: str) -> None:
        try:
            if text:
                self.raw_cache.put(name, text)
            else:
                # Remember the miss for a while; do not ask again at once.
                self.raw_cache.put_miss(name)
        except OSError as exc:
            log.warning("Could not cache %s: %s", name, exc)

    # ----------------------------------------------------------------- #
    def resolve_batch(self, refs: Sequence[PaperRef]) -> None:
        """Fill in PMCIDs for references that only carry a PMID or a DOI.

        Done in batches of ~180 because the converter accepts a list, and one
        request for 180 papers instead of 180 requests is the difference
        between minutes and hours at scale.
        """
        # The converter rejects a batch that mixes identifier types, so PMIDs
        # and DOIs are sent separately.
        by_type: Dict[str, List[str]] = {"pmid": [], "doi": []}
        for ref in refs:
            if ref.ids.get("pmcid"):
                continue
            for kind in ("pmid", "doi"):
                token = ref.ids.get(kind)
                if not token or token in self._resolved:
                    continue
                stored = (self.id_cache.get(token)
                          if self.id_cache is not None else None)
                if stored is not None:
                    with self._resolve_lock:
                        self._resolved[token] = stored
                    break
                by_type[kind].append(token)
                break

        batches: List[List[str]] = []
        for tokens in by_type.values():
            tokens = list(dict.fromkeys(tokens))
            batches.extend(tokens[i:i + IDCONV_BATCH]
                           for i in range(0, len(tokens), IDCONV_BATCH))

        for batch in batches:
            text = self._get(IDCONV, self._params(
                {"format": "json", "ids": ",".join(batch)}))
            if not text:
                continue
            try:
                records = json.loads(text).get(
                    "records", [])
            except ValueError:
                continue
            answered: set[str] = set()
            with self._resolve_lock:
                for record in records:
                    requested = str(record.get("requested-id", "")).strip()
                    mapped = {
                        k: str(record[k]) for k in ("pmcid", "pmid", "doi")
                        if record.get(k)
                    }
                    if requested:
                        self._resolved[requested] = mapped
                        answered.add(requested)
                    if mapped.get("doi"):
                        self._resolved.setdefault(mapped["doi"].lower(), mapped)
            if self.id_cache is not None:
                # A token the converter did not answer has no PMC record. It
                # is stored as such, so a rerun does not ask again.
                for token in batch:
                    mapping = self._resolved.get(token, {})
                    if token in answered or not mapping:
                        self.id_cache.put(token, mapping)

        for ref in refs:
            token = ref.ids.get("pmid") or ref.ids.get("doi")
            mapped = self._resolved.get(token or "")
            if mapped:
                for key, value in mapped.items():
                    ref.ids.setdefault(key, value)

    # ----------------------------------------------------------------- #
    def _fetch_pmc_xml(self, pmcid: str) -> Optional[str]:
        cached = self._cached(f"pmc_{pmcid}")
        if cached is not None:
            return cached or None
        text = self._get(f"{EUTILS}/efetch.fcgi", self._params(
            {"db": "pmc", "id": pmcid, "retmode": "xml"}))
        if text and "<body" in text:
            self._store(f"pmc_{pmcid}", text)
            return text
        # PMC holds a record but no open full text. Europe PMC sometimes does.
        # It answers 500 rather than 404 for an article it does not hold, so
        # this asks once: retrying a miss costs seconds and finds nothing.
        epmc = self._get(f"{EPMC}/{pmcid}/fullTextXML", attempts=1)
        if epmc and "<body" in epmc:
            self._store(f"pmc_{pmcid}", epmc)
            return epmc
        self._store(f"pmc_{pmcid}", "")  # remember the miss; do not refetch
        return None

    def _fetch_pubmed_xml(self, pmid: str) -> Optional[str]:
        cached = self._cached(f"pubmed_{pmid}")
        if cached is not None:
            return cached or None
        text = self._get(f"{EUTILS}/efetch.fcgi", self._params(
            {"db": "pubmed", "id": pmid, "retmode": "xml"}))
        if text and "<PubmedArticle" in text:
            self._store(f"pubmed_{pmid}", text)
            return text
        self._store(f"pubmed_{pmid}", "")
        return None

    def _fetch_epmc_record(self, ref: PaperRef) -> Optional[Dict[str, Any]]:
        """Last resort for a DOI with no PubMed record: Europe PMC search."""
        doi = ref.ids.get("doi")
        if not doi:
            return None
        cached = self._cached(f"epmc_{hashlib.sha1(doi.encode()).hexdigest()[:16]}")
        if cached == "":
            return None
        text = cached or self._get(f"{EPMC}/search", {
            "query": f'DOI:"{doi}"', "resultType": "core", "format": "json",
            "pageSize": 1,
        })
        if not text:
            return None
        try:
            results = json.loads(text)
        except ValueError:
            return None
        hits = (results.get("resultList") or {}).get("result") or []
        if not hits:
            self._store(f"epmc_{hashlib.sha1(doi.encode()).hexdigest()[:16]}", "")
            return None
        if cached is None:
            self._store(f"epmc_{hashlib.sha1(doi.encode()).hexdigest()[:16]}", text)
        return hits[0]

    # ----------------------------------------------------------------- #
    def fetch(self, ref: PaperRef) -> Dict[str, Any]:
        """Return a parsed document for one paper. Always returns a document.

        The document holds `blocks`, not chunks. The paper cache chunks it.
        """
        notes: List[str] = []
        document: Optional[Dict[str, Any]] = None

        if not ref.ids.get("pmcid") and (ref.ids.get("pmid") or ref.ids.get("doi")):
            self.resolve_batch([ref])

        pmcid = ref.ids.get("pmcid")
        if pmcid:
            xml = self._fetch_pmc_xml(pmcid)
            if xml:
                document = chunking.parse_pmc_article(xml)
                if document is None:
                    notes.append("PMC XML could not be parsed")
            else:
                notes.append("no open-access full text in PMC")

        if document is None and ref.ids.get("pmid"):
            xml = self._fetch_pubmed_xml(ref.ids["pmid"])
            if xml:
                document = chunking.parse_pubmed_article(xml)
                if document is None:
                    notes.append("PubMed XML could not be parsed")

        if document is None:
            record = self._fetch_epmc_record(ref)
            if record:
                ids = {
                    k: str(record[v]) for k, v in
                    (("pmid", "pmid"), ("pmcid", "pmcid"), ("doi", "doi"))
                    if record.get(v)
                }
                document = chunking.document_from_text(
                    title=record.get("title", "") or ref.title,
                    abstract=record.get("abstractText", "") or ref.abstract,
                    ids={**ref.ids, **ids},
                )
                document["journal"] = record.get("journalTitle", "") or ""
                document["year"] = str(record.get("pubYear", "") or "")
                document["source_api"] = "europepmc"
                notes.append("abstract from Europe PMC")

        if document is None:
            document = chunking.document_from_text(
                title=ref.title, abstract=ref.abstract, ids=ref.ids)
            if ref.abstract:
                notes.append("abstract supplied with the input")
            else:
                notes.append("no text found for this paper")

        # Keep identifiers and bibliographic fields the source supplied but the
        # fetched record lacks.
        merged_ids = {**ref.ids}
        merged_ids.update({k: v for k, v in (document.get("ids") or {}).items()
                           if v})
        document["ids"] = merged_ids
        for key in ("title", "year", "journal"):
            if not clean_text(document.get(key, "")):
                document[key] = clean_text(
                    ref.extra.get(key, "") or getattr(ref, key, ""))
        if not clean_text(document.get("abstract", "")) and ref.abstract:
            document["abstract"] = ref.abstract
            if not document.get("blocks"):
                document["blocks"] = [{
                    "section": "abstract", "title": "Abstract",
                    "kind": "text", "text": ref.abstract,
                }]

        document["fetch_notes"] = "; ".join(notes)
        if not any(clean_text(b.get("text", ""))
                   for b in document.get("blocks", [])):
            document["source_type"] = "no_text"
        return document
