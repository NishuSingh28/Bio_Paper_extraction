"""Turn a paper into labelled, retrievable chunks.

Structure first, size second. A paper is split by its own headings - abstract,
methods, results, discussion, funding - because a section label is the single
most useful retrieval signal a paper gives away for free. Only then is a long
section cut into overlapping passages of a few hundred tokens.

Tables get their own chunks. Sample sizes, counts and outcome numbers live in
result tables far more often than in the running text, and a flattened table row
is what a keyword search can actually match.

Reference lists are dropped. They are large, they match every query, and they
answer nothing.

Parsing and chunking are two steps. A parser returns a document with labelled
`blocks` (one per section, table or figure). `chunk_document` then cuts the
blocks into chunks. The paper cache stores the two results as separate layers,
so a change to the chunk size never causes a new download or a new parse.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET

import sections as S
from normalize import clean_text

log = logging.getLogger(__name__)

# Bump CHUNKER_VERSION when the chunking code changes its output. The version
# and the size parameters name the cache namespace, so chunks and vectors made
# by different chunkers are never mixed.
CHUNKER_VERSION = 3
TARGET_TOKENS = 400
OVERLAP_TOKENS = 80
MIN_CHUNK_TOKENS = 20
MAX_TABLE_CHARS = 4000


def chunker_namespace(target_tokens: int = TARGET_TOKENS,
                      overlap_tokens: int = OVERLAP_TOKENS,
                      min_chunk_tokens: int = MIN_CHUNK_TOKENS) -> str:
    """The cache namespace for chunks made with these parameters."""
    return (f"chunker-v{CHUNKER_VERSION}-t{target_tokens}"
            f"-o{overlap_tokens}-m{min_chunk_tokens}")

# Inline elements whose text is noise inside a sentence.
_SKIP_INLINE = {"xref", "label", "graphic", "inline-graphic", "media",
                "supplementary-material"}


def approx_tokens(text: str) -> int:
    """Rough token count. Four characters per token is close enough here.

    It is used for budgets, never for billing, so an exact tokenizer would add
    a dependency for no gain.
    """
    return max(1, len(text or "") // 4)


def _tag(element: ET.Element) -> str:
    """Local tag name, with any XML namespace stripped."""
    tag = element.tag
    if isinstance(tag, str) and "}" in tag:
        return tag.rsplit("}", 1)[1]
    return str(tag)


def _text_of(element: Optional[ET.Element], skip: Iterable[str] = ()) -> str:
    """Flatten an element to text, dropping citation markers and figures."""
    if element is None:
        return ""
    skip_set = set(skip) | _SKIP_INLINE
    parts: List[str] = []

    def walk(node: ET.Element) -> None:
        if _tag(node) in skip_set:
            # Keep the tail: "...as shown<xref>[3]</xref> in Table 1" must not
            # lose " in Table 1".
            if node.tail:
                parts.append(node.tail)
            return
        if node.text:
            parts.append(node.text)
        for child in list(node):
            walk(child)
        if node.tail:
            parts.append(node.tail)

    if _tag(element) in skip_set:
        return ""
    if element.text:
        parts.append(element.text)
    for child in list(element):
        walk(child)
    return clean_text("".join(parts))


def _findall_local(root: ET.Element, name: str) -> List[ET.Element]:
    """Find descendants by local tag name, namespace-insensitively."""
    return [e for e in root.iter() if _tag(e) == name]


def _children_local(root: Optional[ET.Element], name: str) -> List[ET.Element]:
    """Find direct children by local tag name. Do not search deeper."""
    if root is None:
        return []
    return [e for e in root if _tag(e) == name]


def _find_local(root: ET.Element, name: str) -> Optional[ET.Element]:
    for element in root.iter():
        if _tag(element) == name:
            return element
    return None


# --------------------------------------------------------------------------- #
# Sentence-aware splitting
# --------------------------------------------------------------------------- #
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    raw = _SENTENCE_RE.split(text)
    # Abbreviations such as "et al." and "Fig. 2" split badly; rejoin a
    # fragment that is too short to be a sentence.
    out: List[str] = []
    for piece in raw:
        piece = piece.strip()
        if not piece:
            continue
        if out and (len(piece) < 25 or len(out[-1]) < 25):
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def split_passage(text: str, target_tokens: int = TARGET_TOKENS,
                  overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
    """Cut a long passage into overlapping windows, on sentence boundaries."""
    text = clean_text(text)
    if not text:
        return []
    if approx_tokens(text) <= target_tokens:
        return [text]

    sentences = split_sentences(text)
    if not sentences:
        return [text]

    windows: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = approx_tokens(sentence)
        if current and current_tokens + sentence_tokens > target_tokens:
            windows.append(" ".join(current))
            # Carry the tail of the window forward, so a fact that straddles
            # the cut survives in at least one chunk whole.
            carry: List[str] = []
            carried = 0
            for previous in reversed(current):
                previous_tokens = approx_tokens(previous)
                if carried + previous_tokens > overlap_tokens:
                    break
                carry.insert(0, previous)
                carried += previous_tokens
            current, current_tokens = list(carry), carried
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        windows.append(" ".join(current))
    return [w for w in windows if w.strip()]


# --------------------------------------------------------------------------- #
# Chunk assembly
# --------------------------------------------------------------------------- #
def _locate(text: str, piece: str, start: int) -> tuple[int, int]:
    """Character span of `piece` inside `text`, searching from `start`.

    Returns (-1, -1) when the piece is not a verbatim substring, which can only
    happen if sentence splitting changed the whitespace.
    """
    position = text.find(piece, max(0, start))
    if position < 0:
        position = text.find(piece)
    if position < 0:
        return -1, -1
    return position, position + len(piece)


def make_chunks(blocks: List[Dict[str, Any]],
                target_tokens: int = TARGET_TOKENS,
                overlap_tokens: int = OVERLAP_TOKENS,
                min_chunk_tokens: int = MIN_CHUNK_TOKENS,
                ) -> List[Dict[str, Any]]:
    """Turn (section, title, text, kind) blocks into numbered chunks.

    Each chunk records the block it came from and its character span inside
    that block's cleaned text, so a value can be traced back to the source.
    A very short tail of the next block of the same section is appended to the
    chunk before it; the span then still covers the first block only.
    """
    chunks: List[Dict[str, Any]] = []
    for block_index, block in enumerate(blocks):
        text = clean_text(block.get("text", ""))
        if not text:
            continue
        kind = block.get("kind", "text")
        section = block.get("section", S.OTHER)
        title = clean_text(block.get("title", ""))
        pieces = ([text] if kind == "table"
                  else split_passage(text, target_tokens, overlap_tokens))
        cursor = 0
        for piece in pieces:
            start, end = _locate(text, piece, cursor)
            if start >= 0:
                cursor = start + 1
            if approx_tokens(piece) < min_chunk_tokens and kind == "text":
                # Too short to answer anything on its own; append to the
                # previous chunk of the same section rather than lose it. The
                # span grows only when the text comes from the same block.
                previous = chunks[-1] if chunks else None
                if previous and previous["section"] == section:
                    previous["text"] = f"{previous['text']} {piece}"
                    if (previous["block"] == block_index and end >= 0
                            and previous["char_start"] >= 0):
                        previous["char_end"] = max(previous["char_end"], end)
                    continue
            chunks.append({
                "id": f"c{len(chunks) + 1}",
                "section": section,
                "title": title,
                "kind": kind,
                "text": piece,
                "tokens": approx_tokens(piece),
                "block": block_index,
                "char_start": start,
                "char_end": end,
            })
    return chunks


def content_hash(document: Dict[str, Any]) -> str:
    """SHA-256 of the parsed text a paper's chunks are made from.

    A new download that yields the same text keeps the hash, so the chunks and
    vectors stay valid. Changed text gives a new hash, and the chunks and
    vectors made from the old text are not used.
    """
    payload = json.dumps(
        {"title": document.get("title", ""),
         "blocks": [[b.get("section", ""), b.get("title", ""),
                     b.get("kind", ""), clean_text(b.get("text", ""))]
                    for b in document.get("blocks", [])]},
        ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chunk_document(document: Dict[str, Any], paper_key: str = "",
                   target_tokens: int = TARGET_TOKENS,
                   overlap_tokens: int = OVERLAP_TOKENS,
                   min_chunk_tokens: int = MIN_CHUNK_TOKENS,
                   ) -> List[Dict[str, Any]]:
    """Chunk a parsed document and give each chunk a globally unique id.

    `id` ("c1", "c2", ...) is short, because the model cites it. `chunk_id`
    ("<paper>:<content hash>:<n>") is unique across the whole cache, and it is
    the key the vectors are stored under.
    """
    digest = document.get("content_hash") or content_hash(document)
    key = paper_key or document.get("paper_key", "")
    chunks = make_chunks(document.get("blocks", []), target_tokens,
                         overlap_tokens, min_chunk_tokens)
    for number, chunk in enumerate(chunks, start=1):
        chunk["chunk_id"] = f"{key}:{digest[:12]}:{number}"
    return chunks


# --------------------------------------------------------------------------- #
# PMC / JATS full text
# --------------------------------------------------------------------------- #
def _collect_sections(parent: ET.Element, inherited: str,
                      blocks: List[Dict[str, Any]], depth: int = 0) -> None:
    """Walk a <sec> tree, labelling each level and flattening its paragraphs."""
    title_element = None
    for child in list(parent):
        if _tag(child) == "title":
            title_element = child
            break
    title = _text_of(title_element)
    sec_type = parent.get("sec-type", "") if _tag(parent) == "sec" else ""
    label = S.label_section(title, sec_type)
    if label == S.OTHER and inherited != S.OTHER:
        # An unnamed subsection belongs to its parent's section.
        label = inherited

    paragraphs: List[str] = []
    for child in list(parent):
        tag = _tag(child)
        if tag == "p":
            paragraphs.append(_text_of(child))
        elif tag in {"list", "def-list"}:
            paragraphs.append(_text_of(child))
        elif tag == "table-wrap":
            blocks.append({
                "section": S.TABLE, "title": title or "Table",
                "kind": "table", "text": _flatten_table(child),
            })
        elif tag == "fig":
            caption = _text_of(_find_local(child, "caption"))
            if caption:
                blocks.append({
                    "section": S.FIGURE, "title": title or "Figure",
                    "kind": "figure", "text": caption,
                })
        elif tag == "sec":
            if paragraphs:
                blocks.append({"section": label, "title": title,
                               "kind": "text", "text": " ".join(paragraphs)})
                paragraphs = []
            _collect_sections(child, label, blocks, depth + 1)

    if paragraphs:
        blocks.append({"section": label, "title": title, "kind": "text",
                       "text": " ".join(paragraphs)})


def _flatten_table(table_wrap: ET.Element) -> str:
    """Render a table as caption plus pipe-separated rows."""
    parts: List[str] = []
    label = _text_of(_find_local(table_wrap, "label"))
    caption = _text_of(_find_local(table_wrap, "caption"))
    header = " ".join(p for p in (label, caption) if p)
    if header:
        parts.append(header)

    for row in _findall_local(table_wrap, "tr"):
        cells = [
            _text_of(cell)
            for cell in list(row)
            if _tag(cell) in {"td", "th"}
        ]
        line = " | ".join(c for c in cells if c is not None)
        if line.strip(" |"):
            parts.append(line)

    footer = _text_of(_find_local(table_wrap, "table-wrap-foot"))
    if footer:
        parts.append(footer)
    return clean_text(" \n".join(parts))[:MAX_TABLE_CHARS]


def _article_ids(article: ET.Element) -> Dict[str, str]:
    """The IDs of this article only.

    Read only the first <article-meta>. A sub-article, a related article or a
    reference can also hold an <article-id>, and that ID is not this paper.
    """
    ids: Dict[str, str] = {}
    for element in _children_local(_find_local(article, "article-meta"),
                                   "article-id"):
        kind = (element.get("pub-id-type") or "").lower()
        value = clean_text(element.text)
        if not value:
            continue
        if kind == "pmid":
            ids["pmid"] = value
        elif kind == "doi":
            ids["doi"] = value
        elif kind in {"pmc", "pmcid"}:
            ids["pmcid"] = value if value.upper().startswith("PMC") else f"PMC{value}"
    return ids


def _authors(article: ET.Element) -> List[str]:
    names: List[str] = []
    for contrib in _findall_local(article, "contrib"):
        if (contrib.get("contrib-type") or "author") != "author":
            continue
        surname = _text_of(_find_local(contrib, "surname"))
        given = _text_of(_find_local(contrib, "given-names"))
        full = " ".join(p for p in (given, surname) if p)
        if full:
            names.append(full)
    return names


def _pub_year(article: ET.Element) -> str:
    for pub_date in _findall_local(article, "pub-date"):
        year = _find_local(pub_date, "year")
        if year is not None and clean_text(year.text):
            return clean_text(year.text)
    return ""


def _footnote_text(article: ET.Element, wanted: Iterable[str]) -> str:
    wanted_set = {w.lower() for w in wanted}
    out: List[str] = []
    for footnote in _findall_local(article, "fn"):
        kind = (footnote.get("fn-type") or "").lower()
        if kind in wanted_set:
            text = _text_of(footnote)
            if text:
                out.append(text)
    return " ".join(out)


def _child_local(parent: ET.Element, name: str) -> Optional[ET.Element]:
    """Direct child by local tag name. Does not reach into subsections."""
    for child in list(parent):
        if _tag(child) == name:
            return child
    return None


# Publishers disagree on the element that holds a funding or ethics statement.
# MDPI uses <notes>, Elsevier uses <fn-group>, others use a back-matter <sec>.
_STATEMENT_CONTAINERS = ("sec", "notes", "ack", "fn-group", "fn", "app")


def _named_section_text(article: ET.Element, pattern: str) -> str:
    """Text of any back-matter block whose heading matches `pattern`."""
    regex = re.compile(pattern, re.I)
    out: List[str] = []
    for element in article.iter():
        if _tag(element) not in _STATEMENT_CONTAINERS:
            continue
        title = _text_of(_child_local(element, "title"))
        labels = [
            title,
            element.get("sec-type", ""),
            element.get("notes-type", ""),
            element.get("fn-type", ""),
        ]
        if not any(regex.search(label or "") for label in labels):
            continue
        body = " ".join(
            _text_of(child) for child in element if _tag(child) == "p"
        ).strip()
        if not body:
            body = _text_of(element)
            if title and body.startswith(title):
                body = body[len(title):].strip()
        if body:
            out.append(body)
    return " ".join(dict.fromkeys(out))


def parse_pmc_article(xml_text: str) -> Optional[Dict[str, Any]]:
    """Parse a PMC / JATS article into a document with labelled blocks."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("PMC XML parse failed: %s", exc)
        return None

    article = root if _tag(root) == "article" else _find_local(root, "article")
    if article is None:
        return None
    if _find_local(article, "body") is None:
        return None  # Record exists in PMC but carries no full text.

    blocks: List[Dict[str, Any]] = []

    abstract_text_parts: List[str] = []
    for abstract in _findall_local(article, "abstract"):
        if (abstract.get("abstract-type") or "") in {"graphical", "teaser"}:
            continue
        text = _text_of(abstract, skip={"title"})
        if text:
            abstract_text_parts.append(text)
    abstract_text = clean_text(" ".join(abstract_text_parts))
    if abstract_text:
        blocks.append({"section": S.ABSTRACT, "title": "Abstract",
                       "kind": "text", "text": abstract_text})

    body = _find_local(article, "body")
    if body is not None:
        direct_paragraphs = [
            _text_of(child) for child in list(body) if _tag(child) == "p"
        ]
        if any(direct_paragraphs):
            blocks.append({"section": S.OTHER, "title": "", "kind": "text",
                           "text": " ".join(p for p in direct_paragraphs if p)})
        for child in list(body):
            if _tag(child) == "sec":
                _collect_sections(child, S.OTHER, blocks)
            elif _tag(child) == "table-wrap":
                blocks.append({"section": S.TABLE, "title": "Table",
                               "kind": "table", "text": _flatten_table(child)})

    # Many publishers park every table and figure in <floats-group> at article
    # level instead of inside the section that refers to them. Without this the
    # result tables - where the counts live - would be dropped entirely.
    seen_tables = {id(t) for t in _findall_local(body, "table-wrap")} if body is not None else set()
    floats = _find_local(article, "floats-group")
    if floats is not None:
        for child in list(floats):
            tag = _tag(child)
            if tag == "table-wrap" and id(child) not in seen_tables:
                blocks.append({"section": S.TABLE, "title": "Table",
                               "kind": "table", "text": _flatten_table(child)})
            elif tag == "fig":
                caption = _text_of(_find_local(child, "caption"))
                if caption:
                    blocks.append({"section": S.FIGURE, "title": "Figure",
                                   "kind": "figure", "text": caption})

    back = _find_local(article, "back")
    if back is not None:
        for child in list(back):
            tag = _tag(child)
            if tag == "ref-list":
                continue
            if tag == "sec":
                _collect_sections(child, S.OTHER, blocks)
            elif tag in {"ack", "fn-group", "notes"}:
                text = _text_of(child)
                if text:
                    label = S.label_section(_text_of(_find_local(child, "title")))
                    blocks.append({
                        "section": label if label != S.OTHER else S.FUNDING,
                        "title": "Back matter", "kind": "text", "text": text,
                    })

    journal = _text_of(_find_local(article, "journal-title"))
    keywords = [_text_of(k) for k in _findall_local(article, "kwd")]
    funding_statement = _text_of(_find_local(article, "funding-statement"))
    award_sources = [
        _text_of(f) for f in _findall_local(article, "funding-source")
    ]
    if not funding_statement:
        funding_statement = _footnote_text(
            article, ["financial-disclosure", "supported-by", "funding"])
    if not funding_statement:
        # MDPI, Frontiers and many others write "Funding" as a back-matter
        # section rather than a <funding-statement> element.
        funding_statement = _named_section_text(
            article, r"^\s*funding|financial support|grant support")

    affiliations = [_text_of(a) for a in _findall_local(article, "aff")]

    document: Dict[str, Any] = {
        "ids": _article_ids(article),
        "source_type": "full_text",
        "source_api": "pmc",
        "title": _text_of(_find_local(article, "article-title")),
        "journal": journal,
        "year": _pub_year(article),
        "authors": _authors(article),
        "article_type": article.get("article-type", ""),
        "language": article.get("{http://www.w3.org/XML/1998/namespace}lang", ""),
        "volume": _text_of(_find_local(article, "volume")),
        "issue": _text_of(_find_local(article, "issue")),
        "pages": "-".join(
            p for p in (_text_of(_find_local(article, "fpage")),
                        _text_of(_find_local(article, "lpage"))) if p
        ),
        "keywords": [k for k in keywords if k],
        "abstract": abstract_text,
        "funding_statement": funding_statement,
        "award_sources": [a for a in award_sources if a],
        "conflict_statement": _footnote_text(
            article, ["conflict", "coi-statement", "COI-statement",
                      "competing-interests"])
        or _named_section_text(article, r"conflicts? of interest|"
                                        r"competing interests?|"
                                        r"declaration of interest"),
        "ethics_statement": _named_section_text(
            article, r"ethic|consent|institutional review|iacuc"),
        "data_availability": _named_section_text(
            article, r"data availability|data sharing|accession"),
        "affiliations": [a for a in affiliations if a],
    }
    document["blocks"] = blocks
    return document


# --------------------------------------------------------------------------- #
# PubMed abstract-only records
# --------------------------------------------------------------------------- #
def parse_pubmed_article(xml_text: str) -> Optional[Dict[str, Any]]:
    """Parse a PubMed efetch record. Abstract and metadata only."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("PubMed XML parse failed: %s", exc)
        return None

    article = _find_local(root, "PubmedArticle")
    if article is None:
        return None

    # Read the IDs only from MedlineCitation/PMID and PubmedData/ArticleIdList.
    # The reference list holds an ArticleIdList for each cited paper, and
    # those IDs are not this paper.
    ids: Dict[str, str] = {}
    for pmid_element in _children_local(
            _find_local(article, "MedlineCitation"), "PMID")[:1]:
        ids["pmid"] = clean_text(pmid_element.text)
    id_lists = _children_local(_find_local(article, "PubmedData"),
                               "ArticleIdList")
    for other in _children_local(id_lists[0] if id_lists else None,
                                 "ArticleId"):
        kind = (other.get("IdType") or "").lower()
        value = clean_text(other.text)
        if kind == "doi" and value:
            ids["doi"] = value
        elif kind == "pmc" and value:
            ids["pmcid"] = value

    # A structured abstract has several labelled parts; keep the labels, they
    # are section names in miniature.
    parts: List[str] = []
    blocks: List[Dict[str, Any]] = []
    for element in _findall_local(article, "AbstractText"):
        label = clean_text(element.get("Label") or "")
        text = _text_of(element)
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
        blocks.append({
            "section": S.label_section(label) if label else S.ABSTRACT,
            "title": label or "Abstract", "kind": "text", "text": text,
        })
    abstract_text = clean_text(" ".join(parts))

    authors: List[str] = []
    for author in _findall_local(article, "Author"):
        surname = _text_of(_find_local(author, "LastName"))
        given = _text_of(_find_local(author, "ForeName"))
        full = " ".join(p for p in (given, surname) if p)
        if full:
            authors.append(full)

    grants = []
    for grant in _findall_local(article, "Grant"):
        agency = _text_of(_find_local(grant, "Agency"))
        if agency:
            grants.append(agency)

    year = ""
    pub_date = _find_local(article, "PubDate")
    if pub_date is not None:
        year_element = _find_local(pub_date, "Year")
        if year_element is not None:
            year = clean_text(year_element.text)
        else:
            medline = _find_local(pub_date, "MedlineDate")
            if medline is not None:
                match = re.search(r"(1[6-9]\d{2}|20\d{2})",
                                  clean_text(medline.text))
                year = match.group(1) if match else ""

    document: Dict[str, Any] = {
        "ids": ids,
        "source_type": "abstract_only",
        "source_api": "pubmed",
        "title": _text_of(_find_local(article, "ArticleTitle")),
        "journal": _text_of(_find_local(article, "Title")),
        "year": year,
        "authors": authors,
        "article_type": "; ".join(
            _text_of(p) for p in _findall_local(article, "PublicationType")
        ),
        "language": _text_of(_find_local(article, "Language")),
        "volume": _text_of(_find_local(article, "Volume")),
        "issue": _text_of(_find_local(article, "Issue")),
        "pages": _text_of(_find_local(article, "MedlinePgn")),
        "keywords": [_text_of(k) for k in _findall_local(article, "Keyword")],
        "abstract": abstract_text,
        "funding_statement": "; ".join(dict.fromkeys(grants)),
        "award_sources": list(dict.fromkeys(grants)),
        "conflict_statement": "",
        "ethics_statement": "",
        "data_availability": "",
        "affiliations": [
            _text_of(a) for a in _findall_local(article, "Affiliation")
        ],
    }
    document["blocks"] = blocks
    return document


def document_from_text(title: str, abstract: str,
                       ids: Optional[Dict[str, str]] = None,
                       **extra: Any) -> Dict[str, Any]:
    """Build an abstract-only document from text we already hold.

    Used when a paper has no PMC record and no PubMed record, but the uploaded
    workbook carried its abstract. The paper is still processed and still
    produces a value for every column.
    """
    abstract = clean_text(abstract)
    blocks = []
    if abstract:
        blocks.append({"section": S.ABSTRACT, "title": "Abstract",
                       "kind": "text", "text": abstract})
    document: Dict[str, Any] = {
        "ids": dict(ids or {}),
        "source_type": "abstract_only",
        "source_api": "supplied",
        "title": clean_text(title),
        "journal": "", "year": "", "authors": [], "article_type": "",
        "language": "", "volume": "", "issue": "", "pages": "",
        "keywords": [], "abstract": abstract, "funding_statement": "",
        "award_sources": [], "conflict_statement": "", "ethics_statement": "",
        "data_availability": "", "affiliations": [],
    }
    document.update(extra)
    document["blocks"] = blocks
    return document


def section_counts(document: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for chunk in document.get("chunks", []):
        counts[chunk["section"]] = counts.get(chunk["section"], 0) + 1
    return counts
