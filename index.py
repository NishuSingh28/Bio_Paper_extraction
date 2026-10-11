"""Hybrid retrieval over one paper: BM25 plus semantic, fused by rank.

Two searchers, because they fail in opposite ways.

* BM25 matches exact terms. Gene names, drug names, strain codes and dosages
  are the tokens that matter most in a paper, and an embedding blurs them.
* Semantic search matches meaning. It finds "participants were randomly
  assigned" for a column named "randomisation", which BM25 cannot.

Reciprocal rank fusion combines them without tuning a weight, because the two
scores are not on a comparable scale. The section a column was routed to adds a
rank bonus - a prior, never a filter, so a mis-routed column still reaches its
evidence.

Chunks, BM25 statistics and vectors are built once per paper and kept in the
paper cache (cache.py). A second run over the same papers, with any columns,
costs no compute here.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import chunking
import sections as S
from embedder import Embedder, get_embedder

log = logging.getLogger(__name__)

# Bump when tokenize() or the BM25 state changes, so stored indexes rebuild.
BM25_VERSION = 1

RRF_K = 60.0
SECTION_BOOST = 0.6
DEFAULT_TOP_K = 12
DEFAULT_TOKEN_BUDGET = 2600

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-/+%\.]*")
_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with",
    "is", "are", "was", "were", "be", "been", "by", "at", "as", "that",
    "this", "these", "those", "from", "it", "its", "we", "our", "their",
    "which", "than", "then", "there", "here", "what", "when", "how", "any",
    "all", "each", "per", "into", "during", "between", "about", "used",
    "using", "use", "study", "paper", "article", "column", "value", "values",
}


def tokenize(text: str) -> List[str]:
    """Lowercase tokens that keep hyphens, slashes and decimals.

    "SARS-CoV-2", "IL-6" and "0.5" have to survive as single tokens, because
    they are exactly the terms a keyword search exists to match.
    """
    tokens = _TOKEN_RE.findall((text or "").lower())
    return [t.strip("-./") for t in tokens
            if t.strip("-./") and t not in _STOPWORDS and len(t) > 1]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
class BM25:
    """Okapi BM25 over one paper's chunks. Small enough to build per query set."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5,
                 b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.lengths = np.array([len(d) for d in documents], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if self.doc_count else 0.0

        self.term_frequency: List[Dict[str, int]] = []
        document_frequency: Dict[str, int] = {}
        for document in documents:
            counts: Dict[str, int] = {}
            for token in document:
                counts[token] = counts.get(token, 0) + 1
            self.term_frequency.append(counts)
            for token in counts:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        self.idf: Dict[str, float] = {}
        for token, frequency in document_frequency.items():
            # The +1 keeps the idf positive for a term present in every chunk.
            self.idf[token] = math.log(
                1.0 + (self.doc_count - frequency + 0.5) / (frequency + 0.5)
            )

    def to_state(self) -> Dict[str, Any]:
        """Plain data for the cache. `from_state` gives back the same index."""
        return {
            "version": BM25_VERSION, "k1": self.k1, "b": self.b,
            "lengths": [int(x) for x in self.lengths],
            "term_frequency": self.term_frequency, "idf": self.idf,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "BM25":
        if int(state.get("version", 0)) != BM25_VERSION:
            raise ValueError("BM25 state from another version")
        index = cls.__new__(cls)
        index.k1 = float(state["k1"])
        index.b = float(state["b"])
        index.lengths = np.array(state["lengths"], dtype=np.float32)
        index.doc_count = len(index.lengths)
        index.avg_length = (float(index.lengths.mean())
                            if index.doc_count else 0.0)
        index.term_frequency = [dict(t) for t in state["term_frequency"]]
        index.idf = {k: float(v) for k, v in state["idf"].items()}
        if len(index.term_frequency) != index.doc_count:
            raise ValueError("BM25 state is inconsistent")
        return index

    @classmethod
    def for_chunks(cls, chunks: Sequence[Dict[str, Any]]) -> "BM25":
        return cls([tokenize(c["text"]) for c in chunks])

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        out = np.zeros((self.doc_count,), dtype=np.float32)
        if not self.doc_count or not query_tokens:
            return out
        for token in query_tokens:
            idf = self.idf.get(token)
            if idf is None:
                continue
            for index, counts in enumerate(self.term_frequency):
                frequency = counts.get(token)
                if not frequency:
                    continue
                norm = 1.0 - self.b + self.b * (
                    self.lengths[index] / (self.avg_length or 1.0))
                out[index] += idf * (frequency * (self.k1 + 1.0)) / (
                    frequency + self.k1 * norm)
        return out


def _ranks(scores: np.ndarray) -> np.ndarray:
    """0-based rank per item, best score first. Ties share the better rank."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks


# --------------------------------------------------------------------------- #
# One paper's index
# --------------------------------------------------------------------------- #
@dataclass
class Retrieved:
    chunk: Dict[str, Any]
    score: float
    bm25_rank: int
    vector_rank: int


class PaperIndex:
    """Holds one paper's chunks, its BM25 statistics and its vectors."""

    def __init__(self, document: Dict[str, Any],
                 vectors: Optional[np.ndarray] = None,
                 embedder: Optional[Embedder] = None,
                 bm25: Optional[BM25] = None):
        self.document = document
        self.chunks: List[Dict[str, Any]] = list(document.get("chunks", []))
        self.embedder = embedder if embedder is not None else get_embedder()
        self.vectors = (
            vectors if vectors is not None and vectors.size
            else np.zeros((len(self.chunks), 0), dtype=np.float32)
        )
        self._bm25 = (bm25 if bm25 is not None
                      and bm25.doc_count == len(self.chunks)
                      else BM25.for_chunks(self.chunks))

    @property
    def has_vectors(self) -> bool:
        return (self.vectors.size > 0
                and self.vectors.shape[0] == len(self.chunks))

    def search(
        self,
        query: str,
        *,
        keyword_terms: Sequence[str] = (),
        boost_sections: Sequence[str] = (),
        top_k: int = DEFAULT_TOP_K,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        always_include_abstract: bool = True,
    ) -> List[Retrieved]:
        """Return the chunks most likely to answer `query`, within a budget."""
        if not self.chunks:
            return []

        query_tokens = tokenize(query) + tokenize(" ".join(keyword_terms))
        bm25_scores = self._bm25.scores(query_tokens)
        bm25_rank = _ranks(bm25_scores)

        if self.has_vectors:
            query_vector = self.embedder.encode([query])
            if query_vector.size and query_vector.shape[1] == self.vectors.shape[1]:
                vector_scores = (self.vectors.astype(np.float32)
                                 @ query_vector[0].astype(np.float32))
            else:
                vector_scores = np.zeros((len(self.chunks),), dtype=np.float32)
        else:
            vector_scores = np.zeros((len(self.chunks),), dtype=np.float32)
        vector_rank = _ranks(vector_scores)

        section_order = {name: i for i, name in enumerate(boost_sections)}
        fused = np.zeros((len(self.chunks),), dtype=np.float32)
        for index, chunk in enumerate(self.chunks):
            score = 1.0 / (RRF_K + bm25_rank[index] + 1)
            if self.has_vectors:
                score += 1.0 / (RRF_K + vector_rank[index] + 1)
            position = section_order.get(chunk["section"])
            if position is not None:
                # The prior is worth roughly one rank position, so it breaks
                # ties and lifts near-misses without overruling a strong match.
                score += SECTION_BOOST / (RRF_K + position + 1)
            fused[index] = score

        order = list(np.argsort(-fused, kind="stable"))
        if always_include_abstract:
            abstract_positions = [
                i for i, c in enumerate(self.chunks)
                if c["section"] == S.ABSTRACT
            ]
            # The abstract states the headline answer for many columns and
            # costs little, so it leads the context regardless of rank.
            for position in reversed(abstract_positions[:2]):
                if position in order:
                    order.remove(position)
                order.insert(0, position)

        picked: List[Retrieved] = []
        used_tokens = 0
        for index in order:
            if len(picked) >= top_k:
                break
            chunk = self.chunks[index]
            tokens = int(chunk.get("tokens") or chunking.approx_tokens(
                chunk["text"]))
            if picked and used_tokens + tokens > token_budget:
                continue
            used_tokens += tokens
            picked.append(Retrieved(
                chunk=chunk, score=float(fused[index]),
                bm25_rank=int(bm25_rank[index]),
                vector_rank=int(vector_rank[index]),
            ))
        return picked


def format_context(items: Sequence[Retrieved]) -> str:
    """Render retrieved chunks for a prompt, with a citable label each."""
    parts: List[str] = []
    for item in items:
        chunk = item.chunk
        heading = chunk.get("title") or chunk["section"].title()
        kind = "" if chunk.get("kind") == "text" else f" ({chunk['kind']})"
        parts.append(
            f"[{chunk['id']} | section: {chunk['section']}{kind} | {heading}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)
