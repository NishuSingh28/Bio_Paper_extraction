"""One paper, one column set -> one validated value per column.

The unit of work is a *column group*, not a column: the columns the user
submitted are routed into groups (grouping.py), each group retrieves its own
passages (index.py), and each group costs one LLM call. Eight columns answered
from the methods section cost one call, not eight.

Metadata columns cost no call at all. They are read out of the source record.

Nothing the model returns is trusted. Every value goes through validate.py
before it reaches the sheet, and the confidence that survives is the model's
number damped by what those checks found.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import grouping
import index as index_module
import validate
from columns import ColumnSpec, group_json_schema
from grouping import ColumnGroup
from index import PaperIndex, Retrieved
from llm import BaseLLM, LLMError

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 12
DEFAULT_TOKEN_BUDGET = 2600
RETRIEVAL_FLOOR = 0.012  # below this the fused rank is barely better than noise

SYSTEM_PROMPT = (
    "You extract data from research papers into a table.\n\n"
    "You read only the passages you are given. You report only what those "
    "passages state. You never use outside knowledge about the paper, its "
    "authors, or its subject, and you never carry a number from one column to "
    "another.\n\n"
    "For every column you return four fields:\n"
    '- "value": the answer and nothing else - a number, a name, a date, or a '
    "short phrase. Never write a sentence, never repeat the question, and "
    "never copy the evidence into it. When a column lists allowed values, the "
    "value must be exactly one of them, spelled the same way. When a column "
    "asks for a number, write the digits alone, with no unit, no percent sign "
    'and no range. Write "Not stated" when the passages do not give the '
    "answer. Never guess, and never leave it empty.\n"
    '- "evidence_span": one sentence or clause copied word for word from the '
    "passages, which shows the value is correct. Copy it exactly, including "
    "numbers and spelling. Keep it under 300 characters. Leave it empty only "
    'when the value is "Not stated".\n'
    '- "source_section": the section label of the passage you took the '
    "evidence from, copied from the passage header.\n"
    '- "confidence": a number from 0.0 to 1.0. Use a low number when the '
    "passages are indirect, ambiguous, or say something close to the question "
    "without answering it.\n\n"
    "A wrong value is worse than a missing one. When two passages disagree, "
    'report the one the paper presents as its result, and lower the '
    "confidence."
)


def _paper_header(document: Dict[str, Any]) -> str:
    bits = [f"Title: {document.get('title') or '(not given)'}"]
    year = document.get("year")
    journal = document.get("journal")
    if year:
        bits.append(f"Year: {year}")
    if journal:
        bits.append(f"Journal: {journal}")
    coverage = (
        "full text" if document.get("source_type") == "full_text"
        else "abstract only"
    )
    bits.append(f"Text available: {coverage}")
    return "\n".join(bits)


def build_user_prompt(document: Dict[str, Any], group: ColumnGroup,
                      retrieved: Sequence[Retrieved]) -> str:
    """Assemble the per-paper half of the prompt."""
    context = index_module.format_context(retrieved)
    column_lines = "\n".join(spec.prompt_line() for spec in group.columns)
    key_lines = "\n".join(
        f'  "{spec.key}"  ->  {spec.name}' for spec in group.columns
    )
    sections_seen = sorted({r.chunk["section"] for r in retrieved})

    return (
        f"{_paper_header(document)}\n\n"
        f"Passages from this paper:\n"
        f"-------------------------\n{context}\n"
        f"-------------------------\n\n"
        f"Fill these columns for this paper:\n{column_lines}\n\n"
        f"Return one object for each of these keys:\n{key_lines}\n\n"
        f"Rules:\n"
        f'- Use only the passages above. If they do not answer a column, the '
        f'value is "Not stated".\n'
        f"- Copy the evidence_span from the passages word for word. Do not "
        f"paraphrase it, shorten it inside the sentence, or write a sentence "
        f"that is not there.\n"
        f"- Give source_section as one of: {', '.join(sections_seen)}.\n"
        f"- Answer every column, even when several columns take the same "
        f"value, and even when the answer is \"Not stated\".\n"
        f"- Put the answer in value and the supporting sentence in "
        f"evidence_span. The two are never the same text.\n"
        f"- Where a column lists allowed values, copy one of them exactly. "
        f"Pick the closest listed value for what the paper reports; do not "
        f"write the paper's own wording instead."
    )


def _looks_truncated(text: str) -> bool:
    return bool(text) and not text.rstrip().endswith("}")


def _max_output_tokens(group: ColumnGroup) -> int:
    # Four fields per column, with an evidence span up to 300 characters.
    return min(4096, 300 + 190 * len(group.columns))


def run_group(
    llm: BaseLLM,
    document: Dict[str, Any],
    paper_index: PaperIndex,
    group: ColumnGroup,
    *,
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    confidence_threshold: float = validate.DEFAULT_CONFIDENCE_THRESHOLD,
    max_retries: int = 1,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Extract one group's columns. Returns (values by column key, call stats)."""
    stats: Dict[str, Any] = {
        "group": group.key, "attempts": 0, "elapsed_s": 0.0,
        "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
        "cost_usd": 0.0, "error": None, "chunks": [], "llm_call": True,
    }
    started = time.perf_counter()

    retrieved = paper_index.search(
        group.retrieval_query(),
        keyword_terms=group.keyword_terms(),
        boost_sections=group.sections,
        top_k=top_k,
        token_budget=token_budget,
    )
    stats["chunks"] = [r.chunk["id"] for r in retrieved]
    best_score = max((r.score for r in retrieved), default=0.0)
    stats["retrieval_score"] = round(best_score, 5)

    if not retrieved:
        stats["error"] = "no text to search"
        stats["elapsed_s"] = time.perf_counter() - started
        values = {
            spec.key: validate.missing_value(
                spec, "the paper has no usable text", group.group_type)
            for spec in group.columns
        }
        return values, stats

    haystack = " ".join(r.chunk["text"] for r in retrieved)
    schema = group_json_schema(group.columns)
    prompt = build_user_prompt(document, group, retrieved)
    budget = _max_output_tokens(group)

    payload: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None

    for attempt in range(1, max_retries + 2):
        stats["attempts"] = attempt
        try:
            result = llm.complete_json(SYSTEM_PROMPT, prompt, schema,
                                       max_tokens=budget)
        except LLMError as exc:
            last_error = f"model call failed: {exc}"
            log.warning("[%s] %s (attempt %d)", group.key, last_error, attempt)
            continue

        stats["input_tokens"] += result.input_tokens
        stats["output_tokens"] += result.output_tokens
        stats["cached_input_tokens"] += result.cached_input_tokens
        stats["cost_usd"] += result.cost_usd

        try:
            parsed = json.loads(result.text)
        except json.JSONDecodeError as exc:
            last_error = f"could not read the response as JSON: {exc}"
            if _looks_truncated(result.text):
                # A schema-constrained response can still run out of output
                # budget mid-string. Retrying unchanged would stop in the same
                # place, so the budget grows.
                budget = min(4096, budget * 2)
                last_error += f" (response was cut off; budget raised to {budget})"
            log.warning("[%s] %s", group.key, last_error)
            continue

        if not isinstance(parsed, dict):
            last_error = f"expected a JSON object, got {type(parsed).__name__}"
            continue
        payload = parsed
        break

    stats["elapsed_s"] = time.perf_counter() - started

    if payload is None:
        stats["error"] = last_error or "unknown failure"
        values = {
            spec.key: validate.missing_value(
                spec, stats["error"], group.group_type)
            for spec in group.columns
        }
        return values, stats

    chunks = [r.chunk for r in retrieved]
    values: Dict[str, Dict[str, Any]] = {}
    for spec in group.columns:
        proposal = payload.get(spec.key)
        if not isinstance(proposal, dict):
            values[spec.key] = validate.missing_value(
                spec, "the model returned nothing for this column",
                group.group_type)
            continue
        values[spec.key] = validate.validate_value(
            spec, proposal,
            haystack=haystack,
            chunks=chunks,
            source_type=document.get("source_type", "full_text"),
            retrieval_score=best_score,
            retrieval_floor=RETRIEVAL_FLOOR,
            confidence_threshold=confidence_threshold,
            group_type=group.group_type,
        )
    return values, stats


def run_metadata_group(document: Dict[str, Any], group: ColumnGroup,
                       ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Fill metadata columns from the record. No model call.

    A field the record does not carry falls through to `missing_value`, so the
    column still gets a value and still reaches the review queue.
    """
    started = time.perf_counter()
    values: Dict[str, Dict[str, Any]] = {}
    for spec in group.columns:
        field_name = grouping.metadata_field_for(spec) or ""
        text = grouping.resolve_metadata(field_name, document)
        values[spec.key] = validate.metadata_value(spec, text, field_name)
    return values, {
        "group": group.key, "attempts": 0,
        "elapsed_s": time.perf_counter() - started,
        "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
        "cost_usd": 0.0, "error": None, "chunks": [], "llm_call": False,
    }


def extract_paper(
    llm: BaseLLM,
    document: Dict[str, Any],
    paper_index: PaperIndex,
    groups: Sequence[ColumnGroup],
    *,
    top_k: int = DEFAULT_TOP_K,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    confidence_threshold: float = validate.DEFAULT_CONFIDENCE_THRESHOLD,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """Run every group for one paper and return one record for the run store."""
    started = time.perf_counter()
    values: Dict[str, Dict[str, Any]] = {}
    values_by_group: Dict[str, Dict[str, Any]] = {}
    call_stats: List[Dict[str, Any]] = []

    for group in groups:
        if group.is_metadata:
            group_values, stats = run_metadata_group(document, group)
        else:
            group_values, stats = run_group(
                llm, document, paper_index, group,
                top_k=top_k, token_budget=token_budget,
                confidence_threshold=confidence_threshold,
                max_retries=max_retries,
            )
        values.update(group_values)
        values_by_group[group.key] = group_values
        call_stats.append(stats)

    flagged = sum(1 for v in values.values() if v.get("needs_review"))
    return {
        "paper_key": document.get("paper_key", ""),
        "article_id": document.get("article_id", ""),
        "ids": document.get("ids", {}),
        "title": document.get("title", ""),
        "year": document.get("year", ""),
        "journal": document.get("journal", ""),
        "source_type": document.get("source_type", ""),
        "source_api": document.get("source_api", ""),
        "fetch_notes": document.get("fetch_notes", ""),
        "chunk_count": document.get("chunk_count", 0),
        "values": values,
        "values_by_group": values_by_group,
        "calls": call_stats,
        "flagged_values": flagged,
        "llm_calls": sum(1 for s in call_stats if s.get("llm_call")),
        "cost_usd": round(sum(s.get("cost_usd", 0.0) for s in call_stats), 6),
        "input_tokens": sum(s.get("input_tokens", 0) for s in call_stats),
        "output_tokens": sum(s.get("output_tokens", 0) for s in call_stats),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "errors": [s["error"] for s in call_stats if s.get("error")],
    }
