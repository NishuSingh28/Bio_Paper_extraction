"""Deterministic validation: the model proposes, Python decides.

Nothing the model returns is accepted on trust. Every value is coerced to the
type the user declared, checked against the allowed values, the range and the
pattern they gave, and its evidence is matched back against the paper's own
text. The confidence that reaches the sheet is the model's number multiplied by
what these checks found, so a confident claim with an unverifiable quote scores
low and lands in the review queue.

Where the user declared no type, a type is inferred but never enforced. A
mismatch raises a soft flag instead of rejecting the value, because discarding
a correct answer on a guess loses real data.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sections as S
from columns import NOT_STATED, ColumnSpec
from normalize import clean_text, verify_quote

# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #
FLAG_NOT_FOUND = "not_found"
FLAG_LOW_CONFIDENCE = "low_confidence"
FLAG_FAILED_VALIDATION = "failed_validation"
FLAG_MISSING_EVIDENCE = "missing_evidence"
FLAG_UNVERIFIED_EVIDENCE = "unverified_evidence"
FLAG_THIN_RETRIEVAL = "thin_retrieval"
FLAG_ABSTRACT_ONLY_GAP = "abstract_only_gap"
FLAG_TYPE_MISMATCH_SOFT = "type_mismatch_soft"
FLAG_SECTION_REPAIRED = "section_repaired"

REVIEW_FLAGS = {
    FLAG_NOT_FOUND, FLAG_LOW_CONFIDENCE, FLAG_FAILED_VALIDATION,
    FLAG_MISSING_EVIDENCE, FLAG_UNVERIFIED_EVIDENCE, FLAG_THIN_RETRIEVAL,
    FLAG_ABSTRACT_ONLY_GAP, FLAG_TYPE_MISMATCH_SOFT,
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.60

_NOT_STATED_VALUES = {
    "", "not stated", "not reported", "not mentioned", "not specified",
    "not available", "not applicable", "none", "n/a", "na", "null", "nil",
    "unknown", "unclear", "no information", "not found", "-", "--",
}

# How much a failed check costs. A value that cannot be checked is worth less
# than one that passed, and far less than the model's own optimism suggests.
_QUOTE_FACTOR = {"Yes": 1.0, "Partial": 0.75, "No": 0.25, "No quote": 0.20}
_VALIDATION_FACTOR = {"ok": 1.0, "soft": 0.85, "failed": 0.30, "not_stated": 0.0}


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_RANGE_RE = re.compile(
    r"^\s*[-+]?\d[\d,.]*\s*(?:-|–|—|to|and)\s*[-+]?\d[\d,.]*\s*$", re.I)

_TRUE_WORDS = {
    "yes", "true", "y", "1", "present", "positive", "detected", "reported",
    "confirmed", "applicable", "done", "used",
}
_FALSE_WORDS = {
    "no", "false", "n", "0", "absent", "negative", "not detected",
    "not reported", "not used", "none",
}

# "1,234 animals" and "about 16.7 %" are numbers; a clause is not.
_MAX_NUMBER_WORDS = 4

_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")


def is_not_stated(value: str) -> bool:
    return clean_text(value).lower().strip(" .") in _NOT_STATED_VALUES


def parse_number(text: str) -> Tuple[Optional[float], str]:
    """Return (number, error). Units, commas and a percent sign are stripped."""
    raw = clean_text(text)
    if not raw:
        return None, "empty"
    if _RANGE_RE.match(raw):
        return None, f"a range, not a single number: {raw!r}"
    matches = _NUMBER_RE.findall(raw.replace("%", " "))
    if not matches:
        return None, f"no number in {raw!r}"
    if len(matches) > 1:
        # "12 of 45" or "3.2 (1.1-5.0)" is more than one number. Which one the
        # user wanted is not ours to decide, so it goes to a human.
        return None, f"more than one number in {raw!r}"
    if len(raw.split()) > _MAX_NUMBER_WORDS:
        # One number inside a sentence. Pulling it out would look like a clean
        # answer while hiding that the model returned prose, and the number in
        # the sentence is not always the one the column asked for.
        return None, f"a sentence, not a number: {raw!r}"
    try:
        return float(matches[0].replace(",", "")), ""
    except ValueError:
        return None, f"could not read a number from {raw!r}"


def parse_boolean(text: str) -> Tuple[Optional[bool], str]:
    raw = clean_text(text).lower().strip(" .")
    if raw in _TRUE_WORDS:
        return True, ""
    if raw in _FALSE_WORDS:
        return False, ""
    first = raw.split()[0] if raw else ""
    if first in _TRUE_WORDS:
        return True, ""
    if first in _FALSE_WORDS:
        return False, ""
    return None, f"not a yes/no value: {clean_text(text)!r}"


def parse_date(text: str) -> Tuple[str, str]:
    """Return (normalized date, error). A bare year is a valid answer."""
    raw = clean_text(text)
    match = _ISO_DATE_RE.search(raw)
    if match:
        return match.group(0), ""
    years = _YEAR_RE.findall(raw)
    if years:
        if len(set(years)) > 1:
            # "2019-2021" is a real answer for a column like "sampling years",
            # so it is kept rather than rejected.
            return f"{years[0]}-{years[-1]}", ""
        return years[0], ""
    return "", f"no year or date in {raw!r}"


def match_allowed(value: str, allowed: Sequence[str]) -> Tuple[str, str]:
    """Snap a value onto the allowed list. Returns (matched, error)."""
    raw = clean_text(value)
    for option in allowed:
        if raw == option:
            return option, ""
    lowered = raw.lower()
    for option in allowed:
        if lowered == option.lower():
            return option, ""
    # Close spelling is a formatting difference, not a different answer.
    best, best_score = "", 0.0
    for option in allowed:
        score = SequenceMatcher(None, lowered, option.lower()).ratio()
        if score > best_score:
            best, best_score = option, score
    if best_score >= 0.90:
        return best, ""
    return raw, (f"{raw!r} is not one of the allowed values "
                 f"({', '.join(allowed)})")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if number > 1.0:
        # Models report 85 as often as 0.85.
        number = number / 100.0 if number <= 100.0 else 1.0
    return max(0.0, min(1.0, number))


def _repair_section(reported: str, chunks: Sequence[Dict[str, Any]],
                    evidence: str) -> Tuple[str, bool]:
    """Keep the model's section if it is real; otherwise find the true one."""
    reported_clean = clean_text(reported).lower()
    available = {c["section"] for c in chunks}
    if reported_clean in available:
        return reported_clean, False
    for section in available:
        if section in reported_clean or reported_clean in section:
            return section, True

    folded = clean_text(evidence).lower()
    if folded:
        best_section, best_score = "", 0.0
        for chunk in chunks:
            text = chunk["text"].lower()
            if folded[:80] and folded[:80] in text:
                return chunk["section"], True
            score = SequenceMatcher(None, folded[:200], text[:400]).ratio()
            if score > best_score:
                best_section, best_score = chunk["section"], score
        if best_score >= 0.35 and best_section:
            return best_section, True
    return reported_clean or S.OTHER, bool(reported_clean)


def validate_value(
    spec: ColumnSpec,
    proposal: Dict[str, Any],
    *,
    haystack: str,
    chunks: Sequence[Dict[str, Any]] = (),
    source_type: str = "full_text",
    retrieval_score: Optional[float] = None,
    retrieval_floor: float = 0.0,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    group_type: str = "",
) -> Dict[str, Any]:
    """Check one proposed value and return the record that reaches the sheet."""
    raw_value = clean_text(proposal.get("value", ""))
    evidence = clean_text(proposal.get("evidence_span", ""))
    reported_section = clean_text(proposal.get("source_section", ""))
    model_confidence = _clamp_confidence(proposal.get("confidence", 0.5))

    flags: List[str] = []
    errors: List[str] = []
    status = "ok"
    value_out = raw_value
    normalized: Any = raw_value

    # -- presence ---------------------------------------------------------- #
    if is_not_stated(raw_value):
        status = "not_stated"
        value_out = NOT_STATED
        normalized = NOT_STATED
        flags.append(FLAG_NOT_FOUND)
    else:
        # -- type ---------------------------------------------------------- #
        enforce = spec.type_declared
        if spec.type in ("number", "integer"):
            number, error = parse_number(raw_value)
            if error:
                if enforce:
                    status, normalized = "failed", None
                    errors.append(error)
                else:
                    flags.append(FLAG_TYPE_MISMATCH_SOFT)
            else:
                if spec.type == "integer" and number is not None:
                    if abs(number - round(number)) > 1e-9:
                        if enforce:
                            status = "failed"
                            errors.append(f"{number} is not a whole number")
                        else:
                            flags.append(FLAG_TYPE_MISMATCH_SOFT)
                    else:
                        number = int(round(number))
                normalized = number
                if enforce and status == "ok":
                    value_out = str(number)

        elif spec.type == "boolean":
            flag, error = parse_boolean(raw_value)
            if error:
                if enforce:
                    status, normalized = "failed", None
                    errors.append(error)
                else:
                    flags.append(FLAG_TYPE_MISMATCH_SOFT)
            else:
                normalized = flag
                if enforce:
                    value_out = "Yes" if flag else "No"

        elif spec.type == "date":
            parsed, error = parse_date(raw_value)
            if error:
                if enforce:
                    status, normalized = "failed", None
                    errors.append(error)
                else:
                    flags.append(FLAG_TYPE_MISMATCH_SOFT)
            else:
                normalized = parsed
                if enforce:
                    value_out = parsed

        elif spec.type == "categorical" and spec.allowed_values:
            matched, error = match_allowed(raw_value, spec.allowed_values)
            if error:
                status = "failed"
                errors.append(error)
                normalized = matched
            else:
                value_out = normalized = matched

        # -- range --------------------------------------------------------- #
        if status == "ok" and isinstance(normalized, (int, float)) and \
                not isinstance(normalized, bool):
            if spec.minimum is not None and normalized < spec.minimum:
                status = "failed"
                errors.append(f"{normalized} is below the minimum "
                              f"{spec.minimum}")
            if spec.maximum is not None and normalized > spec.maximum:
                status = "failed"
                errors.append(f"{normalized} is above the maximum "
                              f"{spec.maximum}")

        # -- pattern ------------------------------------------------------- #
        if status == "ok" and spec.pattern:
            try:
                if not re.search(spec.pattern, value_out):
                    status = "failed"
                    errors.append(f"{value_out!r} does not match the pattern "
                                  f"{spec.pattern!r}")
            except re.error as exc:
                errors.append(f"the pattern {spec.pattern!r} is not valid: {exc}")

    if status == "failed":
        flags.append(FLAG_FAILED_VALIDATION)

    # -- evidence ---------------------------------------------------------- #
    if status == "not_stated":
        # A value the paper does not give needs no quote, and demanding one
        # would push every honest "Not stated" into the review queue twice.
        quote_check = "Not required"
        quote_factor = 1.0
    elif not evidence:
        quote_check = "No quote"
        quote_factor = _QUOTE_FACTOR["No quote"]
        flags.append(FLAG_MISSING_EVIDENCE)
    else:
        quote_check = verify_quote(evidence, haystack)
        quote_factor = _QUOTE_FACTOR.get(quote_check, 0.3)
        if quote_check == "No":
            flags.append(FLAG_UNVERIFIED_EVIDENCE)

    section, repaired = _repair_section(reported_section, chunks, evidence)
    if repaired:
        flags.append(FLAG_SECTION_REPAIRED)

    # -- coverage ---------------------------------------------------------- #
    if (status == "not_stated" and source_type != "full_text"
            and group_type in (S.GROUP_DESIGN, S.GROUP_RESULTS,
                               S.GROUP_POPULATION)):
        # Only a real gap: the value was not found *and* the full text was
        # never available. A value the abstract did answer is not a gap, and
        # flagging it would bury the ones that are.
        flags.append(FLAG_ABSTRACT_ONLY_GAP)
    if (retrieval_score is not None and retrieval_floor > 0
            and retrieval_score < retrieval_floor):
        flags.append(FLAG_THIN_RETRIEVAL)

    # -- confidence -------------------------------------------------------- #
    validation_key = status
    if status == "ok" and FLAG_TYPE_MISMATCH_SOFT in flags:
        validation_key = "soft"
    confidence = round(
        model_confidence * quote_factor
        * _VALIDATION_FACTOR.get(validation_key, 0.5),
        3,
    )
    if status != "not_stated" and confidence < confidence_threshold:
        flags.append(FLAG_LOW_CONFIDENCE)

    return {
        "column_key": spec.key,
        "column": spec.name,
        "value": value_out,
        "value_normalized": normalized,
        "evidence_span": evidence,
        "source_section": section,
        "confidence": confidence,
        "model_confidence": round(model_confidence, 3),
        "quote_check": quote_check,
        "validation": status,
        "validation_error": "; ".join(errors),
        "flags": sorted(set(flags)),
        "needs_review": bool(set(flags) & REVIEW_FLAGS),
    }


def missing_value(spec: ColumnSpec, reason: str,
                  group_type: str = "") -> Dict[str, Any]:
    """The record used when a group produced nothing for a column at all."""
    return {
        "column_key": spec.key,
        "column": spec.name,
        "value": NOT_STATED,
        "value_normalized": NOT_STATED,
        "evidence_span": "",
        "source_section": "",
        "confidence": 0.0,
        "model_confidence": 0.0,
        "quote_check": "No quote",
        "validation": "failed",
        "validation_error": reason,
        "flags": sorted({FLAG_NOT_FOUND, FLAG_FAILED_VALIDATION}),
        "needs_review": True,
    }


def metadata_value(spec: ColumnSpec, value: str, field_name: str) -> Dict[str, Any]:
    """The record for a column answered straight from the source record."""
    text = clean_text(value)
    if not text:
        return {
            "column_key": spec.key, "column": spec.name, "value": NOT_STATED,
            "value_normalized": NOT_STATED, "evidence_span": "",
            "source_section": "metadata", "confidence": 0.0,
            "model_confidence": 0.0, "quote_check": "Not required",
            "validation": "not_stated", "validation_error": "",
            "flags": [FLAG_NOT_FOUND], "needs_review": True,
        }
    return {
        "column_key": spec.key, "column": spec.name, "value": text,
        "value_normalized": text,
        "evidence_span": text[:300],
        "source_section": "metadata",
        # The value came out of the record itself, so there is nothing for a
        # model to be wrong about.
        "confidence": 1.0, "model_confidence": 1.0,
        "quote_check": "Not required", "validation": "ok",
        "validation_error": "", "flags": [], "needs_review": False,
    }
