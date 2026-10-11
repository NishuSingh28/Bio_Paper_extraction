"""Text helpers shared by the parser, the fetcher and the validator.

Nothing here calls a model. These are cheap, reproducible functions.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_WS_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Collapse whitespace and strip the stray newlines that spreadsheet exports carry."""
    if value is None:
        return ""
    s = str(value).replace(" ", " ")
    return _WS_RE.sub(" ", s).strip()


def _fold(s: str) -> str:
    """Aggressively normalise text so quote matching survives typography."""
    s = unicodedata.normalize("NFKD", s or "")
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"[‐-―−]", "-", s)  # dash variants
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return _WS_RE.sub(" ", s).strip()


def verify_quote(quote: str, abstract: str) -> str:
    """Check the model's evidence quote really came from the abstract.

    Returns "Yes" for a verbatim match, "Partial" when nearly all of the quote's
    words are present (the model paraphrased or joined clauses), and "No" when
    the quote is not supported by the text at all. "No" is the strongest
    hallucination signal in the whole pipeline.
    """
    q, a = _fold(quote), _fold(abstract)
    if not q:
        return "No quote"
    if not a:
        return "No"
    if q in a:
        return "Yes"

    q_words = q.split()
    if not q_words:
        return "No"
    a_words = set(a.split())
    overlap = sum(1 for w in q_words if w in a_words) / len(q_words)
    return "Partial" if overlap >= 0.9 else "No"
