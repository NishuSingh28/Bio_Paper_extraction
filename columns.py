"""User-defined column specifications for one extraction run.

A run does not have a fixed schema. The user types the column names they want,
optionally with a type, a description and allowed values. Everything downstream
- grouping, prompting, validation, the Excel sheet - is built from these specs
at run time, so changing what gets extracted never means changing code.

The specs travel with the run (see runstore.py), so two runs with different
column sets cannot mix.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #
# "text" is the safe default: a value is stored as the model wrote it and only
# checked for presence and evidence. The other types add a deterministic check.
COLUMN_TYPES = ["text", "number", "integer", "boolean", "categorical", "date"]

NOT_STATED = "Not stated"

# Section a column is most likely answered from. "auto" lets grouping.py decide.
SECTION_HINTS = [
    "auto",
    "metadata",
    "design_methods",
    "population_sample",
    "results_outcomes",
    "discussion_limitations",
    "general",
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Turn a column name into a stable JSON-safe key."""
    s = _SLUG_RE.sub("_", str(name).strip().lower()).strip("_")
    return s or "column"


# --------------------------------------------------------------------------- #
# Type inference for columns the user did not type
# --------------------------------------------------------------------------- #
# Used only to raise a soft review flag. An inferred type is never enforced,
# because rejecting a correct value on a guessed type loses real data.
_NUMBER_CUES = [
    "sample size", "n =", "number of", "count", "total", "how many",
    "dose", "dosage", "age", "duration", "length", "weight", "height",
    "percent", "percentage", "rate", "ratio", "score", "mean", "median",
    "prevalence", "incidence", "follow up", "follow-up", "year",
    "temperature", "concentration", "titre", "titer", "volume",
]
_INTEGER_CUES = [
    "sample size", "number of", "count", "n participants", "n patients",
    "n animals", "cohort size", "how many",
]
_BOOLEAN_CUES = [
    "is ", "was ", "has ", "have ", "whether", "did ", "does ",
    "randomised", "randomized", "blinded", "controlled", "placebo",
    "peer reviewed", "open access", "preregistered", "registered",
]
_DATE_CUES = ["date", "year of", "publication year", "sampling year", "period"]


def infer_type(name: str, description: str = "") -> str:
    """Guess a column's type from its name. Advisory only - never enforced."""
    text = f"{name} {description}".lower().strip()
    if not text:
        return "text"
    for cue in _DATE_CUES:
        if cue in text:
            return "date"
    for cue in _INTEGER_CUES:
        if cue in text:
            return "integer"
    for cue in _BOOLEAN_CUES:
        if text.startswith(cue) or f" {cue}" in text:
            return "boolean"
    for cue in _NUMBER_CUES:
        if cue in text:
            return "number"
    return "text"


def _as_list(value: Any) -> List[str]:
    """Accept a list, or a string with commas, semicolons or newlines."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    parts = re.split(r"[;\n|]+|,(?![^(]*\))", str(value))
    return [p.strip() for p in parts if p.strip()]


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #
@dataclass
class ColumnSpec:
    """One column the user asked for."""

    name: str
    type: str = ""
    description: str = ""
    allowed_values: List[str] = field(default_factory=list)
    unit: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    pattern: str = ""
    section_hint: str = "auto"
    # True when the user chose the type. False means infer_type() supplied it
    # and validation must stay soft. None means "work it out from `type`",
    # which is what a fresh row from the UI carries.
    type_declared: Optional[bool] = None
    key: str = ""

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.description = str(self.description or "").strip()
        self.unit = str(self.unit or "").strip()
        self.pattern = str(self.pattern or "").strip()
        self.allowed_values = _as_list(self.allowed_values)
        self.minimum = _as_float(self.minimum)
        self.maximum = _as_float(self.maximum)

        declared = str(self.type or "").strip().lower()
        if self.type_declared is False:
            # Reloaded from a template or a run: the stored type was inferred,
            # so it must stay soft even though it names a real type.
            self.type = declared if declared in COLUMN_TYPES else infer_type(
                self.name, self.description)
            self.type_declared = False
        elif declared in COLUMN_TYPES:
            self.type = declared
            self.type_declared = True
        else:
            self.type = infer_type(self.name, self.description)
            self.type_declared = False

        # Allowed values imply a categorical column even if the user did not
        # pick the type, because the check is then explicit rather than guessed.
        if self.allowed_values and self.type != "categorical":
            self.type = "categorical"
            self.type_declared = True

        # A minimum or a maximum is an explicit numeric constraint, so it makes
        # the type declared too. Otherwise the range would be checked against a
        # type that was only guessed, which is the one thing inference must
        # never do.
        if (self.minimum is not None or self.maximum is not None) \
                and not self.type_declared:
            self.type = "integer" if self.type == "integer" else "number"
            self.type_declared = True

        if self.section_hint not in SECTION_HINTS:
            self.section_hint = "auto"
        if not self.key:
            self.key = slugify(self.name)

    # ----------------------------------------------------------------- #
    @property
    def inferred_type(self) -> str:
        """The type a guess would give. Equals .type when nothing was declared."""
        return self.type if not self.type_declared else infer_type(
            self.name, self.description)

    def retrieval_text(self) -> str:
        """The text used to embed and to build a retrieval query."""
        parts = [self.name]
        if self.description:
            parts.append(self.description)
        if self.allowed_values:
            parts.append(" ".join(self.allowed_values[:12]))
        if self.unit:
            parts.append(self.unit)
        return " ".join(parts)

    def prompt_line(self) -> str:
        """How this column is described to the model."""
        bits = [f'- "{self.name}"']
        if self.type_declared:
            bits.append(f"(type: {self.type})")
        if self.unit:
            bits.append(f"(unit: {self.unit})")
        if self.description:
            bits.append(f"- {self.description}")
        if self.allowed_values:
            allowed = ", ".join(f'"{v}"' for v in self.allowed_values)
            bits.append(f'- must be one of: {allowed}, or "{NOT_STATED}"')
        if self.minimum is not None or self.maximum is not None:
            lo = "any" if self.minimum is None else self.minimum
            hi = "any" if self.maximum is None else self.maximum
            bits.append(f"- expected range {lo} to {hi}")
        return " ".join(str(b) for b in bits)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnSpec":
        allowed = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        return cls(**{k: v for k, v in data.items() if k in allowed})


# --------------------------------------------------------------------------- #
# Column sets
# --------------------------------------------------------------------------- #
class DuplicateColumnError(ValueError):
    """Two columns resolved to the same key."""


def build_column_set(rows: Iterable[Dict[str, Any]]) -> List[ColumnSpec]:
    """Turn UI / JSON rows into specs, with unique keys and no blank names."""
    specs: List[ColumnSpec] = []
    seen: Dict[str, int] = {}
    names_seen: set[str] = set()

    for row in rows:
        name = str(row.get("name", "") or "").strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered in names_seen:
            raise DuplicateColumnError(f"Column {name!r} appears more than once.")
        names_seen.add(lowered)

        spec = ColumnSpec.from_dict({**row, "name": name, "key": ""})
        base = spec.key
        if base in seen:
            seen[base] += 1
            spec.key = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
        specs.append(spec)
    return specs


def column_set_fingerprint(specs: List[ColumnSpec]) -> str:
    """Stable hash of a column set. Two runs differ if this differs."""
    import hashlib

    payload = json.dumps(
        [
            {
                "key": s.key, "name": s.name, "type": s.type,
                # Whether the type was declared changes how strictly the value
                # is checked, so two runs that differ only in that are not the
                # same run.
                "type_declared": bool(s.type_declared),
                "description": s.description, "allowed_values": s.allowed_values,
                "unit": s.unit, "minimum": s.minimum, "maximum": s.maximum,
                "pattern": s.pattern, "section_hint": s.section_hint,
            }
            for s in specs
        ],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def specs_to_rows(specs: List[ColumnSpec]) -> List[Dict[str, Any]]:
    """Flatten specs for a Streamlit data editor."""
    return [
        {
            "name": s.name,
            "type": s.type if s.type_declared else "",
            "description": s.description,
            "allowed_values": ", ".join(s.allowed_values),
            "unit": s.unit,
            "minimum": s.minimum,
            "maximum": s.maximum,
            "pattern": s.pattern,
            "section_hint": s.section_hint,
        }
        for s in specs
    ]


# --------------------------------------------------------------------------- #
# Response schema handed to the model
# --------------------------------------------------------------------------- #
def group_json_schema(specs: List[ColumnSpec]) -> Dict[str, Any]:
    """JSON Schema for one LLM call covering `specs`.

    Every value is a string, including numbers and categories. Python does the
    coercion afterwards. Constraining the decoder to an enum would guarantee a
    *valid* value, not a *correct* one, and would remove the strongest review
    trigger the pipeline has: a value that fails a deterministic check.
    """
    props: Dict[str, Any] = {}
    for spec in specs:
        props[spec.key] = {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
                "evidence_span": {"type": "string"},
                "source_section": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["value", "evidence_span", "source_section",
                         "confidence"],
        }
    return {
        "type": "object",
        "properties": props,
        "required": [s.key for s in specs],
        # Required by OpenAI strict mode and accepted everywhere else.
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def list_templates(directory: Optional[Path] = None) -> List[str]:
    d = directory or TEMPLATE_DIR
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def save_template(name: str, specs: List[ColumnSpec],
                  directory: Optional[Path] = None) -> Path:
    d = directory or TEMPLATE_DIR
    d.mkdir(parents=True, exist_ok=True)
    safe = slugify(name)
    path = d / f"{safe}.json"
    payload = {"name": name, "columns": [s.to_dict() for s in specs]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def load_template(name: str,
                  directory: Optional[Path] = None) -> List[ColumnSpec]:
    d = directory or TEMPLATE_DIR
    path = d / f"{name}.json"
    if not path.exists():
        path = d / f"{slugify(name)}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ColumnSpec.from_dict(c) for c in data.get("columns", [])]


def delete_template(name: str, directory: Optional[Path] = None) -> bool:
    d = directory or TEMPLATE_DIR
    for candidate in (d / f"{name}.json", d / f"{slugify(name)}.json"):
        if candidate.exists():
            candidate.unlink()
            return True
    return False
