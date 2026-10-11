"""Canonical paper sections, and the vocabulary that maps onto them.

Two things use this module:

* chunking.py labels each chunk of a paper with a canonical section.
* grouping.py routes a user-defined column to the section that most likely
  answers it.

Nothing here names a column. The lexicon describes *sections*, so a column the
user invents tomorrow is routed by its similarity to a section, not by a lookup.
"""

from __future__ import annotations

import re
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Canonical sections of a paper
# --------------------------------------------------------------------------- #
ABSTRACT = "abstract"
INTRODUCTION = "introduction"
METHODS = "methods"
RESULTS = "results"
DISCUSSION = "discussion"
CONCLUSION = "conclusion"
FUNDING = "funding"
ETHICS = "ethics"
TABLE = "table"
FIGURE = "figure"
OTHER = "other"

SECTIONS = [
    ABSTRACT, INTRODUCTION, METHODS, RESULTS, DISCUSSION, CONCLUSION,
    FUNDING, ETHICS, TABLE, FIGURE, OTHER,
]

# Section titles as they appear in PMC XML, in priority order. The first
# pattern that matches a heading wins.
_TITLE_PATTERNS: List[tuple[str, str]] = [
    (FUNDING, r"funding|financial (support|disclosure)|grant|sponsor|"
              r"acknowledg"),
    (ETHICS, r"ethic|consent|institutional review|iacuc|animal welfare|"
             r"declaration|competing interest|conflict of interest|"
             r"data availability"),
    (METHODS, r"method|material|procedure|protocol|experimental|study design|"
              r"patients and|subjects and|design|participants|sampling|"
              r"statistical analys|data collection|measurement|assay"),
    (RESULTS, r"result|finding|outcome|observation"),
    (DISCUSSION, r"discussion|limitation|interpretation|implication"),
    (CONCLUSION, r"conclusion|summary|take[- ]home"),
    (INTRODUCTION, r"introduction|background|rationale|objective|aim"),
    (ABSTRACT, r"abstract|synopsis"),
]

_COMPILED = [(label, re.compile(pat, re.I)) for label, pat in _TITLE_PATTERNS]


def label_section(title: str, sec_type: str = "") -> str:
    """Map a raw PMC heading (and its sec-type attribute) to a canonical name."""
    for source in (sec_type, title):
        text = str(source or "").strip()
        if not text:
            continue
        for label, pattern in _COMPILED:
            if pattern.search(text):
                return label
    return OTHER


# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #
# A group is one LLM call. Each group names the sections its answers live in;
# retrieval gives those sections a rank bonus, and never excludes the rest.
GROUP_METADATA = "metadata"
GROUP_DESIGN = "design_methods"
GROUP_POPULATION = "population_sample"
GROUP_RESULTS = "results_outcomes"
GROUP_DISCUSSION = "discussion_limitations"
GROUP_GENERAL = "general"

LLM_GROUPS = [
    GROUP_DESIGN, GROUP_POPULATION, GROUP_RESULTS,
    GROUP_DISCUSSION, GROUP_GENERAL,
]

GROUP_LABELS = {
    GROUP_METADATA: "Metadata (no model call)",
    GROUP_DESIGN: "Design and methods",
    GROUP_POPULATION: "Population and sample",
    GROUP_RESULTS: "Results and outcomes",
    GROUP_DISCUSSION: "Discussion and limitations",
    GROUP_GENERAL: "General",
}

# Which canonical sections each group prefers, best first.
GROUP_SECTIONS: Dict[str, List[str]] = {
    GROUP_DESIGN: [METHODS, ABSTRACT, RESULTS, OTHER],
    GROUP_POPULATION: [METHODS, RESULTS, TABLE, ABSTRACT],
    GROUP_RESULTS: [RESULTS, TABLE, ABSTRACT, FIGURE],
    GROUP_DISCUSSION: [DISCUSSION, CONCLUSION, ABSTRACT, INTRODUCTION],
    GROUP_GENERAL: [ABSTRACT, INTRODUCTION, CONCLUSION, RESULTS, METHODS],
    GROUP_METADATA: [FUNDING, ETHICS, ABSTRACT, OTHER],
}

# Cue terms per group. Matched against a column's name plus its description.
# These describe what a *section* talks about, not what a column is called.
GROUP_CUES: Dict[str, List[str]] = {
    GROUP_DESIGN: [
        "design", "method", "protocol", "procedure", "intervention",
        "randomis", "randomiz", "blind", "placebo", "control group",
        "comparator", "arm", "allocation", "assay", "technique",
        "instrument", "questionnaire", "measurement", "dose", "dosage",
        "regimen", "administration", "route", "duration of treatment",
        "follow up", "follow-up", "setting", "study type", "trial",
        "cohort", "cross sectional", "case control", "prospective",
        "retrospective", "in vitro", "in vivo", "experimental",
        "statistical analysis", "model used", "software", "sequencing",
        "pcr", "elisa", "detection method", "sampling method",
        "inclusion criteria", "exclusion criteria", "eligibility",
    ],
    GROUP_POPULATION: [
        "sample size", "participant", "patient", "subject", "animal",
        "species", "strain", "population", "cohort size", "n =",
        "number of", "age", "sex", "gender", "male", "female", "ethnic",
        "demographic", "baseline characteristic", "recruit", "enrol",
        "enroll", "site", "country", "region", "location", "setting",
        "year of sampling", "sampling period", "host", "tissue",
        "specimen", "weight", "body mass", "comorbid",
    ],
    GROUP_RESULTS: [
        "result", "outcome", "finding", "effect", "efficacy",
        "prevalence", "incidence", "mortality", "survival", "response rate",
        "odds ratio", "hazard ratio", "risk ratio", "relative risk",
        "confidence interval", "p value", "p-value", "significan",
        "mean difference", "median", "percentage", "proportion",
        "positive", "negative", "detected", "titre", "titer",
        "sensitivity", "specificity", "accuracy", "auc", "correlation",
        "adverse event", "side effect", "toxicity", "complication",
        "primary endpoint", "secondary endpoint", "seroprevalence",
    ],
    GROUP_DISCUSSION: [
        "limitation", "bias", "confound", "generalis", "generaliz",
        "implication", "interpretation", "future work", "recommendation",
        "strength", "weakness", "caveat", "conclusion", "clinical relevance",
        "policy", "comparison with", "consistent with", "novelty",
    ],
    GROUP_GENERAL: [
        "topic", "subject", "theme", "summary", "aim", "objective",
        "purpose", "research question", "hypothesis", "background",
        "rationale", "scope", "keyword", "category", "classification",
    ],
}

# One short paragraph per group. Embedded once, then compared against a
# column's embedded text. This is the half of routing that generalises to
# vocabulary the cue lists never anticipated.
GROUP_PROTOTYPES: Dict[str, str] = {
    GROUP_DESIGN: (
        "How the study was carried out. The study design and type, the "
        "protocol and procedures, the intervention or exposure and how it was "
        "administered, the comparator or control, randomisation and blinding, "
        "laboratory assays and detection techniques, instruments and "
        "questionnaires, doses and regimens, duration and follow-up, "
        "inclusion and exclusion criteria, and the statistical analysis plan."
    ),
    GROUP_POPULATION: (
        "Who or what was studied. The number of participants, patients, "
        "animals or specimens, the sample size and how it was reached, the "
        "species and strain, age, sex and demographic characteristics, "
        "recruitment and enrolment, the study sites, the country and region, "
        "the calendar period of data collection, and baseline characteristics."
    ),
    GROUP_RESULTS: (
        "What the study found. Primary and secondary outcomes, effect sizes, "
        "prevalence and incidence, counts of positive and negative cases, "
        "rates and proportions, odds ratios and hazard ratios, confidence "
        "intervals and p-values, sensitivity and specificity, survival and "
        "mortality, adverse events, and the numbers reported in result tables."
    ),
    GROUP_DISCUSSION: (
        "What the authors make of the findings. The interpretation, how the "
        "results compare with earlier work, the limitations and sources of "
        "bias, confounding, how far the findings generalise, the clinical or "
        "policy implications, recommendations, and the authors' conclusions."
    ),
    GROUP_GENERAL: (
        "What the paper is about as a whole. The topic and subject area, the "
        "research question, the stated aim, objective or hypothesis, the "
        "background and rationale, the scope of the work, and a summary of "
        "the article overall."
    ),
}
