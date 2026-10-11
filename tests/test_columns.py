"""Self-checks for the column-extraction pipeline.

No model and no network. Everything here is the deterministic half: column
parsing, generic grouping, chunking, hybrid retrieval, validation, the run
store, resume, and the exported sheets. A fake model stands in for the real
one, so the resume test proves the checkpoint logic rather than the model.

    pytest tests/test_columns.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import chunking
import export
import fetch
import grouping
import index as index_module
import sections as S
import validate
from columns import (
    ColumnSpec,
    build_column_set,
    column_set_fingerprint,
    delete_template,
    group_json_schema,
    infer_type,
    list_templates,
    load_template,
    save_template,
    slugify,
)
from llm import LLMResult
from runstore import RunRegistry, RunStore

def check(got: Any, want: Any, label: str) -> None:
    assert got == want, f"{label}: got {got!r}, want {want!r}"


def check_true(condition: Any, label: str) -> None:
    check(bool(condition), True, label)


# --------------------------------------------------------------------------- #
def test_column_specs() -> None:
    print("column specs")
    check(slugify("Sample size (n)"), "sample_size_n", "slugify")
    check(infer_type("sample size"), "integer", "infer integer")
    check(infer_type("was the trial blinded"), "boolean", "infer boolean")
    check(infer_type("prevalence rate"), "number", "infer number")
    check(infer_type("host species"), "text", "infer text")

    specs = build_column_set([
        {"name": "sample size", "type": "integer"},
        {"name": "sample size (n)"},
        {"name": "outcome", "allowed_values": "good, bad"},
        {"name": ""},
    ])
    check(len(specs), 3, "blank names dropped")
    check([s.key for s in specs],
          ["sample_size", "sample_size_n", "outcome"], "unique keys")
    check(specs[0].type_declared, True, "declared type kept")
    check(specs[1].type_declared, False, "undeclared type inferred, not locked")
    check(specs[1].type, "integer", "inferred type recorded")
    check(specs[2].type, "categorical", "allowed values imply categorical")
    check(specs[2].type_declared, True, "allowed values make the check explicit")

    ranged = build_column_set([{"name": "some measure", "minimum": 0,
                                "maximum": 100}])[0]
    check(ranged.type_declared, True,
          "a range is an explicit constraint, so the type is declared")
    check(ranged.type, "number", "a range implies a numeric column")

    duplicated = False
    try:
        build_column_set([{"name": "Dose"}, {"name": "dose"}])
    except Exception:
        duplicated = True
    check(duplicated, True, "duplicate column name refused")

    schema = group_json_schema(specs)
    check(sorted(schema["required"]), sorted(s.key for s in specs),
          "schema requires every column")
    check(schema["properties"]["outcome"]["properties"]["value"]["type"],
          "string", "values are strings; Python does the coercion")
    check(schema["additionalProperties"], False, "schema is strict")

    a = build_column_set([{"name": "dose"}])
    b = build_column_set([{"name": "dose"}])
    c = build_column_set([{"name": "dose", "type": "number"}])
    check(column_set_fingerprint(a), column_set_fingerprint(b),
          "same columns, same fingerprint")
    check(column_set_fingerprint(a) != column_set_fingerprint(c), True,
          "different columns, different fingerprint")


def test_templates() -> None:
    print("templates")
    directory = Path(tempfile.mkdtemp())
    try:
        specs = build_column_set([
            {"name": "dose", "type": "number", "unit": "mg"},
            {"name": "outcome", "allowed_values": "good, bad"},
        ])
        save_template("My Set", specs, directory)
        check(list_templates(directory), ["my_set"], "template listed")
        loaded = load_template("my_set", directory)
        check([s.name for s in loaded], ["dose", "outcome"], "template loaded")
        check(loaded[0].unit, "mg", "template keeps the unit")
        check(loaded[1].allowed_values, ["good", "bad"],
              "template keeps allowed values")
        check(delete_template("my_set", directory), True, "template deleted")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_grouping() -> None:
    print("column grouping")
    specs = build_column_set([
        {"name": "title"}, {"name": "doi"}, {"name": "funding source"},
        {"name": "sample size", "type": "integer"},
        {"name": "randomisation method"},
        {"name": "primary outcome"}, {"name": "odds ratio", "type": "number"},
        {"name": "main limitation"},
    ])
    groups, routings = grouping.plan_groups(specs)
    by_column = {r.name: r.group for r in routings}

    check(by_column["title"], S.GROUP_METADATA, "title needs no model call")
    check(by_column["doi"], S.GROUP_METADATA, "doi needs no model call")
    check(by_column["funding source"], S.GROUP_METADATA,
          "funding comes from the record")
    check(by_column["sample size"], S.GROUP_POPULATION, "sample size routed")
    check(by_column["randomisation method"], S.GROUP_DESIGN,
          "randomisation routed to methods")
    check(by_column["odds ratio"], S.GROUP_RESULTS, "odds ratio routed")
    check(by_column["main limitation"], S.GROUP_DISCUSSION, "limitation routed")

    metadata_groups = [g for g in groups if g.is_metadata]
    check(len(metadata_groups), 1, "one metadata group")
    check(len(metadata_groups[0].columns), 3, "three metadata columns")

    # A column nobody anticipated must still be routed, not dropped.
    odd = build_column_set([{"name": "CRISPR guide RNA sequence"},
                            {"name": "xyzzy"}])
    odd_groups, odd_routings = grouping.plan_groups(odd)
    check(len(odd_routings), 2, "unknown columns still routed")
    check_true(all(r.group in S.LLM_GROUPS for r in odd_routings),
               "unknown columns land in a real group")
    check(sum(len(g.columns) for g in odd_groups), 2,
          "every column reaches a group")

    # Oversized groups split; no group exceeds the cap.
    many = build_column_set([
        {"name": f"outcome measure {i}", "description": "a reported result"}
        for i in range(14)
    ])
    split_groups, _ = grouping.plan_groups(many, max_per_call=5)
    check_true(all(len(g.columns) <= 5 for g in split_groups),
               "no group is over the cap")
    check(sum(len(g.columns) for g in split_groups), 14,
          "splitting loses no column")

    # Every column ends up in exactly one group.
    all_keys = [c.key for g in groups for c in g.columns]
    check(sorted(all_keys), sorted(s.key for s in specs),
          "each column appears exactly once")


def test_chunking() -> None:
    print("chunking")
    check(S.label_section("Materials and Methods"), S.METHODS, "methods title")
    check(S.label_section("", "results"), S.RESULTS, "sec-type used")
    check(S.label_section("Funding"), S.FUNDING, "funding title")
    check(S.label_section("Qwerty"), S.OTHER, "unknown title")

    long_text = " ".join(f"Sentence number {i} says something useful here."
                         for i in range(120))
    pieces = chunking.split_passage(long_text, target_tokens=100,
                                    overlap_tokens=20)
    check_true(len(pieces) > 3, "a long passage is split")
    check_true(all(chunking.approx_tokens(p) <= 160 for p in pieces),
               "each piece respects the budget")
    check_true(pieces[0].split()[-5] in pieces[1],
               "consecutive pieces overlap")

    xml = """<article article-type="research-article" xml:lang="en">
      <front><journal-meta><journal-title>Test Journal</journal-title></journal-meta>
      <article-meta>
        <article-id pub-id-type="pmid">123456</article-id>
        <article-id pub-id-type="doi">10.1000/test</article-id>
        <title-group><article-title>A Study Of Things</article-title></title-group>
        <contrib-group><contrib contrib-type="author"><name>
          <surname>Smith</surname><given-names>Jane</given-names></name></contrib></contrib-group>
        <pub-date><year>2021</year></pub-date>
        <abstract><p>We tested 42 voles and found 7 positive.</p></abstract>
      </article-meta></front>
      <body>
        <sec sec-type="methods"><title>Methods</title>
          <p>Animals were trapped in Poland during 2019 and screened by PCR<xref ref-type="bibr">[3]</xref>.</p></sec>
        <sec sec-type="results"><title>Results</title>
          <p>Seven of 42 animals were positive, a prevalence of 16.7 percent.</p></sec>
        <sec><title>Discussion</title><p>The sample size was small.</p></sec>
      </body>
      <back>
        <notes><title>Funding</title><p>Funded by the Test Council.</p></notes>
        <notes notes-type="COI-statement"><title>Conflicts of Interest</title>
          <p>The authors declare none.</p></notes>
        <ref-list><ref><p>Someone et al. 2018.</p></ref></ref-list>
      </back>
      <floats-group>
        <table-wrap><label>Table 1</label><caption><p>Counts by site.</p></caption>
          <table><tr><th>Site</th><th>Tested</th></tr><tr><td>North</td><td>42</td></tr></table>
        </table-wrap>
      </floats-group>
    </article>"""
    document = chunking.parse_pmc_article(xml)
    check_true(document is not None, "PMC article parsed")
    check_true("chunks" not in document, "parsing does not chunk")
    document["chunks"] = chunking.chunk_document(document, "p1")
    for chunk in document["chunks"]:
        check_true(chunk["chunk_id"].startswith("p1:"),
                   "chunk ids carry the paper key")
        block = chunking.clean_text(document["blocks"][chunk["block"]]["text"])
        span = block[chunk["char_start"]:chunk["char_end"]]
        check_true(span and chunk["text"].startswith(span),
                   "offsets point at the chunk text in its first block")
    check(len({c["chunk_id"] for c in document["chunks"]}),
          len(document["chunks"]), "chunk ids are unique")
    check(document["ids"]["pmid"], "123456", "pmid read")
    check(document["title"], "A Study Of Things", "title read")
    check(document["year"], "2021", "year read")
    check(document["funding_statement"], "Funded by the Test Council.",
          "funding read from back matter")
    check(document["conflict_statement"], "The authors declare none.",
          "conflict read from back matter")
    counts = chunking.section_counts(document)
    check_true(counts.get(S.METHODS, 0) >= 1, "methods chunk present")
    check_true(counts.get(S.RESULTS, 0) >= 1, "results chunk present")
    check_true(counts.get(S.TABLE, 0) >= 1,
               "floats-group table becomes a chunk")
    joined = " ".join(c["text"] for c in document["chunks"])
    check_true("Someone et al" not in joined, "reference list dropped")
    check_true("[3]" not in joined, "citation marker stripped")
    check_true("North | 42" in joined, "table rows flattened")

    # An article with no body is not full text.
    check(chunking.parse_pmc_article("<article><front/></article>"), None,
          "a record with no body is refused")

    bare = chunking.document_from_text("T", "An abstract with enough text "
                                            "to be a real chunk of prose.")
    check(bare["source_type"], "abstract_only", "text-only document tagged")
    check_true(len(chunking.chunk_document(bare)) == 1,
               "abstract becomes one chunk")

    # Long text: windows overlap, and every offset still points at its text.
    long_doc = chunking.document_from_text("T", long_text)
    long_chunks = chunking.chunk_document(long_doc, "p2", target_tokens=100,
                                          overlap_tokens=20)
    check_true(len(long_chunks) > 3, "a long abstract gives several chunks")
    for chunk in long_chunks:
        block = chunking.clean_text(long_doc["blocks"][0]["text"])
        check(block[chunk["char_start"]:chunk["char_end"]], chunk["text"],
              "offsets survive overlapping windows")
    check_true(chunking.chunker_namespace(100, 20)
               != chunking.chunker_namespace(),
               "chunk parameters name the namespace")


def test_references() -> None:
    print("references")
    cases = [
        ("34203238", {"pmid": "34203238"}),
        ("PMID: 12345678", {"pmid": "12345678"}),
        ("https://pubmed.ncbi.nlm.nih.gov/33036253",
         {"pmid": "33036253", "url": "https://pubmed.ncbi.nlm.nih.gov/33036253"}),
        ("10.3390/v13071258", {"doi": "10.3390/v13071258"}),
        ("PMC8310187", {"pmcid": "PMC8310187"}),
    ]
    for text, want in cases:
        check(fetch.parse_reference(text), want, f"parse {text[:28]!r}")

    refs = fetch.refs_from_lines([
        "34203238", "  34203238  ", "# a comment", "",
        "https://pubmed.ncbi.nlm.nih.gov/33036253",
    ])
    check(len(refs), 2, "duplicates and comments dropped")

    first = fetch.PaperRef(ids={"pmid": "111"})
    second = fetch.PaperRef(ids={"pmid": "111"})
    check(first.key, second.key, "the same paper gets the same key")
    check_true(first.key != fetch.PaperRef(ids={"pmid": "222"}).key,
               "different papers get different keys")


def test_retrieval() -> None:
    print("hybrid retrieval")
    check(index_module.tokenize("SARS-CoV-2 and IL-6 at 0.5 mg"),
          ["sars-cov-2", "il-6", "0.5", "mg"],
          "hyphens and decimals survive tokenizing")

    chunks = [
        {"id": "c1", "section": S.ABSTRACT, "title": "Abstract", "kind": "text",
         "text": "We studied rodents in Poland.", "tokens": 8},
        {"id": "c2", "section": S.METHODS, "title": "Methods", "kind": "text",
         "text": "Animals were screened by RT-qPCR for hantavirus RNA.",
         "tokens": 12},
        {"id": "c3", "section": S.RESULTS, "title": "Results", "kind": "text",
         "text": "Seven of 42 animals were positive, a prevalence of 16.7%.",
         "tokens": 14},
        {"id": "c4", "section": S.DISCUSSION, "title": "Discussion",
         "kind": "text", "text": "The sample size was small.", "tokens": 7},
    ]
    document = {"chunks": chunks, "title": "T", "source_type": "full_text"}
    paper_index = index_module.PaperIndex(document)

    hits = paper_index.search("prevalence positive animals", top_k=4,
                              token_budget=200,
                              always_include_abstract=False)
    check(hits[0].chunk["id"], "c3", "BM25 finds the results chunk")

    boosted = paper_index.search("methods detection", top_k=4,
                                 boost_sections=[S.METHODS],
                                 token_budget=200,
                                 always_include_abstract=False)
    check(boosted[0].chunk["id"], "c2", "the section prior lifts methods")

    with_abstract = paper_index.search("anything", top_k=4, token_budget=200)
    check(with_abstract[0].chunk["id"], "c1", "the abstract always leads")

    budgeted = paper_index.search("prevalence", top_k=4, token_budget=10,
                                  always_include_abstract=False)
    check(len(budgeted), 1, "the token budget is respected")


def test_validation() -> None:
    print("validation")
    check(validate.parse_number("1,234 animals")[0], 1234.0, "commas stripped")
    check(validate.parse_number("16.7%")[0], 16.7, "percent sign stripped")
    check_true(validate.parse_number("7 of 42")[1], "two numbers refused")
    check_true(validate.parse_number("10-20")[1], "a range is refused")
    check(validate.parse_boolean("Yes")[0], True, "yes is true")
    check(validate.parse_boolean("not detected")[0], False, "negation is false")
    check_true(validate.parse_boolean("maybe")[1], "a non-answer is refused")
    check(validate.parse_date("collected in 2019")[0], "2019", "year found")
    check(validate.parse_date("2019-2021")[0], "2019-2021", "year range kept")
    check(validate.match_allowed("pcr", ["PCR", "Serology"])[0], "PCR",
          "case-insensitive match")
    check(validate.match_allowed("Serolgy", ["PCR", "Serology"])[0], "Serology",
          "close spelling matched")
    check_true(validate.match_allowed("RT-qPCR", ["PCR", "Serology"])[1],
               "a different answer is refused, not snapped")

    haystack = ("We tested 42 voles in Poland during 2019. "
                "Seven were positive by PCR.")

    number = ColumnSpec(name="sample size", type="integer", minimum=0)
    good = validate.validate_value(
        number,
        {"value": "42", "evidence_span": "We tested 42 voles in Poland",
         "source_section": "methods", "confidence": 0.9},
        haystack=haystack,
        chunks=[{"id": "c1", "section": "methods", "text": haystack}])
    check(good["validation"], "ok", "a good number validates")
    check(good["value_normalized"], 42, "coerced to an integer")
    check(good["quote_check"], "Yes", "quote verified verbatim")
    check(good["needs_review"], False, "a clean value is not flagged")
    check(good["confidence"], 0.9, "confidence unchanged when all checks pass")

    sentence = validate.validate_value(
        number,
        {"value": "We tested 42 voles in Poland", "evidence_span":
         "We tested 42 voles in Poland", "source_section": "methods",
         "confidence": 0.95},
        haystack=haystack, chunks=[{"id": "c1", "section": "methods",
                                    "text": haystack}])
    check(sentence["validation"], "failed", "a sentence fails a number column")
    check_true(validate.FLAG_FAILED_VALIDATION in sentence["flags"],
               "failed validation flagged")
    check_true(sentence["needs_review"], "failed value goes to review")

    untyped = ColumnSpec(name="sample size")  # type inferred, not declared
    soft = validate.validate_value(
        untyped,
        {"value": "about forty two", "evidence_span": "We tested 42 voles",
         "source_section": "methods", "confidence": 0.8},
        haystack=haystack, chunks=[{"id": "c1", "section": "methods",
                                    "text": haystack}])
    check(soft["validation"], "ok", "an inferred type never rejects a value")
    check_true(validate.FLAG_TYPE_MISMATCH_SOFT in soft["flags"],
               "inferred-type mismatch raises a soft flag")

    invented = validate.validate_value(
        number,
        {"value": "42", "evidence_span": "We tested 900 gerbils in Chile",
         "source_section": "methods", "confidence": 0.99},
        haystack=haystack, chunks=[{"id": "c1", "section": "methods",
                                    "text": haystack}])
    check(invented["quote_check"], "No", "an invented quote is caught")
    check_true(invented["confidence"] < 0.3,
               "a confident claim with no evidence scores low")
    check_true(validate.FLAG_UNVERIFIED_EVIDENCE in invented["flags"],
               "unverifiable evidence flagged")

    missing = validate.validate_value(
        number,
        {"value": "Not stated", "evidence_span": "", "source_section": "",
         "confidence": 0.1},
        haystack=haystack, chunks=[{"id": "c1", "section": "methods",
                                    "text": haystack}])
    check(missing["value"], "Not stated", "a missing value is recorded")
    check(missing["confidence"], 0.0, "a missing value carries no confidence")
    check_true(validate.FLAG_NOT_FOUND in missing["flags"], "not found flagged")
    check_true(validate.FLAG_MISSING_EVIDENCE not in missing["flags"],
               "a missing value is not also asked for a quote")

    ranged = ColumnSpec(name="prevalence", type="number", minimum=0,
                        maximum=100)
    out_of_range = validate.validate_value(
        ranged,
        {"value": "160", "evidence_span": "Seven were positive by PCR",
         "source_section": "results", "confidence": 0.9},
        haystack=haystack, chunks=[{"id": "c1", "section": "results",
                                    "text": haystack}])
    check(out_of_range["validation"], "failed", "out of range fails")

    categorical = ColumnSpec(name="method", allowed_values=["PCR", "Serology"])
    wrong = validate.validate_value(
        categorical,
        {"value": "Electron microscopy",
         "evidence_span": "Seven were positive by PCR",
         "source_section": "results", "confidence": 0.9},
        haystack=haystack, chunks=[{"id": "c1", "section": "results",
                                    "text": haystack}])
    check(wrong["validation"], "failed", "a value outside the list fails")

    patterned = ColumnSpec(name="code", type="text", pattern=r"^[A-Z]{3}-\d+$")
    bad_pattern = validate.validate_value(
        patterned,
        {"value": "abc", "evidence_span": "Seven were positive by PCR",
         "source_section": "results", "confidence": 0.9},
        haystack=haystack, chunks=[{"id": "c1", "section": "results",
                                    "text": haystack}])
    check(bad_pattern["validation"], "failed", "a pattern mismatch fails")

    check(validate.validate_value(
        number, {"value": "42", "evidence_span": "We tested 42 voles",
                 "source_section": "results", "confidence": 85},
        haystack=haystack,
        chunks=[{"id": "c1", "section": "methods", "text": haystack}]
    )["model_confidence"], 0.85, "a confidence of 85 is read as 0.85")


# --------------------------------------------------------------------------- #
# Run store, resume, export - with a fake model
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Answers every column with a value copied out of the passages.

    Counts its calls, and can be told to fail, so the resume test measures the
    checkpoint rather than the model.
    """

    def __init__(self, fail_after: int = 10 ** 9):
        self.calls = 0
        self.fail_after = fail_after

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      *, max_tokens: int = 0) -> LLMResult:
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("stop here")
        payload = {
            key: {"value": "42", "evidence_span": "We tested 42 voles",
                  "source_section": "methods", "confidence": 0.9}
            for key in schema["properties"]
        }
        return LLMResult(text=json.dumps(payload), input_tokens=100,
                         output_tokens=50, cost_usd=0.001, model="fake")

    def warmup(self) -> None:
        return None

    def describe(self) -> str:
        return "fake"


def _fake_document(key: str) -> Dict[str, Any]:
    text = ("We tested 42 voles in Poland during 2019. Seven were positive "
            "by PCR, a prevalence of 16.7 percent.")
    document = chunking.document_from_text(f"Paper {key}", text,
                                           ids={"pmid": key})
    document["paper_key"] = key
    document["article_id"] = key
    document["source_type"] = "full_text"
    document["chunks"] = chunking.chunk_document(document, key)
    document["chunk_count"] = len(document["chunks"])
    document["fetch_notes"] = ""
    document["source_api"] = "test"
    return document


class _FakeCache:
    """The parts of cache.PaperCache that the runner calls."""

    def unknown(self, refs):  # noqa: ANN001
        return []

    def prefetch(self, refs):  # noqa: ANN001
        return None

    def flush(self) -> None:
        return None


def test_run_store_and_resume() -> None:
    print("run store and resume")
    import extract_columns
    import runner as runner_module
    from index import PaperIndex

    root = Path(tempfile.mkdtemp())
    try:
        specs = build_column_set([
            {"name": "sample size", "type": "integer"},
            {"name": "country of sampling"},
            {"name": "primary outcome"},
            {"name": "title"},
        ])
        groups, _ = grouping.plan_groups(specs)
        group_keys = [g.key for g in groups]
        llm_groups = [g for g in groups if not g.is_metadata]

        registry = RunRegistry(root / "runs")
        info = registry.create(specs, {"backend": "fake"}, label="test")
        store = RunStore(registry.run_dir(info.run_id))

        # Enough papers that a stop reliably lands mid-run rather than after
        # the last one.
        paper_count = 10
        refs = [fetch.PaperRef(ids={"pmid": f"{i}"})
                for i in range(paper_count)]
        documents = {r.key: _fake_document(r.ids["pmid"]) for r in refs}

        class FakePaperStore(_FakeCache):
            def index_for(self, ref, fetcher):  # noqa: ANN001
                return PaperIndex(documents[ref.key])

        class FakeFetcher:
            def resolve_batch(self, refs):  # noqa: ANN001
                return None

            def fetch(self, ref):  # noqa: ANN001
                return documents[ref.key]

        # -- first pass, stopped once two papers are stored ---------------- #
        first_model = FakeLLM()

        def should_stop() -> bool:
            return store.paper_count() >= 3

        first = runner_module.ExtractionRunner(
            llm=first_model, fetcher=FakeFetcher(),
            paper_store=FakePaperStore(), run_store=store,
            groups=groups, columns=specs,
            config=runner_module.RunnerConfig(fetch_workers=1,
                                              extract_workers=1),
        )
        first_stats = first.run(refs, should_stop=should_stop)
        check_true(first_stats["stopped"], "the run stopped when asked")
        partial = store.paper_count()
        check_true(0 < partial < paper_count,
                   f"some papers saved before the stop ({partial} of "
                   f"{paper_count})")
        calls_before = first_model.calls

        # -- second pass resumes ------------------------------------------- #
        # A paper can be stopped between its groups, so the work left is
        # counted per (paper, group) rather than per paper. That is exactly
        # the promise: no stored model call is made twice.
        done_before = store.done_groups()
        expected_calls = sum(
            1 for ref in refs for group in llm_groups
            if group.key not in done_before.get(ref.key, set())
        )
        second_model = FakeLLM()
        second = runner_module.ExtractionRunner(
            llm=second_model, fetcher=FakeFetcher(),
            paper_store=FakePaperStore(), run_store=store,
            groups=groups, columns=specs,
            config=runner_module.RunnerConfig(fetch_workers=1,
                                              extract_workers=1),
        )
        second_stats = second.run(refs)
        check(store.paper_count(), paper_count,
              "every paper is done after the resume")
        check(second_stats["skipped"], partial, "finished papers were skipped")
        check(second_stats["done"] + second_stats["skipped"], paper_count,
              "every paper counted exactly once across the resume")
        check(second_model.calls, expected_calls,
              "the resume repeats no model call, down to a part-done paper")
        check_true(calls_before > 0, "the first pass really called the model")

        # -- stored records are complete ----------------------------------- #
        record = store.load_paper(refs[0].key)
        check(sorted(record["values"]), sorted(s.key for s in specs),
              "every column stored for a paper")
        check(record["values"]["sample_size"]["value"], "42",
              "the stored value survives")
        check(record["values"]["title"]["source_section"], "metadata",
              "metadata columns need no model")

        check(store.completed_papers(group_keys), {r.key for r in refs},
              "all papers report complete")

        # -- export --------------------------------------------------------- #
        tables, report = export.build_tables(info, store)
        check(report["papers"], paper_count, "one exported row per paper")
        check(report["evidence_rows"], paper_count * len(specs),
              "one evidence row per value")
        extraction = tables["Extraction"]
        check(list(extraction.columns)[3:7],
              ["sample size", "country of sampling", "primary outcome",
               "title"],
              "the sheet holds the user's columns, in the user's order")
        check(len(extraction), paper_count, "one row per paper")

        columns_only, _ = export.build_tables(info, store,
                                              include_identifiers=False)
        check(list(columns_only["Extraction"].columns),
              ["sample size", "country of sampling", "primary outcome",
               "title"],
              "columns-only mode holds exactly the defined columns")

        # -- a run with different columns cannot be resumed ----------------- #
        other = build_column_set([{"name": "something else"}])
        check(registry.find_matching(other), [],
              "a different column set matches no run")
        check([i.run_id for i in registry.find_matching(specs)],
              [info.run_id], "the same column set finds its run")
        store.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_stop_is_prompt() -> None:
    """A stop must cost about one paper, not one run.

    The extract queue is deliberately kept short. Without that cap every paper
    would be queued within seconds, and shutting the pool down would wait for
    the whole run - so the Stop button and Ctrl-C would both appear to hang.
    """
    print("stop latency")
    import time

    import runner as runner_module
    from index import PaperIndex

    call_seconds = 0.4
    paper_count = 40

    class SlowLLM(FakeLLM):
        def complete_json(self, system, user, schema, *, max_tokens=0):
            time.sleep(call_seconds)
            return super().complete_json(system, user, schema,
                                         max_tokens=max_tokens)

    root = Path(tempfile.mkdtemp())
    try:
        specs = build_column_set([{"name": "sample size", "type": "integer"},
                                  {"name": "primary outcome"}])
        groups, _ = grouping.plan_groups(specs)
        registry = RunRegistry(root / "runs")
        info = registry.create(specs, {}, label="stop")
        store = RunStore(registry.run_dir(info.run_id))

        refs = [fetch.PaperRef(ids={"pmid": str(i)}) for i in range(paper_count)]
        documents = {r.key: _fake_document(r.ids["pmid"]) for r in refs}

        class FakePaperStore(_FakeCache):
            def index_for(self, ref, fetcher):  # noqa: ANN001
                return PaperIndex(documents[ref.key])

        class FakeFetcher:
            def resolve_batch(self, refs):  # noqa: ANN001
                return None

        started = time.time()
        runner = runner_module.ExtractionRunner(
            llm=SlowLLM(), fetcher=FakeFetcher(), paper_store=FakePaperStore(),
            run_store=store, groups=groups, columns=specs,
            config=runner_module.RunnerConfig(fetch_workers=8,
                                              extract_workers=2,
                                              batch_size=200),
        )
        stats = runner.run(refs, should_stop=lambda: time.time() - started > 1.0)
        elapsed = time.time() - started
        whole_run = paper_count * call_seconds / 2

        check_true(stats["stopped"], "the run reports that it stopped")
        check_true(elapsed < whole_run / 3,
                   f"stop returned in {elapsed:.1f}s, well under a full run "
                   f"of about {whole_run:.0f}s")
        check_true(0 < store.paper_count() < paper_count,
                   "some papers were saved before the stop")
        check_true(stats["done"] <= store.paper_count(),
                   f"each paper counted once, not twice "
                   f"(counted {stats['done']}, stored {store.paper_count()})")
        store.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_failed_paper_still_produces_values() -> None:
    print("coverage")
    import runner as runner_module

    root = Path(tempfile.mkdtemp())
    try:
        specs = build_column_set([{"name": "sample size", "type": "integer"},
                                  {"name": "primary outcome"}])
        groups, _ = grouping.plan_groups(specs)
        registry = RunRegistry(root / "runs")
        info = registry.create(specs, {}, label="broken")
        store = RunStore(registry.run_dir(info.run_id))

        class BrokenPaperStore(_FakeCache):
            def index_for(self, ref, fetcher):  # noqa: ANN001
                raise RuntimeError("no text anywhere")

        class FakeFetcher:
            def resolve_batch(self, refs):  # noqa: ANN001
                return None

        runner = runner_module.ExtractionRunner(
            llm=FakeLLM(), fetcher=FakeFetcher(),
            paper_store=BrokenPaperStore(), run_store=store,
            groups=groups, columns=specs,
            config=runner_module.RunnerConfig(fetch_workers=1,
                                              extract_workers=1),
        )
        stats = runner.run([fetch.PaperRef(ids={"pmid": "999"})])
        check(stats["done"], 1, "an unfetchable paper still finishes")
        check(stats["no_text"], 1, "it is counted as having no text")

        tables, report = export.build_tables(info, store)
        check(report["evidence_rows"], 2,
              "a paper with no text still gets a value per column")
        check(report["review_rows"], 2, "and every one goes to review")
        check(tables["Extraction"].iloc[0]["sample size"], "Not stated",
              "the value is 'Not stated', not a blank")
        store.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)
