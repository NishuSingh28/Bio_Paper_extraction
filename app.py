"""Streamlit UI for user-defined column extraction.

Run with:  streamlit run app.py

Four tabs, in the order the work happens: define the columns, give the papers,
run, read the results. The column grid is the whole point - what the user types
there is what comes out of the spreadsheet, with no code change in between.

The run itself happens on a worker thread. Streamlit cannot handle a button
click while the script is blocked, so a blocking run would make the Stop button
dead. A thread plus an auto-refreshing fragment keeps the page responsive and
the Stop button real.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Without a script-run context every st.* call does nothing. That happens when
# the file is run as a plain script (`python app.py`). Relaunch it through
# Streamlit, so either way of starting it works.
if get_script_run_ctx() is None and __name__ == "__main__":
    if os.environ.get("PAPER_EXTRACTOR_RELAUNCHED") == "1":
        sys.stderr.write("\nCould not start under Streamlit. Run it directly:"
                         "\n\n    streamlit run app.py\n\n")
        raise SystemExit(1)
    sys.stderr.write("Starting Streamlit (same as: streamlit run app.py)\n")
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]],
        env={**os.environ, "PAPER_EXTRACTOR_RELAUNCHED": "1"},
    ))

import cache as cache_module
import column_pipeline
import excel_io
import export
import fetch
import grouping
import providers
import validate
from columns import (
    COLUMN_TYPES,
    SECTION_HINTS,
    ColumnSpec,
    DuplicateColumnError,
    build_column_set,
    delete_template,
    list_templates,
    load_template,
    save_template,
    specs_to_rows,
)
from llm import LLMError
from runstore import RunRegistry, RunStore

log = logging.getLogger(__name__)

EDITOR_COLUMNS = ["name", "type", "description", "allowed_values", "unit",
                  "minimum", "maximum", "pattern", "section_hint"]

STARTER_ROWS = [
    {"name": "sample size", "type": "integer",
     "description": "total number of participants, animals or specimens",
     "allowed_values": "", "unit": "", "minimum": 0.0, "maximum": None,
     "pattern": "", "section_hint": "auto"},
    {"name": "study design", "type": "text",
     "description": "the kind of study the authors carried out",
     "allowed_values": "", "unit": "", "minimum": None, "maximum": None,
     "pattern": "", "section_hint": "auto"},
    {"name": "funding source", "type": "text", "description": "who paid for it",
     "allowed_values": "", "unit": "", "minimum": None, "maximum": None,
     "pattern": "", "section_hint": "auto"},
]


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def _state() -> Dict[str, Any]:
    if "cx" not in st.session_state:
        st.session_state["cx"] = {
            "rows": list(STARTER_ROWS),
            "editor_version": 0,
            "refs": [],
            "refs_source": "",
            "run_id": None,
            "worker": None,
            "shared": None,
            "last_stats": None,
            "export_report": None,
        }
    return st.session_state["cx"]


def _paths(settings: Dict[str, Any]) -> column_pipeline.Paths:
    return column_pipeline.Paths.from_settings(settings).ensure()


@st.cache_resource(show_spinner=False)
def _planning_cache(cache_dir: str, provider: str, model: str,
                    revision: str) -> cache_module.PaperCache:
    """A cache handle for planning, so column vectors are cached as well."""
    settings = {"embedding_provider": provider, "embedding_model": model,
                "embedding_revision": revision}
    return cache_module.PaperCache(
        Path(cache_dir), column_pipeline.make_embedder(settings))


def _plan(specs: List[ColumnSpec], settings: Dict[str, Any]):
    handle = _planning_cache(
        str(_paths(settings).cache), str(settings["embedding_provider"]),
        str(settings["embedding_model"]),
        str(settings.get("embedding_revision") or ""))
    return column_pipeline.plan(specs, settings, handle.embedder)


def current_specs(state: Dict[str, Any]) -> List[ColumnSpec]:
    try:
        return build_column_set(state["rows"])
    except DuplicateColumnError as exc:
        st.error(str(exc))
        return []


# --------------------------------------------------------------------------- #
# Tab 1: columns
# --------------------------------------------------------------------------- #
def render_columns_tab(state: Dict[str, Any], settings: Dict[str, Any]) -> None:
    st.subheader("Columns for this run")
    st.caption(
        "Type the column names you want. A type, a description and a list of "
        "allowed values are optional, and each one makes the checks stricter.")

    templates = list_templates()
    load_col, save_col, delete_col = st.columns([2, 2, 1])
    with load_col:
        choice = st.selectbox("Load a template", ["-"] + templates, index=0)
        if choice != "-" and st.button("Load", width="stretch"):
            state["rows"] = specs_to_rows(load_template(choice))
            state["editor_version"] += 1
            st.success(f"Loaded {choice!r}.")
            st.rerun()
    with save_col:
        name = st.text_input("Save this column set as", value="")
        if st.button("Save template", width="stretch", disabled=not name):
            specs = current_specs(state)
            if specs:
                path = save_template(name, specs)
                st.success(f"Saved {len(specs)} columns to {path.name}.")
                st.rerun()
    with delete_col:
        st.write("")
        st.write("")
        if choice != "-" and st.button("Delete", width="stretch"):
            delete_template(choice)
            st.rerun()

    frame = pd.DataFrame(state["rows"], columns=EDITOR_COLUMNS)
    edited = st.data_editor(
        frame,
        num_rows="dynamic",
        width="stretch",
        key=f"column_editor_{state['editor_version']}",
        column_config={
            "name": st.column_config.TextColumn(
                "Column name", required=True,
                help="What you want in the spreadsheet header, for example "
                     "'sample size' or 'funding source'."),
            "type": st.column_config.SelectboxColumn(
                "Type", options=[""] + COLUMN_TYPES,
                help="Optional. Leave blank and the value is kept as text; a "
                     "type you choose here is enforced."),
            "description": st.column_config.TextColumn(
                "Description", width="large",
                help="Optional. Tells the model what the column means."),
            "allowed_values": st.column_config.TextColumn(
                "Allowed values",
                help="Optional, comma separated. A value outside this list "
                     "fails validation and goes to the review queue."),
            "unit": st.column_config.TextColumn("Unit"),
            "minimum": st.column_config.NumberColumn("Min"),
            "maximum": st.column_config.NumberColumn("Max"),
            "pattern": st.column_config.TextColumn(
                "Pattern", help="Optional regular expression the value must "
                                "match."),
            "section_hint": st.column_config.SelectboxColumn(
                "Section", options=SECTION_HINTS, default="auto",
                help="Leave on 'auto' unless a column is routed wrongly."),
        },
    )
    state["rows"] = edited.where(pd.notna(edited), None).to_dict("records")

    specs = current_specs(state)
    if not specs:
        st.info("Add at least one column.")
        return

    st.divider()
    st.subheader("How these columns will be extracted")
    embedder = column_pipeline.make_embedder(settings)
    if not embedder.available:
        st.warning(
            "The embedding model is not loaded, so columns are grouped by "
            "keyword alone and retrieval uses BM25 only. "
            f"{embedder.error or ''}")

    groups, routings = _plan(specs, settings)
    llm_calls = sum(1 for g in groups if not g.is_metadata)

    c1, c2, c3 = st.columns(3)
    c1.metric("Columns", len(specs))
    c2.metric("Groups", len(groups))
    c3.metric("Model calls per paper", llm_calls,
              help="Columns answered from the source record need no model "
                   "call at all.")

    st.dataframe(pd.DataFrame(grouping.describe_plan(groups)),
                 width="stretch", hide_index=True)

    with st.expander("Why each column was routed there", expanded=False):
        st.dataframe(pd.DataFrame([
            {"Column": r.name, "Group": r.group, "How": r.method,
             "Score": r.score, "Lead over next": r.margin}
            for r in routings
        ]), width="stretch", hide_index=True)
        st.caption(
            "A small lead means the routing was close. The section is only a "
            "search preference, so a close call costs rank, not recall. Set "
            "the Section column above to override one.")


# --------------------------------------------------------------------------- #
# Tab 2: papers
# --------------------------------------------------------------------------- #
def render_papers_tab(state: Dict[str, Any]) -> None:
    st.subheader("Papers for this run")
    st.caption("Paste links or identifiers, or upload a list or a screening "
               "export. PMIDs, PMCIDs, DOIs and PubMed URLs all work.")

    paste_tab, upload_tab, workbook_tab = st.tabs(
        ["Paste links", "Upload a list", "Upload a workbook"])

    refs: List[fetch.PaperRef] = []
    source = ""

    with paste_tab:
        text = st.text_area("One paper per line", height=200,
                            placeholder="34203238\n10.3390/v13071258\n"
                                        "https://pubmed.ncbi.nlm.nih.gov/33036253\n"
                                        "PMC7599905")
        if text.strip():
            refs = fetch.refs_from_lines(text.splitlines())
            source = "pasted list"

    with upload_tab:
        uploaded = st.file_uploader("A .txt or .csv file of links",
                                    type=["txt", "csv"], key="links_file")
        if uploaded is not None:
            content = uploaded.getvalue().decode("utf-8", errors="replace")
            refs = fetch.refs_from_lines(content.splitlines())
            source = uploaded.name

    with workbook_tab:
        book = st.file_uploader("A screening export (.xlsx)", type=["xlsx"],
                                key="papers_workbook")
        if book is not None:
            try:
                book.seek(0)
                sheets = excel_io.list_sheets(book)
                sheet = (st.selectbox("Sheet", sheets, index=0)
                         if len(sheets) > 1 else sheets[0])
                book.seek(0)
                papers, report = excel_io.read_papers(book, sheet=sheet)
                refs = fetch.refs_from_records(papers.to_dict("records"))
                source = f"{book.name} [{sheet}]"
                st.caption(f"{report['rows_loaded']} rows read. "
                           f"Detected columns: "
                           f"{', '.join(report['column_mapping'])}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read the workbook: {exc}")

    if refs:
        state["refs"] = refs
        state["refs_source"] = source

    stored = state.get("refs") or []
    if not stored:
        st.info("No papers loaded yet.")
        return

    st.success(f"{len(stored)} papers loaded from {state['refs_source']}.")
    kinds = {"PMID": 0, "PMCID": 0, "DOI": 0, "URL only": 0, "No identifier": 0}
    for ref in stored:
        if ref.ids.get("pmcid"):
            kinds["PMCID"] += 1
        elif ref.ids.get("pmid"):
            kinds["PMID"] += 1
        elif ref.ids.get("doi"):
            kinds["DOI"] += 1
        elif ref.ids.get("url"):
            kinds["URL only"] += 1
        else:
            kinds["No identifier"] += 1
    columns = st.columns(len(kinds))
    for column, (label, count) in zip(columns, kinds.items()):
        column.metric(label, count)

    st.caption(
        "A paper with no PMCID may still have full text; that is resolved "
        "when the run starts. Anything without full text is extracted from "
        "its abstract and tagged `abstract_only`.")
    with st.expander("First 20 papers", expanded=False):
        st.dataframe(pd.DataFrame([
            {"Input": r.raw[:70], "PMID": r.ids.get("pmid", ""),
             "PMCID": r.ids.get("pmcid", ""), "DOI": r.ids.get("doi", ""),
             "Title": r.title[:60]}
            for r in stored[:20]
        ]), width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 3: run
# --------------------------------------------------------------------------- #
def _start_worker(state: Dict[str, Any], specs: List[ColumnSpec],
                  refs: List[fetch.PaperRef], settings: Dict[str, Any],
                  paths: column_pipeline.Paths,
                  resume_run_id: Optional[str], resume: bool) -> None:
    """Run the extraction on a worker thread so the page stays responsive."""
    shared: Dict[str, Any] = {
        "stop": False, "snapshot": {}, "last": {}, "done": False,
        "error": None, "run_id": None, "stats": None,
    }

    try:
        info, store = column_pipeline.prepare_run(
            specs, settings, paths, reuse_run_id=resume_run_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    if resume_run_id:
        settings = column_pipeline.settings_for_run(info, settings)
    shared["run_id"] = info.run_id
    state["run_id"] = info.run_id
    groups, _ = _plan(specs, settings)
    refs = list(refs)

    def on_progress(record: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        shared["snapshot"] = snapshot
        shared["last"] = {
            "Paper": record.get("article_id") or record.get("paper_key"),
            "Title": (record.get("title") or "")[:70],
            "Source": record.get("source_type"),
            "Flagged": record.get("flagged_values"),
        }

    def work() -> None:
        try:
            stats = column_pipeline.execute_run(
                refs, info, store, settings, paths,
                groups=groups, on_progress=on_progress,
                should_stop=lambda: bool(shared["stop"]),
                resume=resume,
            )
            shared["stats"] = stats
            shared["export"] = column_pipeline.export_results(
                info, store, paths, stats=stats,
                include_identifiers=bool(settings.get("include_identifiers",
                                                      True)))
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            log.exception("Run failed")
            shared["error"] = str(exc)
        finally:
            shared["done"] = True
            store.close()

    thread = threading.Thread(target=work, name="extraction-run", daemon=True)
    thread.start()
    state["worker"] = thread
    state["shared"] = shared


def render_live_status(state: Dict[str, Any]) -> None:
    """Draw the run status.

    While the worker runs this goes through an auto-refreshing fragment, so
    Streamlit can still handle the Stop button. Once the run is over the
    fragment would keep waking the page for nothing, so the final status is
    drawn straight from the script instead.
    """
    shared = state.get("shared")
    if not shared:
        return
    if shared.get("done"):
        _draw_status(state, shared)
    else:
        _live_status_fragment(state)


@st.fragment(run_every="2s")
def _live_status_fragment(state: Dict[str, Any]) -> None:
    shared = state.get("shared")
    if shared:
        _draw_status(state, shared)


def _draw_status(state: Dict[str, Any], shared: Dict[str, Any]) -> None:
    snapshot = shared.get("snapshot") or {}
    done = int(snapshot.get("done", 0))
    skipped = int(snapshot.get("skipped", 0))
    total = int(snapshot.get("total", 0)) or 1
    finished = bool(shared.get("done"))

    st.progress(min(1.0, (done + skipped) / total),
                text=f"{done + skipped} of {total} papers")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Papers done", done + skipped)
    c2.metric("Papers for review", snapshot.get("papers_flagged", 0),
              help=f"{snapshot.get('values_flagged', 0)} individual values "
                   f"are flagged across them.")
    c3.metric("Full text", snapshot.get("full_text", 0))
    c4.metric("Per paper",
              f"{snapshot.get('processing_seconds_per_paper', 0):.0f} s",
              help="Processing time for one paper. Wall-clock time is lower "
                   "when several model calls run at once.")
    cost = float(snapshot.get("cost_usd", 0.0))
    c5.metric("Cost", "local model" if cost == 0 else f"${cost:.2f}")

    if not finished:
        eta = float(snapshot.get("eta_s", 0.0))
        st.caption(f"About {eta / 60:.0f} min left. "
                   f"Papers with errors: {snapshot.get('failed', 0)}.")
        if shared.get("last"):
            st.caption(" | ".join(f"{k}: {v}"
                                  for k, v in shared["last"].items()))
        if st.button("Stop after the papers in flight", type="secondary"):
            shared["stop"] = True
            st.warning("Stopping. Everything finished so far is saved.")
    else:
        if shared.get("error"):
            st.error(f"The run failed: {shared['error']}")
        elif snapshot.get("stopped"):
            st.warning("Stopped. Press Start again to continue where this "
                       "left off.")
        else:
            st.success("Run finished.")
        state["last_stats"] = shared.get("stats")
        state["export_report"] = shared.get("export")
        state["worker"] = None
        # Leave `shared` in place so the final numbers stay on screen.


def render_run_tab(state: Dict[str, Any], settings: Dict[str, Any]) -> None:
    st.subheader("Run")
    specs = current_specs(state)
    refs = state.get("refs") or []
    paths = _paths(settings)

    if not specs:
        st.info("Define at least one column first.")
        return
    if not refs:
        st.info("Load some papers first.")
        return

    registry = RunRegistry(paths.runs)
    matching = registry.find_matching(specs)
    resume_id: Optional[str] = None
    resume = True

    if matching:
        described = [registry.describe(info) for info in matching]
        st.caption("Earlier runs with exactly these columns:")
        st.dataframe(pd.DataFrame(described), width="stretch", hide_index=True)
        choice = st.selectbox(
            "Continue one of them, or start a new run",
            ["Start a new run"] + [info.run_id for info in matching])
        if choice != "Start a new run":
            resume_id = choice
    else:
        st.caption("No earlier run has this column set, so this starts a new "
                   "one.")

    left, right = st.columns(2)
    with left:
        limit = st.number_input(
            "Papers to process (0 = all)", 0, len(refs), 0,
            help="Start small to check the output before committing to the "
                 "whole list. Work already done is reused.")
        resume = st.checkbox(
            "Reuse work already stored", value=True,
            help="Off means every paper is extracted again, even if a stored "
                 "result exists.")
    with right:
        st.metric("Papers selected", int(limit) or len(refs))
        groups, _ = _plan(specs, settings)
        calls = sum(1 for g in groups if not g.is_metadata)
        st.metric("Model calls per paper", calls)

    if settings.get("backend") == "ollama":
        seconds = (int(limit) or len(refs)) * calls * 45 / max(
            1, int(settings.get("extract_workers", 2)))
        st.caption(
            f"A local model takes roughly 45 s per call, so this is about "
            f"{seconds / 3600:.1f} h. Progress is saved after every call, so "
            f"stopping is safe.")
    else:
        st.caption(
            "Cost is reported live while the run goes. Start with a small "
            "number of papers to measure it before the whole list.")

    running = state.get("worker") is not None and state["worker"].is_alive()
    if st.button("Start extraction", type="primary", width="stretch",
                 disabled=running):
        selected = refs if not limit else refs[:int(limit)]
        _start_worker(state, specs, selected, settings, paths, resume_id,
                      resume)

    if state.get("shared"):
        st.divider()
        render_live_status(state)


# --------------------------------------------------------------------------- #
# Tab 4: results
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading the run...")
def _load_tables_cached(run_id: str, runs_dir: str, include_identifiers: bool,
                        _stamp: float
                        ) -> Optional[tuple[Dict[str, pd.DataFrame],
                                            Dict[str, Any]]]:
    """Cached on the results file's timestamp, so a finished run loads once."""
    return _load_tables(run_id, Path(runs_dir),
                        {"include_identifiers": include_identifiers})


def _load_tables(run_id: str, runs_dir: Path, settings: Dict[str, Any]
                 ) -> Optional[tuple[Dict[str, pd.DataFrame], Dict[str, Any]]]:
    registry = RunRegistry(runs_dir)
    info = registry.load(run_id)
    if info is None:
        return None
    store = RunStore(registry.run_dir(run_id))
    try:
        tables, report = export.build_tables(
            info, store,
            include_identifiers=bool(settings.get("include_identifiers", True)),
            stats=store.get_meta("last_run_stats", {}),
        )
    finally:
        store.close()
    return tables, report


def render_results_tab(state: Dict[str, Any], settings: Dict[str, Any]) -> None:
    st.subheader("Results")
    paths = _paths(settings)
    registry = RunRegistry(paths.runs)
    runs = registry.list_runs()
    if not runs:
        st.info("No runs yet.")
        return

    ids = [info.run_id for info in runs]
    default = state.get("run_id") if state.get("run_id") in ids else ids[0]
    run_id = st.selectbox("Run", ids, index=ids.index(default))

    database = registry.run_dir(run_id) / "results.sqlite"
    stamp = database.stat().st_mtime if database.exists() else 0.0
    loaded = _load_tables_cached(
        run_id, str(paths.runs),
        bool(settings.get("include_identifiers", True)), stamp)
    if loaded is None:
        st.error("That run could not be read.")
        return
    tables, report = loaded

    if report["papers"] == 0:
        st.info("This run has no results yet.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Papers", report["papers"])
    c2.metric("Values", report["evidence_rows"])
    c3.metric("Needing review", report["review_rows"])
    share = report["review_rows"] / max(1, report["evidence_rows"])
    c4.metric("Share flagged", f"{100 * share:.0f}%")

    if report["evidence_truncated_in_workbook"]:
        st.warning(
            "This run has more values than a spreadsheet can hold. The "
            "Extraction sheet is complete; the full evidence table is in the "
            "CSV beside it in the run folder.")

    tab_extraction, tab_evidence, tab_review, tab_papers, tab_info = st.tabs(
        ["Extraction", "Evidence", "Review queue", "Papers", "Run info"])

    with tab_extraction:
        st.caption("One row per paper, with exactly the columns you defined.")
        st.dataframe(tables["Extraction"], width="stretch", height=460)
    with tab_evidence:
        st.caption("One row per value: the sentence behind it, the section it "
                   "came from, and the confidence after the checks.")
        only_low = st.checkbox("Only values below the confidence threshold")
        frame = tables["Evidence"]
        if only_low:
            threshold = float(settings.get(
                "confidence_threshold", validate.DEFAULT_CONFIDENCE_THRESHOLD))
            frame = frame[frame["Confidence"] < threshold]
        st.dataframe(frame, width="stretch", height=460)
    with tab_review:
        st.caption(
            "Low confidence, failed a check, missing or unverifiable "
            "evidence, or not found. These values are in the sheet; they are "
            "the ones to read before using it.")
        st.dataframe(tables["Review queue"], width="stretch", height=460)
    with tab_papers:
        st.dataframe(tables["Papers"], width="stretch", height=460)
    with tab_info:
        st.dataframe(tables["Run info"], width="stretch", height=400,
                     hide_index=True)

    st.divider()
    workbook = export.workbook_bytes(tables)
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download Excel", data=workbook,
        file_name=f"{run_id}_extraction.xlsx",
        mime="application/vnd.openxmlformats-officedocument."
             "spreadsheetml.sheet",
        width="stretch", type="primary")
    d2.download_button(
        "Download Extraction CSV",
        data=tables["Extraction"].to_csv(index=False).encode("utf-8"),
        file_name=f"{run_id}_extraction.csv", mime="text/csv",
        width="stretch")
    d3.download_button(
        "Download review queue CSV",
        data=tables["Review queue"].to_csv(index=False).encode("utf-8"),
        file_name=f"{run_id}_review_queue.csv", mime="text/csv",
        width="stretch")
    st.caption(f"Run folder: `{registry.run_dir(run_id)}`")


# --------------------------------------------------------------------------- #
# Sidebar settings
# --------------------------------------------------------------------------- #
def render_settings(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Model and run settings. Returns the merged settings dictionary."""
    from llm import list_ollama_models

    settings = column_pipeline.merge_settings(None, defaults)

    with st.sidebar:
        st.header("Model")
        backend = st.selectbox(
            "Backend", ["ollama", "anthropic", "openai"],
            index=["ollama", "anthropic", "openai"].index(
                settings.get("backend", "ollama"))
            if settings.get("backend") in ("ollama", "anthropic", "openai")
            else 0,
        )
        settings["backend"] = backend

        if backend == "ollama":
            settings["host"] = st.text_input(
                "Ollama host", value=settings.get("host",
                                                  "http://localhost:11434"))
            available = list_ollama_models(settings["host"])
            preferred = settings.get("model", "qwen2.5:7b")
            if available:
                index = (available.index(preferred)
                         if preferred in available else 0)
                settings["model"] = st.selectbox("Model", available,
                                                 index=index)
            else:
                settings["model"] = st.text_input("Model", value=preferred)
                st.warning("Ollama is not reachable. Start it with "
                           "`ollama serve`.")
        else:
            models = providers.available_models(backend)
            settings["model"] = st.selectbox("Model", models, index=0)
            settings["api_key"] = st.text_input(
                "API key (blank uses the environment)", type="password",
                value="")
            if backend == "anthropic":
                settings["effort"] = st.select_slider(
                    "Effort", options=["low", "medium", "high"],
                    value=settings.get("effort", "low"),
                    help="Low is right for reading a passage and reporting "
                         "what it says. Raise it for columns that need real "
                         "reasoning.")
            price = providers.PRICING.get(settings["model"])
            if price:
                st.caption(f"${price[0]:.2f} per million input tokens, "
                           f"${price[1]:.2f} per million output tokens.")

        with st.expander("Retrieval and validation", expanded=False):
            settings["top_k"] = int(st.number_input(
                "Passages per model call", 3, 40,
                int(settings.get("top_k", 12))))
            settings["token_budget"] = int(st.number_input(
                "Token budget per model call", 600, 20000,
                int(settings.get("token_budget", 2600)), 200))
            settings["max_columns_per_call"] = int(st.number_input(
                "Most columns in one call", 2, 20,
                int(settings.get("max_columns_per_call", 8)),
                help="A larger number means fewer calls and a longer answer "
                     "for the model to keep straight."))
            settings["confidence_threshold"] = float(st.slider(
                "Confidence below which a value is flagged", 0.0, 1.0,
                float(settings.get("confidence_threshold", 0.6)), 0.05))
            settings["max_retries"] = int(st.number_input(
                "Retries per model call", 0, 3,
                int(settings.get("max_retries", 1))))

        with st.expander("Speed and fetching", expanded=False):
            settings["extract_workers"] = int(st.number_input(
                "Concurrent model calls", 1, 32,
                int(settings.get("extract_workers", 2))))
            settings["fetch_workers"] = int(st.number_input(
                "Concurrent downloads", 1, 32,
                int(settings.get("fetch_workers", 6))))
            settings["batch_size"] = int(st.number_input(
                "Papers per batch", 20, 2000,
                int(settings.get("batch_size", 200)), 20,
                help="Controls memory. It does not change the result."))
            settings["ncbi_email"] = st.text_input(
                "Contact email for NCBI", value=settings.get("ncbi_email", ""),
                help="NCBI asks for one. With an API key as well you may make "
                     "ten requests a second instead of three.")
            settings["ncbi_api_key"] = st.text_input(
                "NCBI API key (optional)", type="password",
                value=settings.get("ncbi_api_key", ""))

        with st.expander("Output", expanded=False):
            settings["include_identifiers"] = st.checkbox(
                "Include Paper ID, Title and flags in the Extraction sheet",
                value=bool(settings.get("include_identifiers", True)),
                help="Turn this off for a sheet holding only the columns you "
                     "defined.")

        with st.expander("Paper cache", expanded=False):
            settings["cache_dir"] = st.text_input(
                "Cache folder", value=str(settings.get("cache_dir")),
                help="Any local folder, for example one that a sync tool "
                     "backs up. Set PAPER_CACHE_DIR in .env to change the "
                     "default.")
            settings["embedding_model"] = st.text_input(
                "Embedding model", value=str(settings.get("embedding_model")),
                help="A different model writes to a new cache namespace. "
                     "Vectors from two models are never mixed.")
            paths = _paths(settings)
            embedder = column_pipeline.make_embedder(settings)
            try:
                summary = cache_module.cache_stats(paths.cache)
                embedded = sum(
                    int(r["papers"] or 0) for r in summary["layers"]
                    if r["layer"] == "embeddings" and r["status"] == "ok"
                    and r["namespace"].endswith("__" + embedder.namespace))
                st.caption(
                    f"{summary['papers']} papers parsed, {embedded} embedded "
                    f"with this model. Another run over them costs no "
                    f"download and no embedding. "
                    f"{summary['bytes_on_disk'] / 1e6:.1f} MB on disk.")
            except Exception as exc:  # noqa: BLE001 - informational only
                st.caption(f"Cache not readable: {exc}")
            st.caption(f"`{paths.cache}`")

        if st.button("Check the model", width="stretch"):
            try:
                column_pipeline.make_llm(settings).warmup()
                st.success("The model answered.")
            except LLMError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"{exc}")

    return settings


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render(defaults: Dict[str, Any]) -> None:
    """Draw the page. `defaults` is the flat config (config.yaml + .env)."""
    state = _state()
    settings = render_settings(defaults)

    st.title("Column extractor")
    st.caption(
        "Define the columns you want, give a list of papers, and get a "
        "spreadsheet with one row per paper and a value in every column - "
        "each with the sentence behind it and a confidence.")

    tabs = st.tabs(["1 Columns", "2 Papers", "3 Run", "4 Results"])
    with tabs[0]:
        render_columns_tab(state, settings)
    with tabs[1]:
        render_papers_tab(state)
    with tabs[2]:
        render_run_tab(state, settings)
    with tabs[3]:
        render_results_tab(state, settings)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    st.set_page_config(page_title="Paper column extractor", page_icon="📄",
                       layout="wide")
    column_pipeline.load_dotenv_file()
    render(column_pipeline.load_config())


if get_script_run_ctx() is not None:
    main()
