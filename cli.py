"""Command line for the column extractor and its paper cache.

    # extract your columns from a list of paper links
    python cli.py run -c templates/rodent_virus_example.json -l links.txt

    # extract from a screening export (.xlsx)
    python cli.py run -c my_columns.json -i export.xlsx -s "Included" -n 100

    # show how the columns will be grouped, and stop
    python cli.py plan -c my_columns.json

    # list runs, and continue one
    python cli.py runs
    python cli.py run --run-id <RUN_ID> -l links.txt

    # the paper cache
    python cli.py cache stats
    python cli.py cache verify          # find and repair or quarantine bad entries
    python cli.py cache compact
    python cli.py cache reset --layer embeddings --yes

Safe to interrupt. Rerun the same command to continue: finished papers are
reused from the run, and fetched, chunked and embedded papers from the cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

import cache as cache_module
import column_pipeline
import excel_io
import fetch
from columns import ColumnSpec, build_column_set, load_template
from runstore import RunRegistry

log = logging.getLogger("cli")


def setup_logging(level: str = "INFO", path: Optional[Path] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("urllib3", "httpx", "sentence_transformers", "filelock",
                  "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_columns(spec_path: str) -> List[ColumnSpec]:
    """Read a column set from a template name or a JSON file."""
    path = Path(spec_path)
    if not path.exists():
        return load_template(spec_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("columns", data) if isinstance(data, dict) else data
    return build_column_set(rows)


def load_refs(args: argparse.Namespace) -> List[fetch.PaperRef]:
    refs: List[fetch.PaperRef] = []
    if getattr(args, "links", None):
        text = Path(args.links).read_text(encoding="utf-8")
        refs.extend(fetch.refs_from_lines(text.splitlines()))
    if getattr(args, "input", None):
        sheet: Any = args.sheet
        if sheet is not None and str(sheet).isdigit():
            sheet = int(sheet)
        papers, report = excel_io.read_papers(
            Path(args.input), sheet=sheet if sheet is not None else 0)
        log.info("Loaded %d rows from %s", report["rows_loaded"], args.input)
        refs.extend(fetch.refs_from_records(papers.to_dict("records")))
    # A reference may arrive from both sources; keep the first.
    unique: Dict[str, fetch.PaperRef] = {}
    for ref in refs:
        unique.setdefault(ref.key, ref)
    ordered = list(unique.values())
    limit = getattr(args, "limit", 0)
    return ordered[:limit] if limit else ordered


def settings_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = {
        "cache_dir": args.cache_dir, "runs_dir": args.runs_dir,
        "backend": getattr(args, "backend", None),
        "model": getattr(args, "model", None),
        "host": getattr(args, "host", None),
        "extract_workers": getattr(args, "workers", None),
        "fetch_workers": getattr(args, "fetch_workers", None),
        "ncbi_email": getattr(args, "email", None),
        "confidence_threshold": getattr(args, "min_confidence", None),
        "top_k": getattr(args, "top_k", None),
        "token_budget": getattr(args, "token_budget", None),
        "embedding_provider": getattr(args, "embedding_provider", None),
        "embedding_model": getattr(args, "embedding_model", None),
    }
    return column_pipeline.load_settings(
        {k: v for k, v in overrides.items() if v is not None},
        Path(args.config) if args.config else None)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_runs(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    paths = column_pipeline.Paths.from_settings(settings).ensure()
    registry = RunRegistry(paths.runs)
    runs = registry.list_runs()
    if not runs:
        print("No runs yet.")
        return 0
    for info in runs:
        described = registry.describe(info)
        print(" | ".join(f"{k}: {v}" for k, v in described.items()))
    return 0


def plan_with_cache(specs: List[ColumnSpec], settings: Dict[str, Any],
                    paths: column_pipeline.Paths):
    """Group the columns, with column vectors read from the paper cache.

    A rerun with the same columns then loads no embedding model at all.
    """
    with column_pipeline.make_paper_cache(paths, settings) as paper_cache:
        return column_pipeline.plan(specs, settings, paper_cache.embedder)


def cmd_plan(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    paths = column_pipeline.Paths.from_settings(settings).ensure()
    specs = load_columns(args.columns)
    if not specs:
        log.error("No columns found in %r.", args.columns)
        return 2
    groups, routings = plan_with_cache(specs, settings, paths)
    print(json.dumps({
        "columns": len(specs),
        "model_calls_per_paper": sum(1 for g in groups if not g.is_metadata),
        "groups": [
            {"group": g.key, "llm_call": not g.is_metadata,
             "columns": [c.name for c in g.columns],
             "sections_searched_first": g.sections[:3]}
            for g in groups
        ],
        "routing": [
            {"column": r.name, "group": r.group, "method": r.method,
             "score": r.score, "lead_over_next": r.margin}
            for r in routings
        ],
    }, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    paths = column_pipeline.Paths.from_settings(settings).ensure()
    setup_logging(args.log_level, paths.runs / "extraction.log")
    registry = RunRegistry(paths.runs)

    if not args.columns and not args.run_id:
        log.error("Give --columns (a template name or a JSON file), "
                  "or --run-id to continue an existing run.")
        return 2

    if args.run_id:
        info = registry.load(args.run_id)
        if info is None:
            log.error("No run named %r. Use `python cli.py runs` to see them.",
                      args.run_id)
            return 2
        specs = info.columns
        settings = column_pipeline.settings_for_run(info, settings)
        store = column_pipeline.RunStore(registry.run_dir(info.run_id))
    else:
        specs = load_columns(args.columns)
        if not specs:
            log.error("No columns found in %r.", args.columns)
            return 2
        info, store = column_pipeline.prepare_run(
            specs, settings, paths, label=args.label or "")

    refs = load_refs(args)
    if not refs:
        log.error("No papers to process. Give --links or --input.")
        store.close()
        return 2
    log.info("Run %s: %d papers submitted", info.run_id, len(refs))
    log.info("Paper cache: %s", paths.cache)

    llm = column_pipeline.make_llm(settings)
    bar = tqdm(total=len(refs), desc="Papers", unit="paper")

    def on_progress(record: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        bar.update(1)
        bar.set_postfix_str(
            f"{record.get('source_type', '')} "
            f"flag={record.get('flagged_values', 0)} "
            f"${snapshot['cost_usd']:.3f} "
            f"{snapshot['seconds_per_paper']:.0f}s/paper")

    groups, _ = plan_with_cache(specs, settings, paths)
    llm_calls = sum(1 for g in groups if not g.is_metadata)
    log.info("%d columns -> %d groups (%d model calls per paper)",
             len(specs), len(groups), llm_calls)
    if llm_calls:
        try:
            llm.warmup()
        except Exception as exc:  # noqa: BLE001
            log.error("Model not available: %s", exc)
            store.close()
            return 2
    log.info("Model: %s", llm.describe())

    started = time.perf_counter()
    try:
        stats = column_pipeline.execute_run(
            refs, info, store, settings, paths,
            llm=llm, groups=groups, on_progress=on_progress,
            resume=not args.no_resume)
    except KeyboardInterrupt:
        bar.close()
        log.warning("Interrupted. Progress is saved - rerun to continue.")
        report = column_pipeline.export_results(
            info, store, paths, include_identifiers=not args.columns_only)
        log.info("Partial results written to %s", report["files"]["workbook"])
        store.close()
        return 130
    bar.close()

    report = column_pipeline.export_results(
        info, store, paths, stats=stats,
        include_identifiers=not args.columns_only)
    store.close()

    elapsed = time.perf_counter() - started
    log.info("Done in %.1f min. %d papers, %d values; %d values flagged for "
             "review across %d papers.",
             elapsed / 60, stats["done"] + stats["skipped"],
             report["evidence_rows"], report["review_rows"],
             stats.get("papers_flagged", 0))
    log.info("Full text: %d  Abstract only: %d  No text: %d",
             stats["full_text"], stats["abstract_only"], stats["no_text"])
    log.info("Cost: $%.4f total, $%.5f per paper.",
             stats["cost_usd"], stats["cost_per_paper"])
    cache_report = stats.get("cache", {})
    for layer, counts in cache_report.get("layers", {}).items():
        if counts["hits"] + counts["misses"]:
            log.info("  cache %-14s hits %5d  misses %5d", layer,
                     counts["hits"], counts["misses"])
    log.info("  embedding calls %d (%d chunks), query encode calls %d, "
             "network requests %d",
             cache_report.get("embedding_calls", 0),
             cache_report.get("chunks_embedded", 0),
             cache_report.get("query_encode_calls", 0),
             cache_report.get("network_requests", 0))
    for name, path in report["files"].items():
        log.info("  %-16s %s", name, path)
    if args.stats_json:
        Path(args.stats_json).write_text(json.dumps(
            {"run_id": info.run_id, **stats}, indent=2, default=str),
            encoding="utf-8")
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    root = column_pipeline.Paths.from_settings(settings).cache
    action = args.cache_command

    if action == "path":
        print(root)
        return 0
    if action == "stats":
        print(json.dumps(cache_module.cache_stats(root), indent=2))
        return 0
    if action == "verify":
        report = cache_module.verify_cache(root, repair=not args.no_repair)
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.ok or not args.no_repair else 1
    if action == "compact":
        done = cache_module.compact_cache(root)
        print(f"Compacted {len(done)} vector namespace(s) in {root}.")
        return 0
    if action == "import-raw":
        counts = cache_module.import_raw(root, Path(args.source))
        print(json.dumps(counts, indent=2))
        return 0
    if action == "reset":
        what = ", ".join(args.layer) if args.layer else "the whole cache"
        if not args.yes:
            answer = input(f"Delete {what} in {root}? Type 'yes': ")
            if answer.strip().lower() != "yes":
                print("Nothing deleted.")
                return 1
        for removed in cache_module.reset_cache(root, args.layer or None):
            print(f"Removed {removed}")
        return 0
    raise ValueError(action)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="Path to config.yaml")
    common.add_argument("--cache-dir", help="Paper cache directory")
    common.add_argument("--runs-dir", help="Run output directory")
    common.add_argument("--log-level", default="INFO")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common],
                         help="Extract columns from papers")
    run.add_argument("-c", "--columns",
                     help="Template name, or a path to a column JSON file")
    run.add_argument("-l", "--links", help="Text file of paper links or IDs")
    run.add_argument("-i", "--input", help="Workbook of papers (.xlsx)")
    run.add_argument("-s", "--sheet", help="Sheet name or index")
    run.add_argument("-n", "--limit", type=int, default=0,
                     help="Process only the first N papers")
    run.add_argument("--run-id", help="Continue an existing run")
    run.add_argument("--label", help="A short name for this run")
    run.add_argument("--no-resume", action="store_true",
                     help="Redo extraction already stored in this run. The "
                          "paper cache is still used.")
    run.add_argument("--columns-only", action="store_true",
                     help="Excel sheet holds only the defined columns")
    run.add_argument("--backend", choices=["ollama", "anthropic", "openai",
                                           "llama_cpp"])
    run.add_argument("--model")
    run.add_argument("--host")
    run.add_argument("--workers", type=int, help="Concurrent model calls")
    run.add_argument("--fetch-workers", type=int)
    run.add_argument("--email", help="Contact email sent to NCBI")
    run.add_argument("--min-confidence", type=float)
    run.add_argument("--top-k", type=int)
    run.add_argument("--token-budget", type=int)
    run.add_argument("--embedding-provider")
    run.add_argument("--embedding-model")
    run.add_argument("--stats-json", help="Also write the run totals here")
    run.set_defaults(func=cmd_run)

    plan = sub.add_parser("plan", parents=[common],
                          help="Show how the columns will be grouped")
    plan.add_argument("-c", "--columns", required=True)
    plan.set_defaults(func=cmd_plan)

    runs = sub.add_parser("runs", parents=[common], help="List runs")
    runs.set_defaults(func=cmd_runs)

    cache = sub.add_parser("cache", help="Inspect and maintain the paper cache")
    cache_sub = cache.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("path", parents=[common],
                         help="Print the cache directory")
    cache_sub.add_parser("stats", parents=[common],
                         help="Papers and bytes per layer")
    verify = cache_sub.add_parser(
        "verify", parents=[common],
        help="Check checksums; repair or quarantine bad entries")
    verify.add_argument("--no-repair", action="store_true",
                        help="Report problems only")
    cache_sub.add_parser("compact", parents=[common],
                         help="Merge vector fragments and vacuum the manifest")
    reset = cache_sub.add_parser(
        "reset", parents=[common],
        help="Delete the cache, or some layers and the layers built on them")
    reset.add_argument("--layer", action="append",
                       choices=list(cache_module.LAYERS))
    reset.add_argument("--yes", action="store_true", help="Do not ask")
    imp = cache_sub.add_parser(
        "import-raw", parents=[common],
        help="Import raw downloads from an older cache folder")
    imp.add_argument("source")
    cache.set_defaults(func=cmd_cache)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":
        setup_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
