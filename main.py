"""
Testing-stage / end-of-day batch entry point -- run this against files (or
whole folders) on disk, no Kafka required. stream_main.py is the separate
real-time entry point.

By default (no --live/--mock) this runs the fully deterministic half of
the pipeline -- parse, classify, pre-filter -- and prints what WOULD be
sent to the LLM agents, so you can sanity-check volume/cost before ever
spending a token.

Usage:
    python main.py                                   # sample_data/, dry run
    python main.py --live                             # sample_data/, real LLM calls
    python main.py --zc mylogs/zc.csv --ats mylogs/ats.log --live
                                                        # single files
    python main.py --zc-dir "E:\\Logs\\ZC" --es-dir "E:\\Logs\\ES" --ats-dir "E:\\Logs\\ATS" --live
                                                        # END-OF-DAY BATCH MODE:
                                                        # every file in each folder is
                                                        # parsed and combined into ONE
                                                        # consolidated run/report set
    python main.py --skip-es --live                   # only test ZC + ATS
    python main.py --zc mylogs/zc.csv --skip-es --skip-ats --show-flagged
                                                        # NO API KEY NEEDED: preview what
                                                        # would be sent to an LLM
    python main.py --zc mylogs/zc.csv --skip-es --skip-ats --mock
                                                        # NO API KEY, FREE: full pipeline
                                                        # with a fake rule-based LLM
    provider              # send EVERYTHING (incl. INFO) to
                                                        # the LLM -- no cost-saving filter.
                                                        # Real option, not a bug -- much
                                                        # higher API usage than the default.
    python main.py --provider ollama --live            # use Ollama (glm-5.2:cloud) instead
                                                        # of Claude -- see llm_client.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mappings import MappingStore
from models import System, Severity
from parsers import parse_zc_csv, parse_ats_file, parse_es_file
from severity_rules import classify_all, prefilter


def _collect_files(directory: str, pattern: str) -> list[str]:
    p = Path(directory)
    if not p.is_dir():
        print(f"[!] Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)
    files = sorted(str(f) for f in p.glob(pattern) if f.is_file())
    if not files:
        print(f"[!] No files matching '{pattern}' found in {directory}", file=sys.stderr)
        sys.exit(1)
    return files


def _parse_all(paths: list[str], parse_fn, mapping_store: MappingStore | None) -> list:
    """Parses every file in `paths` with parse_fn and concatenates the
    results -- this is what makes folder/batch mode work: N files in,
    one combined event list out, same as if it were all one file."""
    events = []
    for path in paths:
        if mapping_store is not None:
            events.extend(parse_fn(path, mapping_store))
        else:
            events.extend(parse_fn(path))
    return events


def dry_run(
    mapping_store: MappingStore,
    zc_paths: list[str], es_paths: list[str], ats_paths: list[str],
    show_flagged: bool = False, show_all: bool = False,
    min_severity: Severity = Severity.WARNING,
) -> dict:
    raw_events = {}

    if zc_paths:
        raw_events[System.ZC.value] = classify_all(_parse_all(zc_paths, parse_zc_csv, mapping_store))
    if es_paths:
        raw_events[System.ES.value] = classify_all(_parse_all(es_paths, parse_es_file, mapping_store))
    if ats_paths:
        raw_events[System.ATS.value] = classify_all(_parse_all(ats_paths, parse_ats_file, None))

    if not raw_events:
        print("[!] No log files given -- use --zc/--es/--ats, --zc-dir/--es-dir/--ats-dir, "
              "or drop the --skip-* flags.", file=sys.stderr)
        sys.exit(1)

    print("=== Parse + classify summary ===")
    if min_severity == Severity.INFO:
        print("  [!] --no-filter is ON: ALL severities will be sent to the LLM, including INFO.")
    total_raw = 0
    total_flagged = 0
    flagged_by_system = {}
    for sys_name, events in raw_events.items():
        flagged = prefilter(events, min_severity=min_severity)
        flagged_by_system[sys_name] = flagged
        total_raw += len(events)
        total_flagged += len(flagged)
        pct = (len(flagged) / len(events) * 100) if events else 0
        print(f"  {sys_name}: {len(events)} parsed -> {len(flagged)} would reach an LLM ({pct:.1f}%)")

    if total_raw:
        print(f"\nTotal: {total_raw} log lines parsed, {total_flagged} would be sent to "
              f"LLM agents ({total_flagged/total_raw*100:.1f}%).")

    if show_all:
        print("\n=== ALL parsed events (including INFO, not just flagged) ===")
        for sys_name, events in raw_events.items():
            print(f"\n--- {sys_name} ({len(events)} events) ---")
            for e in events:
                print(f"  [{e.severity.value:8s}] {e.timestamp} "
                      f"machine={e.machine} process={e.process} module={e.module} "
                      f":: {e.message[:120]}")
    elif show_flagged:
        print(f"\n=== Flagged events (what would reach an LLM) ===")
        for sys_name, flagged in flagged_by_system.items():
            print(f"\n--- {sys_name} ({len(flagged)} flagged) ---")
            for e in flagged:
                print(f"  [{e.severity.value:8s}] {e.timestamp} "
                      f"machine={e.machine} process={e.process} module={e.module} "
                      f":: {e.message[:200]}")
            if not flagged:
                print("  (none)")

    return raw_events


def live_run(
    mapping_store: MappingStore, raw_events: dict, use_mock: bool = False,
    min_severity: Severity = Severity.WARNING, provider: str = "anthropic",
    memory_path: str | None = "memory/incident_history.json",
    output_dir: str | None = None,
    excel_path: str | None = None,
    run_meta: dict | None = None,
) -> None:
    from graph import build_graph
    from agents.report_agent import render_markdown

    memory_store = None
    if memory_path:
        from memory_store import MemoryStore
        memory_store = MemoryStore(memory_path)
        print(f"[i] Using persistent memory: {memory_path} "
              f"({len(memory_store._records)} past incident(s) on file)")

    if use_mock:
        from mock_llm_client import MockLLMClient
        print("\n[!] Using --mock: rule-based grouping, NOT real analysis. "
              "See mock_llm_client.py for what this does and doesn't prove.\n")
        client = MockLLMClient()
    else:
        from llm_client import LLMClient
        client = LLMClient(provider=provider)
        print(f"[i] Using provider: {client.provider} (model: {client.model_name})")

    graph = build_graph(mapping_store, client, min_severity=min_severity, memory_store=memory_store)

    result = graph.invoke({"raw_events": raw_events})

    reports = result.get("reports", [])
    print(f"\n=== {len(reports)} report(s) generated ===\n")
    if not reports:
        print("No events crossed the reporting threshold -- nothing to show. "
              "This is expected if your log sample is clean.")
    for i, r in enumerate(reports, start=1):
        md = render_markdown(r)
        print(md)
        print("\n" + "=" * 80 + "\n")
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            out_path = Path(output_dir) / f"report_{i:02d}_{r.incident_id}.md"
            out_path.write_text(md, encoding="utf-8")
    if output_dir and reports:
        print(f"[i] Wrote {len(reports)} report file(s) to {output_dir}/")

    if excel_path and reports:
        # Append THIS run's reports to the single accumulating workbook -- the
        # same .xlsx is reused every run (never overwritten), so history grows.
        from excel_writer import append_run, report_count
        existing = report_count(excel_path)
        run_id = append_run(excel_path, reports, run_meta or {})
        print(f"[i] Appended {len(reports)} report(s) to {excel_path} "
              f"(run {run_id}; workbook now has {existing + len(reports)} incident rows)")


def _resolve_paths(single: str | None, directory: str | None, pattern: str,
                    label: str, skip: bool) -> list[str]:
    if skip:
        return []
    if directory:
        return _collect_files(directory, pattern)
    if single:
        if not os.path.exists(single):
            print(f"[!] --{label} path not found: {single}", file=sys.stderr)
            sys.exit(1)
        return [single]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Run with real LLM API calls")
    parser.add_argument("--mock", action="store_true", help="Run with a FREE rule-based fake LLM")

    parser.add_argument("--zc", default="sample_data/zc_log.csv", help="Single ZC log CSV")
    parser.add_argument("--es", default="sample_data/es_general.log", help="Single ES log file")
    parser.add_argument("--ats", default="sample_data/ats_trace.log", help="Single ATS trace log")
    parser.add_argument("--zc-dir", help="Folder of ZC log CSVs -- BATCH MODE, all files combined")
    parser.add_argument("--es-dir", help="Folder of ES log files -- BATCH MODE, all files combined")
    parser.add_argument("--ats-dir", help="Folder of ATS trace logs -- BATCH MODE, all files combined")
    parser.add_argument("--zc-pattern", default="*.csv", help="Glob pattern for --zc-dir (default *.csv)")
    parser.add_argument("--es-pattern", default="*.log", help="Glob pattern for --es-dir (default *.log)")
    parser.add_argument("--ats-pattern", default="*.log", help="Glob pattern for --ats-dir (default *.log)")

    parser.add_argument("--mapping-csv", default="sample_data/all_ids_mapping.csv")
    parser.add_argument("--skip-zc", action="store_true")
    parser.add_argument("--skip-es", action="store_true")
    parser.add_argument("--skip-ats", action="store_true")

    parser.add_argument("--show-flagged", action="store_true",
                         help="Print what would reach an LLM, no API key needed")
    parser.add_argument("--show-all", action="store_true",
                         help="Print every parsed event including INFO, no API key needed")

    parser.add_argument("--no-filter", action="store_true",
                         help="Send ALL severities (including INFO) to the LLM -- disables "
                              "the cost-saving pre-filter. Real option, much higher API usage.")

    parser.add_argument("--provider", choices=["anthropic", "ollama", "glm"], default="anthropic",
                         help="Which LLM backend to use: anthropic (='claude'), ollama, or glm "
                              "(Zhipu, OpenAI-compatible -- see llm_client.py)")

    parser.add_argument("--memory-path", default="memory/incident_history.json",
                         help="Path to the persistent cross-run incident history file")
    parser.add_argument("--no-memory", action="store_true",
                         help="Disable historical/accumulative memory for this run")

    parser.add_argument("--output-dir",
                         help="If set, also write each generated report as a .md file here "
                              "(in addition to printing) -- useful for end-of-day batch runs")

    parser.add_argument("--excel-path",
                         help="If set, APPEND this run's reports to a single accumulating "
                              ".xlsx workbook at this path (created on first run, reused every "
                              "run -- never overwritten). Pair with --output-dir to keep the "
                              "workbook inside the same folder, e.g. reports/incident_reports.xlsx")

    args = parser.parse_args()

    if args.mock and args.live:
        print("[!] --mock and --live are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    mapping_store = MappingStore(args.mapping_csv)

    zc_paths = _resolve_paths(args.zc, args.zc_dir, args.zc_pattern, "zc", args.skip_zc)
    es_paths = _resolve_paths(args.es, args.es_dir, args.es_pattern, "es", args.skip_es)
    ats_paths = _resolve_paths(args.ats, args.ats_dir, args.ats_pattern, "ats", args.skip_ats)

    min_severity = Severity.INFO if args.no_filter else Severity.WARNING

    raw_events = dry_run(mapping_store, zc_paths, es_paths, ats_paths,
                          show_flagged=args.show_flagged, show_all=args.show_all,
                          min_severity=min_severity)

    if args.live or args.mock:
        try:
            run_meta = {
                "mode": "mock" if args.mock else "live",
                "provider": args.provider,
                "mapping_csv": args.mapping_csv,
                "zc_files": len(zc_paths), "es_files": len(es_paths), "ats_files": len(ats_paths),
                "events_parsed": sum(len(v) for v in raw_events.values()),
                "events_flagged": sum(len(prefilter(v, min_severity=min_severity)) for v in raw_events.values()),
            }
            live_run(mapping_store, raw_events, use_mock=args.mock,
                     min_severity=min_severity, provider=args.provider,
                     memory_path=None if args.no_memory else args.memory_path,
                     output_dir=args.output_dir,
                     excel_path=args.excel_path,
                     run_meta=run_meta)
        except ImportError as e:
            print(f"\n[!] --live/--mock requires dependencies not installed here: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
