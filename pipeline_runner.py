"""
Folder-based pipeline runner -- the shared engine behind both the CLI
(main.py) and the Streamlit chat interface (app.py).

This is the "the agent communicates with FOLDERS, not files" entry point:
you hand it three directories (one per subsystem -- ZC / ES / ATS), it globs
every matching file out of each, parses and combines them into ONE run,
runs the full LangGraph pipeline (when live/mock), and writes the output
reports to a single accumulating Excel workbook (excel_writer.py) -- every
run appends to the SAME .xlsx, it is never overwritten.

Modes (mirroring main.py, which keeps all its existing options):
  - "dry"  : parse + classify + pre-filter only, NO LLM calls. Returns what
             WOULD be analyzed. Free, deterministic.
  - "live" : real LLM calls via the chosen provider (anthropic/ollama/glm).
  - "mock" : free, no-API-key rule-based fake LLM -- proves wiring end-to-end.

Everything here is a thin orchestration layer; the deterministic core
(parsers, severity_rules, correlation, decode) and the LLM layer
(llm_client/graph) are unchanged from the rest of the repo.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from mappings import MappingStore
from models import Severity, System, Report
from parsers import parse_zc_csv, parse_ats_file, parse_es_file
from severity_rules import classify_all, prefilter


ProgressFn = Optional[Callable[[str, str], None]]


@dataclass
class RunResult:
    """Everything one run produced, for the CLI/Streamlit to render."""
    mode: str
    provider: str
    mapping_csv: str
    zc_files: list[str] = field(default_factory=list)
    es_files: list[str] = field(default_factory=list)
    ats_files: list[str] = field(default_factory=list)
    raw_events: dict = field(default_factory=dict)            # system -> events (classified)
    flagged_by_system: dict = field(default_factory=dict)     # system -> flagged events
    events_parsed: int = 0
    events_flagged: int = 0
    reports: list = field(default_factory=list)              # list[Report]
    excel_path: Optional[str] = None
    excel_run_id: Optional[str] = None
    excel_rows_before: int = 0
    md_paths: list[str] = field(default_factory=list)
    memory_path: Optional[str] = None
    errors: list[str] = field(default_factory=list)

    @property
    def reports_count(self) -> int:
        return len(self.reports)


def _collect_files(directory: str, pattern: str) -> list[str]:
    """Collect every file matching `pattern` from `directory`, sorted for stable
    run-to-run ordering. Returns [] if the directory is missing/empty -- the
    caller decides whether an empty subsystem is an error (skip) or not.

    `directory` may be either a FOLDER (every file matching the glob pattern is
    collected) or a single FILE (returned as-is, ignoring the pattern). This
    lets a user point a field at one specific log file instead of a folder --
    handy when there's only one big CSV/log to analyze."""
    p = Path(directory)
    if p.is_file():
        return [str(p)]
    if not p.is_dir():
        raise NotADirectoryError(
            f"Not a directory (and not a file): {directory}")
    return sorted(str(f) for f in p.glob(pattern) if f.is_file())


def _parse_all(paths: list[str], parse_fn, mapping_store) -> list:
    """Parse every file in `paths` and concatenate -- N files in, one
    combined event list out (same as if it were all one file). This is what
    makes folder/batch mode work."""
    events = []
    for path in paths:
        # parse_es_file/parse_zc_csv need the mapping store; parse_ats_file doesn't.
        try:
            sig_arity = parse_fn.__code__.co_argcount
        except AttributeError:
            sig_arity = 2
        if sig_arity >= 2 and mapping_store is not None:
            events.extend(parse_fn(path, mapping_store))
        else:
            events.extend(parse_fn(path))
    return events


def run_pipeline(
    zc_dir: Optional[str] = None,
    es_dir: Optional[str] = None,
    ats_dir: Optional[str] = None,
    *,
    zc_pattern: str = "*.csv",
    es_pattern: str = "*.log",
    ats_pattern: str = "*.log",
    mapping_csv: str = "sample_data/all_ids_mapping.csv",
    mode: str = "dry",                 # "dry" | "live" | "mock"
    provider: str = "anthropic",       # "anthropic"|"claude"|"ollama"|"glm"
    no_filter: bool = False,
    use_memory: bool = True,
    memory_path: str = "memory/incident_history.json",
    excel_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    progress: ProgressFn = None,
) -> RunResult:
    """Run the full pipeline against three subsystem log FOLDERS.

    See module docstring for modes/providers. Raises NotADirectoryError if a
    given folder isn't a directory; an empty folder simply means that
    subsystem contributes zero events (not fatal -- you can run on just the
    subsystems you have logs for, same as main.py's --skip-* flags).

    `progress(stage, message)` is an optional callback so the Streamlit UI
    can render live stage updates without this module knowing about Streamlit.
    """
    def _emit(stage: str, msg: str):
        if progress:
            progress(stage, msg)

    result = RunResult(mode=mode, provider=provider, mapping_csv=mapping_csv)
    min_severity = Severity.INFO if no_filter else Severity.WARNING

    _emit("setup", f"Loading mapping store from {mapping_csv}")
    mapping_store = MappingStore(mapping_csv)

    # ---- collect + parse folders ----
    if zc_dir:
        result.zc_files = _collect_files(zc_dir, zc_pattern)
        _emit("parse", f"ZC: {len(result.zc_files)} file(s) in {zc_dir}")
    if es_dir:
        result.es_files = _collect_files(es_dir, es_pattern)
        _emit("parse", f"ES: {len(result.es_files)} file(s) in {es_dir}")
    if ats_dir:
        result.ats_files = _collect_files(ats_dir, ats_pattern)
        _emit("parse", f"ATS: {len(result.ats_files)} file(s) in {ats_dir}")

    raw_events: dict[str, list] = {}
    if result.zc_files:
        raw_events[System.ZC.value] = classify_all(
            _parse_all(result.zc_files, parse_zc_csv, mapping_store))
    if result.es_files:
        raw_events[System.ES.value] = classify_all(
            _parse_all(result.es_files, parse_es_file, mapping_store))
    if result.ats_files:
        raw_events[System.ATS.value] = classify_all(
            _parse_all(result.ats_files, parse_ats_file, mapping_store))

    if not raw_events:
        raise ValueError(
            "No log files found in any of the given folders. Point zc_dir/"
            "es_dir/ats_dir at folders that contain matching files (or relax "
            "the glob pattern).")

    # ---- pre-filter (the cost gate) ----
    flagged_by_system = {}
    for sys_name, events in raw_events.items():
        flagged = prefilter(events, min_severity=min_severity)
        flagged_by_system[sys_name] = flagged
        result.events_parsed += len(events)
        result.events_flagged += len(flagged)
        pct = (len(flagged) / len(events) * 100) if events else 0
        _emit("filter", f"{sys_name}: {len(events)} parsed -> {len(flagged)} flagged ({pct:.1f}%)")

    result.raw_events = raw_events
    result.flagged_by_system = flagged_by_system

    # ---- dry run stops here ----
    if mode == "dry":
        _emit("done", f"Dry run complete: {result.events_parsed} parsed, "
                      f"{result.events_flagged} would reach an LLM. No reports generated.")
        return result

    # ---- live / mock: run the graph ----
    _emit("llm", f"Initializing {'mock' if mode == 'mock' else provider} client")
    if mode == "mock":
        from mock_llm_client import MockLLMClient
        client = MockLLMClient()
    else:
        from llm_client import LLMClient
        client = LLMClient(provider=provider)

    memory_store = None
    if use_memory:
        from memory_store import MemoryStore
        memory_store = MemoryStore(memory_path)
        result.memory_path = str(memory_store.path)
        _emit("memory", f"Loaded memory: {len(memory_store._records)} past incident(s)")

    from graph import build_graph
    _emit("graph", "Building LangGraph pipeline")
    graph = build_graph(mapping_store, client, min_severity=min_severity,
                        memory_store=memory_store)

    _emit("analyze", f"Running pipeline over {result.events_flagged} flagged events...")
    try:
        graph_result = graph.invoke({"raw_events": raw_events})
    except Exception as e:
        result.errors.append(f"Pipeline invocation failed: {e}")
        _emit("error", f"Pipeline failed: {e}")
        return result

    reports = graph_result.get("reports", [])
    result.reports = reports
    _emit("analyze", f"Generated {len(reports)} report(s)")

    # ---- write Excel (append to the same workbook) ----
    if excel_path:
        from excel_writer import append_run, report_count
        result.excel_path = str(Path(excel_path))
        result.excel_rows_before = report_count(excel_path)
        run_meta = {
            "mode": mode,
            "provider": provider,
            "mapping_csv": mapping_csv,
            "zc_files": len(result.zc_files),
            "es_files": len(result.es_files),
            "ats_files": len(result.ats_files),
            "events_parsed": result.events_parsed,
            "events_flagged": result.events_flagged,
        }
        run_id = append_run(excel_path, reports, run_meta)
        result.excel_run_id = run_id
        _emit("excel", f"Appended {len(reports)} report(s) to {excel_path} (run {run_id})")

    # ---- optional per-report .md files (kept for parity with main.py) ----
    if output_dir and reports:
        from agents.report_agent import render_markdown
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(reports, start=1):
            md = render_markdown(r)
            out_path = Path(output_dir) / f"report_{i:02d}_{r.incident_id}.md"
            out_path.write_text(md, encoding="utf-8")
            result.md_paths.append(str(out_path))
        _emit("markdown", f"Wrote {len(reports)} .md file(s) to {output_dir}")

    _emit("done", f"Run complete: {len(reports)} report(s) generated.")
    return result