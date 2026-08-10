"""
LangGraph orchestrator -- the "LANGGRAPH MAIN AGENT" box in the
architecture diagram.

Graph shape:

    parse_and_filter (per system, fan-out)
            |
    zc_agent   es_agent   ats_agent   (parallel, only called if that
       \\         |         /          system has WARNING+ events)
        \\        |        /
              correlate                (deterministic)
                  |
              root_cause                (LLM, per correlated incident)
                  |
                report                  (deterministic decode + format)

Requires: `pip install langgraph`. Not exercised in the build sandbox
(no network there) -- the deterministic stages (parse/filter/correlate/
report) are fully tested standalone in this repo; wire in a real
ANTHROPIC_API_KEY to exercise the agent/root_cause nodes.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from models import CorrelatedIncident, Finding, LogEvent, Report, RootCause, Severity, System
from mappings import MappingStore, encode_event
from severity_rules import classify_all, prefilter
from llm_client import LLMClient
from memory_store import MemoryStore
from agents.system_agent import analyze_system_events
from agents.correlation import correlate_findings, synthesize_root_cause
from agents.report_agent import build_report


class PipelineState(TypedDict, total=False):
    raw_events: dict[str, list[LogEvent]]      # system.value -> events (pre-filter)
    flagged_events: dict[str, list[LogEvent]]  # system.value -> events (WARNING+)
    # zc_agent / es_agent / ats_agent all write to `findings` in the SAME
    # parallel step (fan-out from "filter"). LangGraph requires an explicit
    # reducer for any state key written by more than one node in one step --
    # `operator.add` here means "concatenate the lists" instead of "last
    # write wins" (which is the default and is what caused
    # `InvalidUpdateError: Can receive only one value per step`). Each node
    # below returns ONLY its own new findings; LangGraph does the
    # concatenation automatically via this reducer.
    findings: Annotated[list[Finding], operator.add]
    incidents: list[CorrelatedIncident]
    root_causes: dict[str, RootCause]          # incident_id -> RootCause
    reports: list[Report]


def build_graph(
    mapping_store: MappingStore,
    llm_client: LLMClient,
    min_severity: Severity = Severity.WARNING,
    memory_store: "MemoryStore | None" = None,
):
    """
    min_severity: the pre-filter cost gate. Severity.WARNING (default) is
        the cost-saving behavior (~97% of raw volume dropped before any
        LLM call, per earlier measurement). Pass Severity.INFO to disable
        filtering entirely and send every parsed line to the agents --
        this is a real, deliberate option (not a bug) for cases where you
        want maximum visibility at the cost of far higher API usage.
    memory_store: optional persistent history of past incidents (see
        memory_store.py). When provided, the root-cause step looks up
        prior incidents for the same component before asking the LLM to
        reason, so it can say "this also happened 3 times last week"
        instead of treating every run as if it has no history. When
        None, root-cause synthesis behaves exactly as before (no
        historical context).
    """
    from langgraph.graph import StateGraph, END

    def node_filter(state: PipelineState) -> PipelineState:
        flagged = {}
        for sys_name, events in state["raw_events"].items():
            classified = classify_all(events)
            selected = prefilter(classified, min_severity=min_severity)
            # Encode process/machine/module/object fields to their generic
            # mapping IDs HERE, before the agents ever see the events --
            # per the "encode first, analyze, decode at the end" design.
            # encode_event() is a safe no-op for any field not present in
            # the mapping table (which covers most ES/ATS field values, and
            # ZC fields not in its small process/machine table) -- see
            # mappings.py for the encode/decode contract.
            flagged[sys_name] = [encode_event(mapping_store, e) for e in selected]
        return {"flagged_events": flagged}

    def node_zc_agent(state: PipelineState) -> PipelineState:
        events = state["flagged_events"].get(System.ZC.value, [])
        findings = analyze_system_events(System.ZC, events, llm_client)
        return {"findings": findings}

    def node_es_agent(state: PipelineState) -> PipelineState:
        events = state["flagged_events"].get(System.ES.value, [])
        findings = analyze_system_events(System.ES, events, llm_client)
        return {"findings": findings}

    def node_ats_agent(state: PipelineState) -> PipelineState:
        events = state["flagged_events"].get(System.ATS.value, [])
        findings = analyze_system_events(System.ATS, events, llm_client)
        return {"findings": findings}

    def node_correlate(state: PipelineState) -> PipelineState:
        incidents = correlate_findings(state.get("findings", []), mapping_store)
        return {"incidents": incidents}

    def node_root_cause(state: PipelineState) -> PipelineState:
        root_causes = {}
        for incident in state.get("incidents", []):
            root_causes[incident.incident_id] = synthesize_root_cause(
                incident, llm_client, memory_store=memory_store)
            # record this incident into persistent memory so FUTURE runs
            # (tomorrow, next week) can reference it -- this is what makes
            # "did this happen last week" possible across separate process runs.
            if memory_store is not None:
                memory_store.record_incident(incident, root_causes[incident.incident_id])
        return {"root_causes": root_causes}

    def node_report(state: PipelineState) -> PipelineState:
        reports = []
        for incident in state.get("incidents", []):
            rc = state["root_causes"][incident.incident_id]
            reports.append(build_report(incident, rc, mapping_store))
        return {"reports": reports}

    g = StateGraph(PipelineState)
    g.add_node("filter", node_filter)
    g.add_node("zc_agent", node_zc_agent)
    g.add_node("es_agent", node_es_agent)
    g.add_node("ats_agent", node_ats_agent)
    g.add_node("correlate", node_correlate)
    g.add_node("root_cause", node_root_cause)
    g.add_node("report", node_report)

    g.set_entry_point("filter")
    # fan-out: all three system agents run off of "filter"
    g.add_edge("filter", "zc_agent")
    g.add_edge("filter", "es_agent")
    g.add_edge("filter", "ats_agent")
    # fan-in: correlate waits for all three
    g.add_edge("zc_agent", "correlate")
    g.add_edge("es_agent", "correlate")
    g.add_edge("ats_agent", "correlate")
    g.add_edge("correlate", "root_cause")
    g.add_edge("root_cause", "report")
    g.add_edge("report", END)

    return g.compile()
