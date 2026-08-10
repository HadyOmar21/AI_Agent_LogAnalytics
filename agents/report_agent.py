"""
Report builder -- decodes generic IDs back to original names and produces
the final engineer-facing Report.

Decoding is a plain dictionary lookup (mappings.py: MappingStore.decode),
not an LLM call -- see architecture rationale in mappings.py's docstring.
The only LLM involvement at this stage, if you want it, is optional
prose polishing of the already-decoded, already-correct content -- never
letting the model choose the ID->name mapping itself.
"""

from __future__ import annotations

from models import CorrelatedIncident, Report, RootCause
from mappings import MappingStore


def _decode_component(store: MappingStore, system: str, generic_id: str) -> str:
    for mtype in ("process", "object", "machine"):
        name = store.decode(system, mtype, generic_id)
        if name != generic_id:
            return name
    return generic_id


def _decode_machine(store: MappingStore, system: str, machine: str | None) -> str | None:
    if machine is None:
        return None
    decoded = store.decode(system, "machine", machine)
    return decoded  # decode() already no-ops (returns input) if not found


def build_report(
    incident: CorrelatedIncident,
    root_cause: RootCause,
    mapping_store: MappingStore,
) -> Report:
    systems = [s.value for s in incident.systems_involved]

    # decode each generic id using the system(s) involved as candidates
    affected_components = []
    for gid in root_cause.affected_components_encoded:
        decoded = gid
        for sys_name in systems:
            candidate = _decode_component(mapping_store, sys_name, gid)
            if candidate != gid:
                decoded = candidate
                break
        affected_components.append(decoded)

    evidence = []
    for f in incident.findings:
        # Since events are now encoded BEFORE reaching the agents (see
        # graph.py: node_filter), a Finding's suspected_component/machine
        # may themselves be generic IDs (whatever the agent echoed back
        # from what it was given) -- decode both for the evidence line,
        # same as affected_components above. This was the specific gap
        # flagged earlier: machine was never decoded anywhere.
        decoded_component = f.suspected_component
        decoded_machine = f.machine
        for sys_name in ([f.system.value] + [s for s in systems if s != f.system.value]):
            c = _decode_component(mapping_store, sys_name, f.suspected_component or "")
            if c != f.suspected_component:
                decoded_component = c
            m = _decode_machine(mapping_store, sys_name, f.machine)
            if m != f.machine:
                decoded_machine = m
            if decoded_component != f.suspected_component and decoded_machine != f.machine:
                break

        evidence.append(
            f"[{f.system.value}/{f.severity.value}] {decoded_component} "
            f"(machine={decoded_machine}): {f.summary} -- events: {f.evidence_event_ids}"
        )

    title_component = affected_components[0] if affected_components else "unknown component"
    title = f"[{incident.max_severity.value}] {'/'.join(systems)} incident involving {title_component}"

    return Report(
        incident_id=incident.incident_id,
        title=title,
        severity=incident.max_severity,
        systems_involved=systems,
        affected_components=affected_components,
        time_window=f"{incident.window_start.isoformat()} - {incident.window_end.isoformat()}",
        root_cause_summary=root_cause.root_cause_summary,
        recommended_actions=root_cause.recommended_actions,
        evidence=evidence,
        confidence=root_cause.confidence,
    )


def render_markdown(report: Report) -> str:
    lines = [
        f"# {report.title}",
        "",
        f"**Severity:** {report.severity.value}  ",
        f"**Systems involved:** {', '.join(report.systems_involved)}  ",
        f"**Time window:** {report.time_window}  ",
        f"**Confidence:** {report.confidence:.0%}",
        "",
        "## Root Cause",
        report.root_cause_summary,
        "",
        "## Affected Components",
        *[f"- {c}" for c in report.affected_components],
        "",
        "## Recommended Actions",
        *[f"- [ ] {a}" for a in report.recommended_actions],
        "",
        "## Evidence",
        *[f"- {e}" for e in report.evidence],
    ]
    return "\n".join(lines)
