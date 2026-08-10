"""
Correlation engine (deterministic) + root-cause synthesis (LLM).

Architecture decision: correlating findings across ZC/ES/ATS by "does this
look related" is exactly the kind of task where a pure-LLM approach
hallucinates plausible-but-wrong causal links, especially at volume. So:

  1. correlate_findings() is 100% deterministic: it encodes each finding's
     suspected_component to its generic mapping ID (the whole point of
     having those IDs -- they're a reliable join key across systems that
     use completely different naming conventions) and clusters findings
     that share an ID AND fall within a time window.

  2. synthesize_root_cause() then hands the LLM a SMALL, CONDENSED cluster
     (not the raw log firehose) to reason over and produce a narrative +
     recommendations. This is where the LLM adds real value: turning a
     structured incident graph into an engineer-readable explanation.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from models import CorrelatedIncident, Finding, RootCause, Severity
from mappings import MappingStore
from llm_client import LLMClient
from schemas import RootCauseResponse

DEFAULT_CORRELATION_WINDOW = timedelta(seconds=60)


def _encode_component(store: MappingStore, system: str, name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    # try each mapping_type in turn -- component could be a process,
    # machine, or object depending on which system produced the finding
    for mtype in ("process", "object", "machine"):
        gid = store.encode(system, mtype, name)
        if gid != name:  # found a real mapping
            return gid
    return name  # not in the mapping table -- keep original as the join key


def correlate_findings(
    findings: list[Finding],
    mapping_store: MappingStore,
    window: timedelta = DEFAULT_CORRELATION_WINDOW,
) -> list[CorrelatedIncident]:
    """Clusters findings that (a) share an encoded component ID and
    (b) fall within `window` of each other. Findings that don't cluster
    with anything else become single-finding incidents (still worth
    reporting -- not every real fault is multi-system)."""
    if not findings:
        return []

    # attach encoded id to each finding for clustering
    enriched = []
    for f in findings:
        gid = _encode_component(mapping_store, f.system.value, f.suspected_component)
        enriched.append((gid, f))

    enriched.sort(key=lambda pair: pair[1].window_start or pair[1].window_end or _epoch())

    clusters: list[list[tuple[Optional[str], Finding]]] = []
    for gid, finding in enriched:
        placed = False
        for cluster in clusters:
            if _fits_cluster(cluster, gid, finding, window):
                cluster.append((gid, finding))
                placed = True
                break
        if not placed:
            clusters.append([(gid, finding)])

    incidents = []
    for cluster in clusters:
        c_findings = [f for _, f in cluster]
        gids = sorted({gid for gid, _ in cluster if gid})
        starts = [f.window_start for f in c_findings if f.window_start]
        ends = [f.window_end for f in c_findings if f.window_end]
        incidents.append(CorrelatedIncident(
            incident_id=CorrelatedIncident.new_id(),
            findings=c_findings,
            systems_involved=sorted({f.system for f in c_findings}, key=lambda s: s.value),
            generic_ids_involved=gids,
            window_start=min(starts) if starts else _epoch(),
            window_end=max(ends) if ends else _epoch(),
            max_severity=max((f.severity for f in c_findings), key=lambda s: s.rank),
        ))
    return incidents


def _fits_cluster(cluster, gid, finding, window: timedelta) -> bool:
    ids_in_cluster = {g for g, _ in cluster if g}
    if gid and gid in ids_in_cluster:
        # same component id -- check time proximity too
        for g2, f2 in cluster:
            if g2 == gid and _within_window(f2, finding, window):
                return True
    return False


def _within_window(a: Finding, b: Finding, window: timedelta) -> bool:
    a_t = a.window_end or a.window_start
    b_t = b.window_start or b.window_end
    if a_t is None or b_t is None:
        return False
    return abs((b_t - a_t)) <= window


def _epoch():
    from datetime import datetime
    return datetime(1970, 1, 1)


# ---------------------------------------------------------------------------
# LLM root-cause synthesis
# ---------------------------------------------------------------------------

# --- Subtask-structured prompt design ------------------------------------
# The root-cause prompt is broken into ordered subtasks instead of one
# dense paragraph, so the model's reasoning path is explicit: understand
# the cluster -> form hypotheses -> pick the root cause -> rate confidence
# -> give actionable recommendations -> honor the output contract. The
# structured-output contract (schemas.RootCauseResponse) is unchanged, so
# everything downstream (report, Excel, memory) behaves identically.
ROOT_CAUSE_SYSTEM_PROMPT = """\
You are a senior rail-signalling reliability engineer performing root cause
analysis. You will be given a correlated cluster of findings from one or
more subsystems (ZC/ES/ATS), already grouped by a shared component ID and
time window.

Work through these subtasks IN ORDER:

### Subtask 1 -- Understand the cluster
Read the cluster as a whole: which subsystems are involved, which shared
component IDs tie them together, the time window, and the max severity.
Form a mental picture of the incident before reasoning about cause.

### Subtask 2 -- Form candidate hypotheses
List the plausible root causes that could explain ALL of the findings in
the cluster together (not just one finding in isolation). Prefer a single
root cause that accounts for the cross-system pattern over a separate
explanation per finding.

### Subtask 3 -- Pick the most likely root cause
Choose the best-supported hypothesis. Write `root_cause_summary` as a
plain-language explanation an on-call engineer can act on. Synthesize --
do not restate the findings verbatim.

### Subtask 4 -- Explain your reasoning
In `reasoning`, walk through why this root cause fits the evidence and why
alternatives were rejected. Be concise but concrete; reference the
subsystems/components involved.

### Subtask 5 -- Rate confidence
Set `confidence` (0.0-1.0) honestly. High only when the root cause is
clearly supported across findings; lower when the cluster is ambiguous or
evidence is thin.

### Subtask 6 -- Recommend actions
Give `recommended_actions`: a short list of concrete, executable steps an
on-call engineer could take right now (inspect, restart, capture, escalate).
Each item must be an action, not a restatement of the problem.

### Output contract
Return a single RootCauseResponse. If historical incident context is
provided, take it into account: a component that keeps failing repeatedly
is a pattern, not an isolated event -- say so explicitly. Do not invent
components, ids, or evidence not present in the cluster.
"""

def _format_incident_for_prompt(incident: CorrelatedIncident) -> str:
    lines = [
        f"Incident window: {incident.window_start} - {incident.window_end}",
        f"Systems involved: {[s.value for s in incident.systems_involved]}",
        f"Shared component IDs: {incident.generic_ids_involved}",
        f"Max severity: {incident.max_severity.value}",
        "Findings:",
    ]
    for f in incident.findings:
        lines.append(
            f"  - [{f.system.value}] severity={f.severity.value} "
            f"component={f.suspected_component} machine={f.machine} "
            f"confidence={f.confidence}: {f.summary} "
            f"(evidence: {f.evidence_event_ids})"
        )
    return "\n".join(lines)


def synthesize_root_cause(
    incident: CorrelatedIncident,
    client: LLMClient,
    memory_store=None,  # Optional[MemoryStore]; typed loosely to avoid a hard import here
) -> RootCause:
    prompt_parts = [
        "Synthesize a root cause analysis for the following correlated incident.\n\n",
        _format_incident_for_prompt(incident),
    ]

    if memory_store is not None:
        history_text = memory_store.summarize_history_for_prompt(incident.generic_ids_involved)
        if history_text:
            prompt_parts.append(
                "\n\n" + history_text +
                "\n\nTake this history into account: if the same component keeps failing "
                "repeatedly, say so explicitly and treat it as a pattern, not an isolated event."
            )

    prompt = "".join(prompt_parts)
    result: RootCauseResponse = client.call_structured(
        system_prompt=ROOT_CAUSE_SYSTEM_PROMPT,
        user_content=prompt,
        output_schema=RootCauseResponse,
    )
    return RootCause(
        incident_id=incident.incident_id,
        root_cause_summary=result.root_cause_summary,
        affected_components_encoded=incident.generic_ids_involved,
        confidence=result.confidence,
        reasoning=result.reasoning,
        recommended_actions=result.recommended_actions,
    )
