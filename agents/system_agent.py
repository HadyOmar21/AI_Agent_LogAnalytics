"""
Per-system analysis agent (the "ZC Agent" / "ES Agent" / "ATS Agent" boxes
in the architecture diagram).

Each agent only ever sees events from ONE system, already pre-filtered to
WARNING+ by severity_rules.prefilter() -- it is never handed the raw
firehose. It reasons using ORIGINAL component names (not generic IDs) so
it has real semantic signal to work with (see architecture discussion:
encoding happens later, only for the correlation join key).

Output is forced through LangChain's structured output (schemas.py:
FindingsResponse) -- never free text -- so graph.py can consume it without
re-parsing prose. The pydantic response is converted to models.Finding
dataclasses immediately below, so nothing downstream of this file needs
to know LangChain/pydantic exist.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from models import Finding, LogEvent, Severity, System
from llm_client import LLMClient
from schemas import FindingsResponse

# --- Context-window safety ------------------------------------------------
# A per-system agent is handed every WARNING+ event for ONE subsystem. On a
# big log folder that can be thousands of lines, and formatting them all into
# a single prompt exceeds the model's max context length (HTTP 400 "prompt too
# long"). To stay safely under the window we split the batch by a CHARACTER
# budget before any LLM call: each sub-batch is analyzed independently and the
# resulting findings are concatenated. The deterministic correlation engine
# downstream re-clusters findings by encoded component ID + time window, so
# splitting here does not lose cross-batch signal.
#
# Sized conservatively (~4 chars/token => ~15-20K tokens) to fit the smallest
# supported context window (GLM-4.6 / glm-5.2:cloud) after room is left for the
# system prompt and the structured output. Override via the env var if your
# model has a notably smaller window, or raise it to cut API calls on a large
# window. A non-positive value disables chunking (one call -- the old behavior,
# useful only for small batches).
MAX_EVENTS_CHARS_PER_BATCH = int(
    os.environ.get("MAX_EVENTS_CHARS_PER_BATCH", "60000")
)

# --- Subtask-structured prompt design ------------------------------------
# Each per-system prompt is built from ONE shared subtask skeleton plus a
# short, system-specific role/cues block. The skeleton (below) breaks the
# analysis into ordered subtasks instead of one dense paragraph, which makes
# the agent's job explicit and repeatable: group -> identify component ->
# severity -> summarize -> evidence -> confidence -> output contract.
#
# This changes ONLY how the model is instructed; the structured-output
# contract (schemas.FindingsResponse) is unchanged, so everything downstream
# (graph.py, correlation, report, Excel, memory) behaves identically.

_SUBTASK_SKELETON = """\
Work through these subtasks IN ORDER:

### Subtask 1 -- Group related events
Scan the batch and group events that describe the same underlying issue.
Near-identical repeated lines, the same process/module/object across
consecutive events, or a shared error signature all belong together. Each
group becomes ONE finding. Do not emit one finding per line.

### Subtask 2 -- Identify the suspected component
For each group, name the single most-implicated process / module / object
in its ORIGINAL (un-encoded) form, exactly as it appears in the input.
Set `machine` to the host the events occurred on, if known.

### Subtask 3 -- Assign severity
For each group, set `severity` to the HIGHEST severity observed anywhere in
that group: "WARNING" < "ERROR" < "CRITICAL". Never invent a severity that
does not appear in the group's events.

### Subtask 4 -- Summarize the issue
Write a one-to-two-sentence, plain-language summary of what is actually
going wrong for that group. Explain the failure, not the log line. Do not
restate raw log text verbatim.

### Subtask 5 -- Cite evidence
For each group, list the `event_id`s (the bracketed tokens) of the input
events that support it in `evidence_event_ids`. Only cite ids that appear
in the input you were given -- never invent ids.

### Subtask 6 -- Rate confidence
Set `confidence` (0.0-1.0) honestly: high only when the failure is clear
and well-evidenced; lower when the signal is ambiguous or noisy.

### Output contract
Return a single FindingsResponse with one FindingItem per group. Do not
return free text, commentary, or ids you were not given.
"""

_SYSTEM_SPECIFICS = {
    System.ZC: (
        "You are a Linux/syslog reliability analyst for the ZC (zone controller "
        "host) subsystem of a rail signalling system. The batch is pre-filtered "
        "syslog (already WARNING or worse). The suspected component is usually a "
        "process name (e.g. libvirtd, nautilus, gnome-shell) and the machine is "
        "the host it ran on."
    ),
    System.ES: (
        "You are a reliability analyst for the ES (train/track simulator) "
        "subsystem of a rail signalling system. The batch is pre-filtered "
        "simulator log lines. The suspected component is usually a simulated "
        "object/component named in the message."
    ),
    System.ATS: (
        "You are a reliability analyst for the ATS (automatic train supervision) "
        "subsystem of a rail signalling system. The batch is pre-filtered "
        "trace-log lines, each tagged with a Module::Function. The suspected "
        "component is usually the Module (class) or named object in the trace."
    ),
}


def _build_system_prompt(system: System) -> str:
    """Compose the role + system-specific cues with the shared subtask
    skeleton. Keeping the skeleton shared means adding/removing a subsystem
    only needs the short specifics block above (see README)."""
    return f"{_SYSTEM_SPECIFICS[system]}\n\n{_SUBTASK_SKELETON}"


SYSTEM_PROMPTS = {system: _build_system_prompt(system) for system in System}


def _format_event(e: LogEvent) -> str:
    """One event rendered as a single prompt line. Kept as its own helper so
    the chunker can size batches with the exact same text the model will see
    (no off-by-one between sizing and sending)."""
    return (
        f"[{e.event_id}] {e.timestamp.isoformat()} sev={e.severity.value} "
        f"machine={e.machine} process={e.process} module={e.module} "
        f"object={e.object_ref} channel={e.channel} :: {e.message}"
    )


def _format_events_for_prompt(events: list[LogEvent]) -> str:
    return "\n".join(_format_event(e) for e in events)


def _chunk_events(events: list[LogEvent]) -> list[list[LogEvent]]:
    """Split `events` into sub-batches whose formatted text stays under
    MAX_EVENTS_CHARS_PER_BATCH, so a large flagged set never exceeds the
    model's context window in a single call.

    Greedy fill: keep appending events until the next one would cross the
    budget, then start a new batch. A single event larger than the budget
    gets its own batch (it is never dropped -- better to try and let the
    provider reject that one oversized line than to silently lose data).
    Returns [events] unchanged when chunking is disabled (budget <= 0) or
    the batch already fits.
    """
    budget = MAX_EVENTS_CHARS_PER_BATCH
    if budget <= 0:
        return [events]

    batches: list[list[LogEvent]] = []
    current: list[LogEvent] = []
    current_chars = 0
    for e in events:
        size = len(_format_event(e)) + 1  # +1 for the joining newline
        if current and current_chars + size > budget:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(e)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def analyze_system_events(
    system: System,
    events: list[LogEvent],
    client: LLMClient,
    window_start: Optional[datetime] = None,
    window_end: Optional[datetime] = None,
) -> list[Finding]:
    """Runs the per-system agent over a pre-filtered batch of events for
    ONE system and returns structured Findings. Returns [] if events is
    empty (no API call made -- cost control).

    If window_start/window_end aren't explicitly provided, they're derived
    from the min/max timestamp across the input events. This matters: the
    correlation engine (agents/correlation.py) only clusters findings that
    fall within a shared time window, so every Finding needs a real
    timestamp range or cross-system correlation silently never fires.

    Big batches are split into context-safe sub-batches (see
    _chunk_events) and each is analyzed in its own LLM call; the findings
    are concatenated. The window assigned to every finding is the GLOBAL
    min/max across all input events (not the per-batch range), which
    preserves the original single-call semantics: all findings for a
    subsystem share that subsystem's full time range, so cross-system
    correlation behaves exactly as before regardless of how many batches
    the events were split into.
    """
    if not events:
        return []

    if window_start is None:
        window_start = min(e.timestamp for e in events)
    if window_end is None:
        window_end = max(e.timestamp for e in events)

    batches = _chunk_events(events)
    findings: list[Finding] = []
    for batch in batches:
        prompt = (
            f"Analyze the following {len(batch)} pre-filtered {system.value} log events "
            f"and return the distinct findings you identify.\n\n"
            f"{_format_events_for_prompt(batch)}"
        )

        result: FindingsResponse = client.call_structured(
            system_prompt=SYSTEM_PROMPTS[system],
            user_content=prompt,
            output_schema=FindingsResponse,
        )

        for f in result.findings:
            findings.append(Finding(
                finding_id=Finding.new_id(),
                system=system,
                severity=Severity(f.severity),
                suspected_component=f.suspected_component,
                machine=f.machine,
                summary=f.summary,
                evidence_event_ids=f.evidence_event_ids,
                confidence=f.confidence,
                window_start=window_start,
                window_end=window_end,
            ))
    return findings
