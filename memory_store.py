"""
Persistent, cross-run incident memory -- lets the root-cause agent answer
"has this happened before?" (last week, last month) instead of treating
every single run as if it has no history.

Design choice: this is a plain JSON file on disk, NOT a LangChain
Memory/Agent construct. Per the earlier discussion: we want the LLM call
itself to stay a single, reliable, structured request (same guarantees as
before) -- "memory" here means enriching that ONE prompt with historical
facts we looked up ourselves beforehand, not handing the model an
open-ended conversational memory object and letting it decide what to do
with it. This keeps the reliability properties of the existing design
while still giving genuine accumulative/historical analysis.

Storage: one JSON file, a list of past incident records. For real
production volume this should become a proper database (SQLite at
minimum) -- a flat JSON file is fine for daily-batch-scale usage (one
file per day/week of incidents, not per log line) but will get slow if
you let it grow to tens of thousands of records. See the note in
`_load` for the natural upgrade path.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import CorrelatedIncident, RootCause

DEFAULT_MEMORY_PATH = "memory/incident_history.json"


class MemoryStore:
    def __init__(self, path: str | Path = DEFAULT_MEMORY_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            # Corrupt/unreadable file -- don't crash the pipeline over
            # history, just start fresh. Upgrade to SQLite if this file
            # keeps growing large enough to matter.
            return []

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2, default=str)

    def record_incident(self, incident: "CorrelatedIncident", root_cause: "RootCause") -> None:
        """Called once per incident, right after root-cause synthesis, so
        future runs can see it. Stores just enough to answer "did this
        component have problems before" -- not the full evidence (that
        stays in the report, not in long-term memory)."""
        record = {
            "incident_id": incident.incident_id,
            "systems_involved": [s.value for s in incident.systems_involved],
            "generic_ids_involved": incident.generic_ids_involved,
            "max_severity": incident.max_severity.value,
            "window_start": incident.window_start.isoformat(),
            "window_end": incident.window_end.isoformat(),
            "root_cause_summary": root_cause.root_cause_summary,
            "recorded_at": datetime.utcnow().isoformat(),
        }
        self._records.append(record)
        self._save()

    def history_for_components(
        self, generic_ids: list[str], lookback_days: int = 14, exclude_incident_id: str | None = None
    ) -> list[dict]:
        """Returns past incident records that share at least one generic
        ID with the given list, within the lookback window. This is what
        answers "has this component had issues before" -- called by
        agents/correlation.py before asking the LLM to reason, so the
        historical facts go INTO the prompt as context the LLM is told
        about, not something it has to remember on its own."""
        if not generic_ids:
            return []
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        id_set = set(generic_ids)
        matches = []
        for r in self._records:
            if exclude_incident_id and r["incident_id"] == exclude_incident_id:
                continue
            if not id_set.intersection(r.get("generic_ids_involved", [])):
                continue
            try:
                recorded_at = datetime.fromisoformat(r["recorded_at"])
            except (KeyError, ValueError):
                continue
            if recorded_at >= cutoff:
                matches.append(r)
        return sorted(matches, key=lambda r: r["recorded_at"], reverse=True)

    def summarize_history_for_prompt(self, generic_ids: list[str], lookback_days: int = 14) -> str:
        """Renders history_for_components() as a short text block ready
        to drop into an LLM prompt. Returns "" (nothing) if there's no
        history -- so callers can just conditionally include this without
        extra branching."""
        history = self.history_for_components(generic_ids, lookback_days=lookback_days)
        if not history:
            return ""
        lines = [f"HISTORICAL CONTEXT: this component (or a related one) had "
                 f"{len(history)} prior incident(s) in the last {lookback_days} days:"]
        for r in history[:5]:  # cap what we inject -- this is context, not the whole log
            lines.append(f"  - {r['recorded_at'][:10]}: [{r['max_severity']}] "
                         f"{r['root_cause_summary'][:200]}")
        if len(history) > 5:
            lines.append(f"  ... and {len(history) - 5} more not shown.")
        return "\n".join(lines)
