"""
Core data models for the log-analytics agent pipeline.

Uses plain dataclasses (not pydantic) so the deterministic parts of this
pipeline (parsing, mapping, severity rules, correlation) have zero external
dependencies and can be run/tested with nothing but the standard library.

If you want request/response schema validation against the Claude API
(structured tool-use outputs), see agents/system_agent.py — that layer
defines its own lightweight JSON-schema dicts for tool calls, and you are
free to swap these dataclasses for pydantic BaseModels later without
touching the parsers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class System(str, Enum):
    ZC = "ZC"
    ES = "ES"
    ATS = "ATS"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

    @property
    def rank(self) -> int:
        return {
            Severity.INFO: 0,
            Severity.UNKNOWN: 0,
            Severity.WARNING: 1,
            Severity.ERROR: 2,
            Severity.CRITICAL: 3,
        }[self]


@dataclass
class LogEvent:
    """A single parsed log line, normalized to a common schema.

    `process` / `module` / `machine` are kept as their ORIGINAL names at
    this stage (not yet encoded) because the per-system LLM agents reason
    better with semantic names than with opaque numeric IDs. Encoding to
    generic IDs happens later, only for the correlation join key
    (see mappings.py: encode_event).
    """
    event_id: str
    system: System
    timestamp: datetime
    machine: Optional[str]          # e.g. "man1" (ZC), host/unit for ES/ATS
    process: Optional[str]          # e.g. "dbus-daemon", "systemd" (ZC)
    module: Optional[str]           # e.g. "ConfigIntegerManager" (ATS Module::Function)
    object_ref: Optional[str]       # e.g. ATS "ATC::VOBC::StopCmd" style object, ES object
    severity: Severity
    channel: Optional[str]          # e.g. ATS "<ERRLOG>" log channel tag, if present
    message: str
    raw: str                        # original raw line, kept for audit / report evidence
    source_file: Optional[str] = None
    line_no: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["system"] = self.system.value
        d["severity"] = self.severity.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class Finding:
    """Structured output of a per-system analysis agent (ZC/ES/ATS agent)."""
    finding_id: str
    system: System
    severity: Severity
    suspected_component: Optional[str]   # process/module/object implicated
    machine: Optional[str]
    summary: str                         # LLM's plain-language interpretation
    evidence_event_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5              # 0..1, agent's self-reported confidence
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None

    @staticmethod
    def new_id() -> str:
        return f"find_{uuid.uuid4().hex[:10]}"


@dataclass
class CorrelatedIncident:
    """Output of the deterministic correlation engine: a cluster of findings
    from one or more systems believed to relate to the same underlying event."""
    incident_id: str
    findings: list[Finding]
    systems_involved: list[System]
    generic_ids_involved: list[str]      # encoded IDs shared across findings (join keys)
    window_start: datetime
    window_end: datetime
    max_severity: Severity

    @staticmethod
    def new_id() -> str:
        return f"inc_{uuid.uuid4().hex[:10]}"


@dataclass
class RootCause:
    """Output of the root-cause LLM agent, reasoning over a CorrelatedIncident."""
    incident_id: str
    root_cause_summary: str
    affected_components_encoded: list[str]   # still-encoded IDs at this stage
    confidence: float
    reasoning: str
    recommended_actions: list[str] = field(default_factory=list)


@dataclass
class Report:
    """Final, decoded, human-readable report for engineers."""
    incident_id: str
    title: str
    severity: Severity
    systems_involved: list[str]
    affected_components: list[str]           # DECODED original names
    time_window: str
    root_cause_summary: str
    recommended_actions: list[str]
    evidence: list[str]                      # decoded raw/annotated log lines
    confidence: float
