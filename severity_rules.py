"""
Deterministic severity classification + pre-filter.

This is the most important cost/latency lever in the whole pipeline: the
overwhelming majority of log lines are routine. Nothing here calls an LLM.
Only events that classify as WARNING or above get passed on to the
per-system Claude agents (see agents/system_agent.py + graph.py).

Per-system rules differ because the raw formats differ:

  ZC  -- no structured severity field at all. Classify from message text.
  ATS -- has a <CHANNEL> tag (ERRLOG, DEBUGLOG, STATMONLOG, ...) but that's
         a log-routing channel, not severity (measured: <ERRLOG> appears on
         ~78% of lines in the sample, far more than actual faults). Combine
         channel with message-body keywords.
  ES  -- no structured severity field observed in the sample available.
         Keyword-based, same approach as ZC, until a real ES fault sample
         is available to validate against.
"""

from __future__ import annotations

import re

from models import LogEvent, Severity, System

# Ordered by rank so the FIRST match wins (most severe checked first).
_CRITICAL_RE = re.compile(r"\b(CRIT|CRITICAL|FATAL)\b")
_ERROR_RE = re.compile(r"\b(ERROR|ERR|EXCEPTION|FAIL(?:ED|URE)?)\b", re.IGNORECASE)
_WARNING_RE = re.compile(r"\b(WARN(?:ING)?)\b", re.IGNORECASE)

# ATS-specific: known non-error log channels that should NOT, by
# themselves, upgrade severity even though ERRLOG might be present as a tag.
_ATS_INFO_CHANNELS = {"STATMONLOG", "DEBUGLOG", "MSGLOG", "LIFECYCLELOG",
                       "REGULATIONINFO", "TIMERS", "DATALOGLOG", "WSGATELOG"}


def _classify_by_keywords(text: str) -> Severity:
    if _CRITICAL_RE.search(text):
        return Severity.CRITICAL
    if _ERROR_RE.search(text):
        return Severity.ERROR
    if _WARNING_RE.search(text):
        return Severity.WARNING
    return Severity.INFO


def classify(event: LogEvent) -> Severity:
    """Returns the classified severity; does NOT mutate the event."""
    if event.system == System.ATS:
        text_severity = _classify_by_keywords(event.message)
        # channel tag is a weak secondary signal only: an ERRLOG-channel
        # line with no keyword match stays INFO (routing noise), but if
        # the channel itself signals something unusual (not a known
        # steady-state channel) treat as at least WARNING.
        if text_severity == Severity.INFO and event.channel and \
                event.channel not in _ATS_INFO_CHANNELS and event.channel != "ERRLOG":
            return Severity.WARNING
        return text_severity

    # ZC and ES: pure keyword classification against the message body.
    return _classify_by_keywords(event.message)


def classify_all(events: list[LogEvent]) -> list[LogEvent]:
    """Returns new LogEvent instances with `severity` populated."""
    out = []
    for e in events:
        sev = classify(e)
        out.append(_with_severity(e, sev))
    return out


def _with_severity(event: LogEvent, sev: Severity) -> LogEvent:
    # dataclasses.replace keeps this simple and avoids mutating input
    from dataclasses import replace
    return replace(event, severity=sev)


def prefilter(events: list[LogEvent], min_severity: Severity = Severity.WARNING) -> list[LogEvent]:
    """The cost-control gate: only events at/above min_severity continue
    on to LLM analysis. Call classify_all() first."""
    return [e for e in events if e.severity.rank >= min_severity.rank]
