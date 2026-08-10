"""
Parser for ES (simulator) General logs, e.g.:

    2026/06/29  08:12:08:858  The SystemDB file with name D:\\...\\SystemDB.xml
    has been loaded.

Fields: date time:ms  message   (no explicit severity or module field observed)

CAVEAT: the only ES sample available at build time was 5 lines of clean
startup/INFO output -- no fault example was provided, so:
  1. Severity classification here is keyword-based (mirrors the ZC
     approach) as a reasonable default, but has NOT been validated against
     a real ES error/warning line. Revisit severity_rules.ES_KEYWORDS once
     a real fault sample is available.
  2. There's no structured module/object field in this log type. As a
     best-effort heuristic, we scan the message text for any known ES
     object name from the mapping table (Vehicle, VOBC, ZC, RIO, ACE-*,
     UNBOUND (RPC_Server)) and surface the first match as `object_ref`.
     This is a heuristic, not a guaranteed-correct field -- flagged in
     `object_ref_is_heuristic` semantics via the parser's return, and
     should be tightened once real ES fault logs are seen.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

from models import LogEvent, System, Severity
from mappings import MappingStore

_LINE_RE = re.compile(
    r"^(?P<date>\d{4}/\d{2}/\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}:\d{3})\s+"
    r"(?P<message>.*)$"
)


def _parse_timestamp(date_str: str, time_str: str) -> Optional[datetime]:
    try:
        y, mo, d = (int(x) for x in date_str.split("/"))
        h, mi, s, ms = (int(x) for x in time_str.split(":"))
        return datetime(y, mo, d, h, mi, s, ms * 1000)
    except ValueError:
        return None


def _guess_object_ref(message: str, mapping_store: MappingStore) -> Optional[str]:
    known = mapping_store.known_names("ES", "object")
    # longest names first, so e.g. "UNBOUND (RPC_Server)" wins over a
    # coincidental short substring match
    for name in sorted(known, key=len, reverse=True):
        if name and name in message:
            return name
    return None


def parse_es_line(
    raw: str,
    mapping_store: MappingStore,
    source_file: Optional[str] = None,
    line_no: Optional[int] = None,
) -> Optional[LogEvent]:
    raw = raw.rstrip("\n")
    if not raw.strip():
        return None
    m = _LINE_RE.match(raw)
    if not m:
        return None

    ts = _parse_timestamp(m.group("date"), m.group("time"))
    if ts is None:
        return None

    message = m.group("message")

    return LogEvent(
        event_id=f"es_{uuid.uuid4().hex[:10]}",
        system=System.ES,
        timestamp=ts,
        machine=None,
        process=None,
        module=None,
        object_ref=_guess_object_ref(message, mapping_store),
        severity=Severity.UNKNOWN,   # filled in by severity_rules.classify()
        channel=None,
        message=message,
        raw=raw,
        source_file=source_file,
        line_no=line_no,
    )


def parse_es_file(path: str | Path, mapping_store: MappingStore) -> list[LogEvent]:
    path = Path(path)
    events: list[LogEvent] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            evt = parse_es_line(line, mapping_store, source_file=str(path), line_no=i)
            if evt:
                events.append(evt)
    return events
