"""
Parser for ATS trace logs, e.g.:

    27/07/2026 04:29:37.266 03610 MSF-VS-ARCHIVE[01]<ERRLOG> 00.01.00E2 \
        ConfigIntegerManager::Retrieve:90 Could not resolve:#Enum::...

    27/07/2026 04:29:34.421 00766 MSF-VS-ARCHIVE[01] 00.01.00E2 \
        PrintGroups:146  Group LogFile (2337): 0 us - 3 object(s) : 0 us/object

Fields: date time.ms seq process[instance]<channel?> version Module::Func:line message

Important: the <CHANNEL> tag (e.g. <ERRLOG>) is a LOG ROUTING CHANNEL, not
a severity level by itself -- in the sample file ~78% of lines carry
<ERRLOG> yet only a small fraction contain an actual fault (FAIL/CRIT/
EXCEPTION keywords, or "Could not resolve" style messages). Severity
classification (severity_rules.py) treats the channel as one signal among
several, not as ground truth.

`module` here is "Class::Function" (e.g. "ConfigIntegerManager::Retrieve").
The class portion (e.g. "ConfigIntegerManager") is what lines up against
the ATS `object` mapping table, so it is also surfaced as `object_ref`.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

from models import LogEvent, System, Severity

_LINE_RE = re.compile(
    r"^(?P<date>\d{2}/\d{2}/\d{4})\s+(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+"
    r"(?P<seq>\d+)\s+"
    r"(?P<process>[^\[\s]+)\[(?P<inst>\d+)\](?:<(?P<channel>[A-Z]+)>)?\s+"
    r"(?P<version>\S+)\s+"
    r"(?P<module>[A-Za-z_][A-Za-z0-9_:]*?):(?P<lineno>\d+)\s+"
    r"(?P<message>.*)$"
)


def _parse_timestamp(date_str: str, time_str: str) -> Optional[datetime]:
    try:
        dt_part = datetime.strptime(date_str, "%d/%m/%Y")
        h, m, rest = time_str.split(":")
        s, ms = rest.split(".")
        return dt_part.replace(hour=int(h), minute=int(m), second=int(s),
                                microsecond=int(ms) * 1000)
    except (ValueError, IndexError):
        return None


def parse_ats_line(
    raw: str,
    source_file: Optional[str] = None,
    line_no: Optional[int] = None,
) -> Optional[LogEvent]:
    raw = raw.rstrip("\r\n")
    if not raw.strip():
        return None
    m = _LINE_RE.match(raw)
    if not m:
        return None

    ts = _parse_timestamp(m.group("date"), m.group("time"))
    if ts is None:
        return None

    module_full = m.group("module")           # "Class::Function"
    module_class = module_full.split("::")[0] if module_full else None

    return LogEvent(
        event_id=f"ats_{uuid.uuid4().hex[:10]}",
        system=System.ATS,
        timestamp=ts,
        machine=m.group("process"),            # e.g. "MSF-VS-ARCHIVE" (+ instance)
        process=f"{m.group('process')}[{m.group('inst')}]",
        module=module_full,
        object_ref=module_class,
        severity=Severity.UNKNOWN,             # filled in by severity_rules.classify()
        channel=m.group("channel"),
        message=m.group("message"),
        raw=raw,
        source_file=source_file,
        line_no=line_no,
    )


def parse_ats_file(path: str | Path) -> list[LogEvent]:
    path = Path(path)
    events: list[LogEvent] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            evt = parse_ats_line(line, source_file=str(path), line_no=i)
            if evt:
                events.append(evt)
    return events
