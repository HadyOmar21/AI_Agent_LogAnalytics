"""
Parser for ZC logs (syslog / RFC3164-ish format), delivered as a
single-column quoted CSV, e.g.:

    "Aug  5 02:32:17 man1 1003:  [origin software=""1003""...] 1003 was HUPed
    "
    "Aug  5 03:46:19 man1 gnome-shell[2215]: JS ERROR: TypeError: ...
    "

Observed quirk: some process-tag tokens are already-substituted generic
IDs from the ZC mapping table (e.g. "1001" standing in for "systemd"),
while others are the real process name (e.g. "nautilus", "gnome-shell",
"anacron", "snapd") because those processes were never in the
anonymization/mapping set. `resolve_zc_process` in mappings.py normalizes
both cases back to the original name.

No structured severity field exists in ZC logs — severity must be
inferred from message content (see severity_rules.py).
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

from models import LogEvent, System, Severity
from mappings import MappingStore

_LINE_RE = re.compile(
    r"^(?P<mon>[A-Za-z]{3})\s+(?P<day>\d{1,2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<machine>\S+)\s+"
    r"(?P<process>[^:\[\s]+)"
    r"(\[(?P<pid>\d+)\])?"
    r":\s?(?P<message>.*)$"
)

_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _parse_timestamp(mon: str, day: str, time_str: str, year: int) -> Optional[datetime]:
    if mon not in _MONTHS:
        return None
    h, m, s = (int(x) for x in time_str.split(":"))
    try:
        return datetime(year, _MONTHS[mon], int(day), h, m, s)
    except ValueError:
        return None


def parse_zc_line(
    raw: str,
    mapping_store: MappingStore,
    year: Optional[int] = None,
    source_file: Optional[str] = None,
    line_no: Optional[int] = None,
) -> Optional[LogEvent]:
    raw = raw.rstrip("\n")
    if not raw.strip():
        return None
    m = _LINE_RE.match(raw)
    if not m:
        return None

    year = year or datetime.utcnow().year
    ts = _parse_timestamp(m.group("mon"), m.group("day"), m.group("time"), year)
    if ts is None:
        return None

    process_token = m.group("process")
    process_name = mapping_store.resolve_zc_process(process_token)
    message = m.group("message")

    return LogEvent(
        event_id=f"zc_{uuid.uuid4().hex[:10]}",
        system=System.ZC,
        timestamp=ts,
        machine=m.group("machine"),
        process=process_name,
        module=None,
        object_ref=None,
        severity=Severity.UNKNOWN,   # filled in by severity_rules.classify()
        channel=None,
        message=message,
        raw=raw,
        source_file=source_file,
        line_no=line_no,
    )


def parse_zc_csv(
    path: str | Path,
    mapping_store: MappingStore,
    year: Optional[int] = None,
) -> list[LogEvent]:
    """Reads the single-column quoted CSV export and parses each row."""
    path = Path(path)
    events: list[LogEvent] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip "log" header
        for i, row in enumerate(reader, start=2):
            if not row:
                continue
            raw_line = row[0]
            # multi-line quoted cells can contain an embedded newline;
            # only take the first physical line as the log content
            first_line = raw_line.split("\n", 1)[0]
            evt = parse_zc_line(
                first_line, mapping_store, year=year,
                source_file=str(path), line_no=i,
            )
            if evt:
                events.append(evt)
    return events
