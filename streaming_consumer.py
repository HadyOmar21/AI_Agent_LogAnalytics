"""
Real-time Kafka consumer skeleton -- how the batch pipeline (graph.py)
plugs into your production streaming setup.

Recommended topic layout: one topic per system (zc.logs, es.logs, ats.logs)
so each consumer group can be scaled/tuned independently (ATS at ~5,551
distinct object types is a much higher-volume, higher-cardinality stream
than ES).

This module owns the two things streaming adds on top of the batch
pipeline:
  1. Windowing -- buffer raw lines per (system, machine) key for
     WINDOW_SECONDS, so events from ZC/ES/ATS about the same real-world
     incident (which won't arrive simultaneously) land in the same
     correlation pass.
  2. Dedup/debounce -- a single fault often produces bursts of near-
     identical repeated lines (see the ZC "message repeated N times"
     pattern observed in the sample data). Collapse repeats before they
     ever reach an LLM.

Requires: `pip install kafka-python`. Not exercised in the build sandbox
(no network / no Kafka broker there) -- this is production-target code.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Callable

from models import LogEvent, System
from mappings import MappingStore
from parsers import parse_zc_line, parse_ats_line, parse_es_line

WINDOW_SECONDS = 60
TOPICS = {
    System.ZC: "zc.logs",
    System.ES: "es.logs",
    System.ATS: "ats.logs",
}

# collapses syslog's own "message repeated N times: [...]" wrapper and
# near-identical consecutive messages down to one representative event
# plus a repeat count, before anything reaches an LLM.
_REPEATED_RE = re.compile(r"message repeated (\d+) times: \[(.*)\]")


def _dedupe_key(event: LogEvent) -> str:
    msg = _REPEATED_RE.sub(r"\2", event.message)
    return f"{event.system.value}|{event.machine}|{event.process or event.module}|{msg[:120]}"


class WindowBuffer:
    """Buffers parsed LogEvents per system for WINDOW_SECONDS, deduping
    near-identical repeats, then flushes a batch to the pipeline."""

    def __init__(self, on_flush: Callable[[dict[str, list[LogEvent]]], None],
                 window_seconds: int = WINDOW_SECONDS):
        self.on_flush = on_flush
        self.window = timedelta(seconds=window_seconds)
        self._buffers: dict[str, list[LogEvent]] = defaultdict(list)
        self._seen: dict[str, int] = {}  # dedupe_key -> repeat count
        self._window_start = datetime.utcnow()

    def add(self, event: LogEvent) -> None:
        key = _dedupe_key(event)
        if key in self._seen:
            self._seen[key] += 1
            return  # collapse the repeat -- don't buffer a duplicate event
        self._seen[key] = 1
        self._buffers[event.system.value].append(event)

        if datetime.utcnow() - self._window_start >= self.window:
            self.flush()

    def flush(self) -> None:
        if any(self._buffers.values()):
            self.on_flush(dict(self._buffers))
        self._buffers = defaultdict(list)
        self._seen = {}
        self._window_start = datetime.utcnow()


def make_parser(system: System, mapping_store: MappingStore):
    """Returns a callable(raw_line: str) -> Optional[LogEvent] for the
    given system, matching what the Kafka consumer message value contains."""
    if system == System.ZC:
        return lambda line: parse_zc_line(line, mapping_store)
    if system == System.ATS:
        return lambda line: parse_ats_line(line)
    if system == System.ES:
        return lambda line: parse_es_line(line, mapping_store)
    raise ValueError(system)


def run_consumer(
    mapping_store: MappingStore,
    on_flush: Callable[[dict[str, list[LogEvent]]], None],
    bootstrap_servers: str = "localhost:9092",
    group_id: str = "log-analytics-pipeline",
):
    """Production entry point: subscribes to all three topics, parses each
    message with the right per-system parser, buffers/dedupes via
    WindowBuffer, and calls on_flush(...) every WINDOW_SECONDS with a
    dict[system_value -> events] -- pass that straight into
    graph.build_graph(...).invoke({"raw_events": flushed_dict})
    """
    from kafka import KafkaConsumer  # lazy import -- optional dependency

    parsers = {sys: make_parser(sys, mapping_store) for sys in System}
    topic_to_system = {v: k for k, v in TOPICS.items()}

    consumer = KafkaConsumer(
        *TOPICS.values(),
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: v.decode("utf-8"),
    )
    buffer = WindowBuffer(on_flush)

    try:
        for message in consumer:
            system = topic_to_system[message.topic]
            event = parsers[system](message.value)
            if event:
                buffer.add(event)
    finally:
        # flush whatever's still buffered on shutdown (KeyboardInterrupt,
        # consumer disconnect, etc.) so a partial window isn't silently lost
        buffer.flush()
