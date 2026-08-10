"""
Real-time entry point. This is what you actually run in production --
main.py is for batch/demo testing against files, this is the live path.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python stream_main.py --bootstrap-servers localhost:9092

What it does, every time a window closes (default every 60s, see
streaming_consumer.WINDOW_SECONDS):
  1. Takes the buffered, deduped events for that window (per system)
  2. Runs them through the full pipeline (filter -> agents -> correlate ->
     root cause -> report) via graph.py
  3. Prints each generated report to stdout as JSON, one per line

Swap `emit_report` below for whatever you actually want to happen with a
finished report -- post to Slack, open a ticket, write to a database,
push to a dashboard, etc. That's the one integration point you need to
customize; everything upstream of it is already wired.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from datetime import datetime

from mappings import MappingStore
from graph import build_graph
from llm_client import LLMClient
from streaming_consumer import run_consumer, TOPICS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stream_main")


def _json_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if hasattr(o, "value"):  # Enum
        return o.value
    return str(o)


def emit_report(report) -> None:
    """Customize this: this is where a finished Report goes out the door.
    Default behavior: log it as JSON to stdout."""
    print(json.dumps(dataclasses.asdict(report), default=_json_default))


def main():
    parser = argparse.ArgumentParser(description="Real-time log analytics pipeline (Kafka -> Claude -> report)")
    parser.add_argument("--bootstrap-servers", default="localhost:9092",
                         help="Kafka bootstrap servers, e.g. localhost:9092 or broker1:9092,broker2:9092")
    parser.add_argument("--group-id", default="log-analytics-pipeline")
    parser.add_argument("--mapping-csv", default="sample_data/all_ids_mapping.csv",
                         help="Path to all_ids_mapping.csv (use your production copy, not the sample)")
    args = parser.parse_args()

    log.info("Loading mapping store from %s", args.mapping_csv)
    mapping_store = MappingStore(args.mapping_csv)

    log.info("Initializing Claude client")
    llm_client = LLMClient()

    log.info("Building LangGraph pipeline")
    pipeline = build_graph(mapping_store, llm_client)

    def on_flush(raw_events: dict) -> None:
        counts = {k: len(v) for k, v in raw_events.items()}
        log.info("Window closed, processing batch: %s", counts)
        try:
            result = pipeline.invoke({"raw_events": raw_events})
        except Exception:
            log.exception("Pipeline run failed for this window -- skipping, waiting for next window")
            return

        reports = result.get("reports", [])
        log.info("Generated %d report(s)", len(reports))
        for report in reports:
            emit_report(report)

    log.info("Subscribing to topics: %s", list(TOPICS.values()))
    log.info("Starting consumer (Ctrl+C to stop)...")
    try:
        run_consumer(
            mapping_store,
            on_flush,
            bootstrap_servers=args.bootstrap_servers,
            group_id=args.group_id,
        )
    except KeyboardInterrupt:
        log.info("Shutting down.")
        sys.exit(0)


if __name__ == "__main__":
    main()
