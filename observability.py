"""
Observability wiring: LangSmith (LLM-call tracing) + OpenTelemetry
(pipeline/infra spans), joined by a shared trace/correlation id so you can
pivot between "what did the LLM reason about" (LangSmith) and "how long
did each stage take, where did it run" (OTel).

Since llm_client.py routes every Claude call through LangChain's
ChatAnthropic (not the raw Anthropic SDK), LangSmith tracing is fully
automatic once the env vars below are set -- no manual span creation
needed for the LLM calls themselves. You only need `pipeline_span` for
the surrounding deterministic stages (parse, filter, correlate, decode)
that LangSmith doesn't see.

Requires: `pip install langsmith opentelemetry-sdk opentelemetry-exporter-otlp`

LangSmith: set these env vars and tracing is automatic for every
LangChain/LangGraph call (including every Claude call the agents make):
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=...
    LANGCHAIN_PROJECT=log-analytics-pipeline

OpenTelemetry: use `pipeline_span` below to wrap each deterministic stage
(parse, filter, correlate, report) so infra-level timing/errors show up in
your OTel backend (Jaeger, Honeycomb, Datadog, etc.) alongside the
LangSmith LLM traces.
"""

from __future__ import annotations

import contextlib
import os
import uuid


def configure_langsmith(project: str = "log-analytics-pipeline") -> None:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    # LANGCHAIN_API_KEY must still be set by the caller/deployment env.


def new_correlation_id() -> str:
    """One id per pipeline run, propagated into both LangSmith run
    metadata and OTel span attributes so the two systems can be joined."""
    return f"run_{uuid.uuid4().hex}"


def get_tracer(service_name: str = "log-analytics-pipeline"):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


@contextlib.contextmanager
def pipeline_span(tracer, name: str, correlation_id: str, **attrs):
    """Wrap a pipeline stage in an OTel span tagged with the shared
    correlation_id (also pass this same id into LangGraph's
    `config={"metadata": {"correlation_id": ...}}` for LangSmith joinability).
    """
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("correlation_id", correlation_id)
        for k, v in attrs.items():
            span.set_attribute(k, v)
        yield span
