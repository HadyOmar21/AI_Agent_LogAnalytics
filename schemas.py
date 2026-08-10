"""
Pydantic schemas for LangChain structured-output calls.

These are separate from models.py's dataclasses on purpose: models.py
defines the pipeline's INTERNAL state (typed, stdlib-only, no LLM
dependency). This file defines exactly what shape we force each LLM call
to respond in via LangChain's `.with_structured_output(...)`, which
Claude fills via native tool-calling under the hood. Keeping them
separate means the internal pipeline state never depends on
pydantic/langchain being installed -- only the agent layer does.

agents/system_agent.py and agents/correlation.py convert these into the
dataclasses from models.py right after the LLM call returns.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class FindingItem(BaseModel):
    suspected_component: str = Field(description="Process/module/object name implicated")
    machine: Optional[str] = Field(default=None)
    severity: Literal["WARNING", "ERROR", "CRITICAL"]
    summary: str = Field(description="Plain-language description of the issue")
    evidence_event_ids: list[str] = Field(description="event_ids from the input this finding is based on")
    confidence: float = Field(ge=0.0, le=1.0)


class FindingsResponse(BaseModel):
    """What each per-system (ZC/ES/ATS) agent must return."""
    findings: list[FindingItem]


class RootCauseResponse(BaseModel):
    """What the root-cause synthesis agent must return."""
    root_cause_summary: str
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    recommended_actions: list[str]
