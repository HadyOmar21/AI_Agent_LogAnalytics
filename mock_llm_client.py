"""
Mock LLM client -- runs the ENTIRE pipeline (parse -> filter -> "agents" ->
correlate -> "root cause" -> report) with ZERO API calls and ZERO cost.

This is NOT a substitute for real analysis. It uses simple rule-based
grouping (same process/module + adjacent in the batch = one finding) --
not reasoning. Every finding/root-cause it produces is clearly labeled
"[MOCK]" so it's never mistaken for real Claude output.

What it IS useful for:
  - Proving your new real log files parse and flow through every stage
    correctly (correlation clustering, ID encode/decode, report
    rendering) before spending any API budget.
  - Catching pipeline bugs (crashes, bad field mappings, correlation
    logic errors) for free.
  - CI/smoke tests that shouldn't depend on a live API key or incur cost
    on every run.

What it is NOT useful for:
  - Judging whether the actual analysis is any good. Real severity
    grouping, root-cause reasoning, and recommended actions only come
    from LLMClient (llm_client.py) with a real ANTHROPIC_API_KEY.

Usage:
    python main.py --zc mylogs/zc.csv --skip-es --skip-ats --mock
"""

from __future__ import annotations

from collections import defaultdict

from schemas import FindingsResponse, FindingItem, RootCauseResponse


class MockLLMClient:
    """Drop-in replacement for llm_client.LLMClient -- same
    call_structured(system_prompt, user_content, output_schema) interface,
    used identically by agents/system_agent.py and agents/correlation.py.
    No network access, no API key, no cost.
    """

    def __init__(self, *args, **kwargs):
        pass  # accepts and ignores model/api_key/temperature for interface compatibility

    def call_structured(self, system_prompt: str, user_content: str, output_schema):
        if output_schema is FindingsResponse:
            return self._mock_findings(user_content)
        if output_schema is RootCauseResponse:
            return self._mock_root_cause(user_content)
        raise ValueError(f"MockLLMClient has no handler for schema {output_schema}")

    def call_text(self, system_prompt: str, user_content: str) -> str:
        """Free-text chat for the Streamlit chat stage in mock mode. No real
        reasoning -- just echoes the question back with a clear [MOCK] tag so
        it is never mistaken for a real analysis answer."""
        return (
            "[MOCK] I'm a rule-based stand-in, not a real model, so I can't "
            "actually reason about your reports. Your question was:\n\n"
            f"{user_content[:500]}\n\n"
            "Re-run the analysis with a live provider (claude/ollama/glm) to "
            "get real answers in this chat."
        )

    # -- FindingsResponse: group consecutive lines from the same process/module --
    def _mock_findings(self, prompt_text: str) -> FindingsResponse:
        groups: dict[str, dict] = {}
        order: list[str] = []

        for line in prompt_text.splitlines():
            line = line.strip()
            if not line.startswith("["):
                continue
            event_id = line[1:line.index("]")]
            component = self._extract_field(line, "module") or \
                        self._extract_field(line, "object") or \
                        self._extract_field(line, "process") or "unknown"
            machine = self._extract_field(line, "machine")
            severity = "CRITICAL" if " sev=CRITICAL " in line else \
                       "ERROR" if " sev=ERROR " in line else "WARNING"

            key = component
            if key not in groups:
                groups[key] = {
                    "component": component, "machine": machine,
                    "severity": severity, "event_ids": [],
                }
                order.append(key)
            g = groups[key]
            g["event_ids"].append(event_id)
            if severity == "CRITICAL" or (severity == "ERROR" and g["severity"] == "WARNING"):
                g["severity"] = severity

        findings = []
        for key in order:
            g = groups[key]
            findings.append(FindingItem(
                suspected_component=g["component"],
                machine=g["machine"],
                severity=g["severity"],
                summary=(f"[MOCK] {len(g['event_ids'])} {g['severity']} line(s) grouped "
                         f"by matching component '{g['component']}' -- this is rule-based "
                         f"grouping, not real analysis. Re-run with --live for actual reasoning."),
                evidence_event_ids=g["event_ids"],
                confidence=0.3,  # deliberately low -- signals "not a real judgment"
            ))
        return FindingsResponse(findings=findings)

    def _mock_root_cause(self, prompt_text: str) -> RootCauseResponse:
        components = set()
        for line in prompt_text.splitlines():
            if "component=" in line:
                components.add(self._extract_field(line, "component"))
        components.discard(None)
        comp_str = ", ".join(sorted(components)) if components else "the affected component(s)"

        return RootCauseResponse(
            root_cause_summary=(
                f"[MOCK] This is a placeholder, not real root-cause analysis. The "
                f"deterministic correlation engine grouped findings involving {comp_str} "
                f"within the configured time window. Re-run with --live and a real "
                f"ANTHROPIC_API_KEY to get Claude's actual reasoning about what's wrong "
                f"and why."
            ),
            reasoning="[MOCK] No real reasoning was performed -- rule-based grouping only.",
            confidence=0.2,
            recommended_actions=[
                "[MOCK] Re-run this same command with --live instead of --mock to get real "
                "recommended actions from Claude.",
            ],
        )

    @staticmethod
    def _extract_field(line: str, field: str) -> str | None:
        marker = f"{field}="
        idx = line.find(marker)
        if idx == -1:
            return None
        start = idx + len(marker)
        end = line.find(" ", start)
        val = line[start:end] if end != -1 else line[start:]
        return None if val in ("None", "") else val
