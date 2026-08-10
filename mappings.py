"""
Mapping engine built from all_ids_mapping.csv.

Design decisions (see architecture discussion):
  - Encoding/decoding is 100% deterministic dict lookups. No LLM is ever
    used for this — it's cheap, exact, and an LLM would risk hallucinating
    a wrong name for an ID at ATS's scale (~5,551 object mappings).
  - The full mapping table is NEVER stuffed into an LLM prompt. Agents
    either (a) receive already-resolved original names, or (b) call the
    `lookup_id` / `lookup_name` tool functions on demand for the specific
    IDs they're reasoning about.
  - ZC logs are observed to already contain a MIX of raw names and
    pre-substituted numeric IDs at the source (e.g. "systemd" sometimes
    appears as "1001"). `resolve_zc_process` below normalizes both cases
    to the canonical original name before the rest of the pipeline ever
    sees it, so downstream code only ever deals with one representation
    at a time (original name in LogEvent, generic id at correlation time).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class MappingKey:
    system: str          # ZC / ES / ATS
    mapping_type: str    # process / machine / object


class MappingStore:
    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        # (system, mapping_type, original_name) -> generic_id
        self._name_to_id: dict[tuple[str, str, str], str] = {}
        # (system, mapping_type, generic_id) -> original_name
        self._id_to_name: dict[tuple[str, str, str], str] = {}
        self._load()

    def _load(self) -> None:
        with open(self.csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                system = row["system"].strip()
                mtype = row["mapping_type"].strip()
                name = row["original_name"].strip()
                gid = row["generic_id"].strip()
                self._name_to_id[(system, mtype, name)] = gid
                self._id_to_name[(system, mtype, gid)] = name

    # -------- encode (name -> id) --------
    def encode(self, system: str, mapping_type: str, original_name: Optional[str]) -> Optional[str]:
        """Return the generic id for a name, or the name itself (unchanged)
        if it's not present in the mapping table (e.g. ZC processes like
        'nautilus' that were never in the anonymization set)."""
        if original_name is None:
            return None
        return self._name_to_id.get((system, mapping_type, original_name), original_name)

    # -------- decode (id -> name) --------
    def decode(self, system: str, mapping_type: str, generic_id: Optional[str]) -> Optional[str]:
        """Return the original name for a generic id, or the id itself
        (unchanged) if it's not a recognized mapped id."""
        if generic_id is None:
            return None
        return self._id_to_name.get((system, mapping_type, generic_id), generic_id)

    # -------- ZC-specific: raw log token may ALREADY be a generic id --------
    def resolve_zc_process(self, token: str) -> str:
        """Given a raw ZC syslog process-tag token (which may be a real
        process name OR an already-substituted numeric id, see module
        docstring), return the canonical ORIGINAL name.
        """
        if token.isdigit():
            return self.decode("ZC", "process", token)
        return token

    def known_ids(self, system: str, mapping_type: str) -> set[str]:
        return {gid for (s, t, gid) in self._id_to_name if s == system and t == mapping_type}

    def known_names(self, system: str, mapping_type: str) -> set[str]:
        return {n for (s, t, n) in self._name_to_id if s == system and t == mapping_type}


# ---------------------------------------------------------------------------
# OPTIONAL: tool-call schemas for giving an agent on-demand ID lookup
# instead of a giant prompt-stuffed table. NOT wired into the default
# agents/system_agent.py or agents/correlation.py flow -- those use
# LangChain's `.with_structured_output()`, which forces a single final
# schema and doesn't mix cleanly with intermediate tool calls. If you need
# on-demand lookups for a specific agent later, bind these as LangChain
# tools instead (`from langchain_core.tools import StructuredTool`,
# `chat.bind_tools([...])`) and run a manual tool-calling loop for that
# one node rather than `.with_structured_output()`.
# ---------------------------------------------------------------------------
LOOKUP_ID_TOOL_SCHEMA = {
    "name": "lookup_id",
    "description": (
        "Resolve a generic mapping ID back to its original name for a given "
        "system and mapping type (process, machine, or object)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "system": {"type": "string", "enum": ["ZC", "ES", "ATS"]},
            "mapping_type": {"type": "string", "enum": ["process", "machine", "object"]},
            "generic_id": {"type": "string"},
        },
        "required": ["system", "mapping_type", "generic_id"],
    },
}

LOOKUP_NAME_TOOL_SCHEMA = {
    "name": "lookup_name",
    "description": (
        "Resolve an original component/process/machine name to its generic "
        "mapping ID for a given system and mapping type."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "system": {"type": "string", "enum": ["ZC", "ES", "ATS"]},
            "mapping_type": {"type": "string", "enum": ["process", "machine", "object"]},
            "original_name": {"type": "string"},
        },
        "required": ["system", "mapping_type", "original_name"],
    },
}


def make_tool_handlers(store: MappingStore) -> dict:
    """Returns a dict of {tool_name: callable} to dispatch Claude tool_use
    blocks against, for agents that need on-demand ID resolution."""
    def _lookup_id(system: str, mapping_type: str, generic_id: str) -> dict:
        return {"original_name": store.decode(system, mapping_type, generic_id)}

    def _lookup_name(system: str, mapping_type: str, original_name: str) -> dict:
        return {"generic_id": store.encode(system, mapping_type, original_name)}

    return {"lookup_id": _lookup_id, "lookup_name": _lookup_name}

# ---------------------------------------------------------------------------
# encode_event: encodes a LogEvent's process/machine/module/object fields to
# their generic mapping IDs, applied BEFORE the per-system agents see the
# event (see graph.py: node_filter). This is a deliberate change from the
# earlier design (which kept original names for agent-facing text and only
# encoded at the correlation join-key step) -- per explicit instruction, the
# full "encode first, analyze, decode at the end" flow is now what's used.
#
# Trade-off worth knowing: encoding before the agent sees the data means it
# loses some semantic signal for any field that DOES have a mapping entry
# (e.g. ATS "ConfigIntegerManager" becomes "217" in the prompt) -- the model
# is still told the system/severity/message, just not the human-readable
# component name for matched fields. Unmapped names (the majority, based on
# real sample data -- most ES/ATS values aren't in the mapping table at all)
# pass through unchanged, so this mostly affects the minority of fields that
# ARE in the mapping table.
# ---------------------------------------------------------------------------
def encode_event(store: "MappingStore", event):
    """Returns a new LogEvent (dataclasses.replace) with process/machine/
    module/object_ref each encoded to their generic ID where a mapping
    exists, left unchanged otherwise. `event.system` determines which
    system's mapping table is used."""
    from dataclasses import replace

    sys_name = event.system.value if hasattr(event.system, "value") else event.system

    new_process = store.encode(sys_name, "process", event.process) if event.process else event.process
    new_machine = store.encode(sys_name, "machine", event.machine) if event.machine else event.machine
    # module/object_ref both draw from the "object" mapping type
    new_module = store.encode(sys_name, "object", event.module) if event.module else event.module
    new_object_ref = store.encode(sys_name, "object", event.object_ref) if event.object_ref else event.object_ref

    return replace(
        event,
        process=new_process,
        machine=new_machine,
        module=new_module,
        object_ref=new_object_ref,
    )
