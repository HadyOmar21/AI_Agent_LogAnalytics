# Log Analytics Multi-Agent Pipeline (ZC / ES / ATS)

Real-time log analytics pipeline for a rail signalling system's ZC (zone
controller), ES (simulator), and ATS (automatic train supervision) log
streams: encode → analyze (per-system LLM agents) → correlate → root
cause → decode → engineer report.

## Architecture

```
Kafka topics (zc.logs / es.logs / ats.logs)
        |
   per-system parser  (parsers/*.py)        -- deterministic, stdlib only
        |
   severity classify + pre-filter (severity_rules.py)  -- deterministic
        |            (only ~2-3% of volume passes, see below)
        v
   +------------------------------------------------+
   |            LangGraph orchestrator (graph.py)     |
   |                                                  |
   |   ZC agent    ES agent    ATS agent   (Claude,   |
   |      \\           |           /         parallel,  |
   |       \\          |          /          per-system)|
   |             correlate                (deterministic|
   |                |                       ID-based    |
   |            root_cause                 clustering)  |
   |                |                      (Claude, over |
   |              report                   condensed     |
   |                                        incident graph)|
   +------------------------------------------------+
        |
   decode (mappings.py)  -- deterministic, never an LLM
        |
   Markdown report for engineers
        |
   LangSmith (LLM traces) + OpenTelemetry (pipeline spans), joined by
   a shared correlation_id (observability.py)
```

## Key design decisions (why it's built this way)

1. **Encode/decode is never an LLM call.** Both directions are plain
   dictionary lookups against `all_ids_mapping.csv` (`mappings.py`). With
   ATS alone having 5,551 object mappings, letting an LLM "remember" the
   right ID↔name pairing risks silent hallucination. Determinism here is
   free and exact — spend LLM budget on interpretation, not lookup.

2. **Per-system agents reason on original names, not IDs.** Semantic
   names (`dbus-daemon`, `ConfigIntegerManager`) give the model real
   signal; encoding to generic IDs happens only at the correlation layer,
   where IDs are useful specifically *because* they're a consistent join
   key across ZC/ES/ATS's very different naming conventions.

3. **The mapping table is never stuffed into a prompt.** Agents that need
   an on-demand ID lookup call `lookup_id` / `lookup_name` as a tool
   (`mappings.py: LOOKUP_ID_TOOL_SCHEMA`, wired through `llm_client.py`)
   instead of receiving the whole table as context.

4. **A deterministic pre-filter gates everything before any LLM call**
   (`severity_rules.py`). Measured against the real sample logs, this
   drops the LLM-bound volume to ~2.3% of parsed lines — this is the
   single biggest cost/latency lever in the system, and it's why the
   pipeline is affordable to run continuously against a real-time stream.

5. **Correlation is deterministic (ID + time window), not LLM-guessed.**
   `agents/correlation.py: correlate_findings()` clusters findings that
   share an encoded component ID within a configurable time window
   (default 60s). This avoids the LLM inventing plausible-but-wrong causal
   links across systems, especially at volume. The LLM is only asked to
   *synthesize a narrative* over an already-condensed, already-structured
   incident cluster (`synthesize_root_cause()`) — that's where it adds
   real value.

6. **Every agent boundary is a structured, schema-validated response**,
   never free text. `schemas.py` defines `FindingsResponse` and
   `RootCauseResponse` as pydantic models; `llm_client.py` forces every
   Claude call through LangChain's `ChatAnthropic.with_structured_output()`
   against one of those schemas. LangGraph state stays typed
   (`models.py` dataclasses) end to end — the pydantic schemas are only
   used at the LLM call boundary and converted to dataclasses immediately
   after, so nothing downstream needs to know LangChain/pydantic exist.

7. **LangChain, specifically, for the analysis layer.** `llm_client.py`
   routes every Claude call through `langchain_anthropic.ChatAnthropic`
   rather than the raw Anthropic SDK. Two things this buys you:
   - `with_structured_output(schema)` gets you validated Pydantic objects
     back directly (raises if Claude's response doesn't match the
     schema), instead of hand-parsing tool_use blocks.
   - LangSmith tracing becomes automatic (see `observability.py`) — no
     manual instrumentation of the LLM calls themselves.
   LangGraph (`graph.py`) is part of the same ecosystem, so the
   orchestration and analysis layers now sit on one consistent stack.

8. **Every LLM prompt is divided into ordered subtasks, not one dense
   paragraph.** The per-system agent prompts
   (`agents/system_agent.py: _SUBTASK_SKELETON`), the root-cause synthesis
   prompt (`agents/correlation.py: ROOT_CAUSE_SYSTEM_PROMPT`), and the
   Streamlit chat prompt (`app.py: CHAT_SYSTEM_PROMPT`) are each broken
   into an explicit sequence: *understand → group/identify → reason →
   decide → rate confidence → output contract*. This makes the model's
   job repeatable and keeps it from skipping a step (e.g. citing an
   `event_id` it was never given). Crucially the subtask skeleton is
   shared across all three per-system agents, so adding a subsystem only
   needs a short system-specific role/cues block — the analysis steps stay
   identical (see "Adding or removing a subsystem" below). The
   structured-output contracts (`FindingsResponse`, `RootCauseResponse`)
   are unchanged, so nothing downstream of the prompts behaves any
   differently.

## Prompt design: subtask-structured instructions

All three LLM-facing prompts in this project (per-system analysis,
root-cause synthesis, and the Streamlit follow-up chat) are organized as
numbered **subtasks** the model works through in order, followed by an
explicit **output contract**. For example, each per-system agent prompt
is:

1. **Group** related events into one finding per underlying issue.
2. **Identify** the suspected component (original name) and machine.
3. **Assign** severity (highest observed in the group).
4. **Summarize** the failure in plain language.
5. **Cite** only the `event_id`s present in the input.
6. **Rate** confidence honestly.
7. **Return** one `FindingsResponse` (the schema in `schemas.py`).

The same shape applies to root cause (understand → hypotheses → pick →
reason → confidence → recommend) and chat (ground → classify → answer →
stay honest). Editing a prompt means editing a numbered subtask, not
un-packing a paragraph — and because the subtask skeleton is shared,
swapping/adding a subsystem leaves the analysis steps untouched.

## What was learned from your actual sample logs (and encoded into the parsers)

- **ZC** (`parsers/zc_parser.py`): syslog format, delivered as a single-
  column quoted CSV. No structured severity field — classified from
  message keywords. Notably, the raw ZC logs are **already partially
  pre-encoded at the source**: some process-tag tokens are real names
  (`nautilus`, `gnome-shell`), others are already-substituted generic IDs
  (`1001` standing in for `systemd`) because those particular processes
  were in the original anonymization set and others weren't.
  `mappings.py: resolve_zc_process()` normalizes both cases back to the
  canonical name before anything else touches it.

- **ATS** (`parsers/ats_parser.py`): `date time.ms seq process[inst]<CHANNEL>
  version Module::Function:line message`. The `<CHANNEL>` tag (e.g.
  `<ERRLOG>`) is a **log-routing channel, not a severity level** — it
  appeared on ~78% of lines in the sample file, far more than actual
  faults (measured ~205 `FAIL` occurrences). Severity is classified from
  message-body keywords first, with the channel as a weak secondary
  signal only.

- **ES** (`parsers/es_parser.py`): `date time:ms message`, no structured
  severity or module field. Only a clean startup/INFO sample was
  available at build time — severity classification here is a reasonable
  keyword-based default but **has not been validated against a real ES
  fault line**. Revisit `severity_rules.py` (ES branch) once you have one.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
# for LangSmith tracing (optional but recommended):
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=...
```

## Running (testing stage -- files on disk, no Kafka needed)

```bash
# Deterministic dry run against sample_data/ -- no API calls, shows parse/filter volume:
python main.py

# Full pipeline with real Claude calls (needs ANTHROPIC_API_KEY):
python main.py --live

# Point at your own log exports instead of the bundled samples:
python main.py --zc mylogs/zc_export.csv --es mylogs/es_general.log \
                --ats mylogs/ats_trace.log --mapping-csv mylogs/all_ids_mapping.csv --live

# Only have logs for one or two systems? Skip the rest:
python main.py --skip-es --live
```

This is the entry point for the testing stage — validate parsing, severity
classification, and the LLM agents' analysis quality against real files
before standing up Kafka at all. `stream_main.py` (below) is the separate
production entry point for once you're moving to continuous real-time
processing.

## Running in real-time (production stage)

`stream_main.py` is the actual production entry point — `main.py` is only
for batch/demo testing against files on disk.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python stream_main.py --bootstrap-servers your-kafka-broker:9092 \
                       --mapping-csv /path/to/your/all_ids_mapping.csv
```

It:
- Subscribes to `zc.logs` / `es.logs` / `ats.logs` (one topic per system,
  so each can be scaled independently — ATS is a much higher-cardinality
  stream than ES).
- Parses each message with the matching per-system parser
  (`streaming_consumer.py`).
- Buffers/dedupes for `WINDOW_SECONDS` (default 60s) via `WindowBuffer`,
  collapsing repeat bursts (the ZC "message repeated N times" pattern
  observed in your sample, and near-identical consecutive lines) *before*
  they'd otherwise inflate LLM calls — and flushes whatever's still
  buffered on shutdown so a partial window isn't silently lost.
- On window close, runs the buffered batch through the full LangGraph
  pipeline and prints each generated report to stdout as JSON.

The one thing you'll want to customize is `emit_report()` in
`stream_main.py` — right now it just prints JSON; swap it for posting to
Slack, opening a ticket, writing to a database, or whatever your team
actually uses.

You still need something upstream of this pushing your raw ZC/ES/ATS log
lines onto the `zc.logs` / `es.logs` / `ats.logs` Kafka topics as they're
written (e.g. Filebeat, Fluent Bit, or a small tailer process) — this
project starts at "a message is on the topic," not at "a file is being
written to disk."

## File map

| File | Role | Dependencies |
|---|---|---|
| `models.py` | Typed dataclasses for every stage (LogEvent → Finding → CorrelatedIncident → RootCause → Report) | stdlib only |
| `mappings.py` | Deterministic encode/decode against `all_ids_mapping.csv`, plus `encode_event()` (encodes ALL of process/machine/module/object BEFORE the agents see it) and optional tool schemas for on-demand LLM lookup | stdlib only |
| `parsers/*.py` | Per-system regex parsers, built against real ZC/ES/ATS samples | stdlib only |
| `severity_rules.py` | Deterministic severity classification + the pre-filter cost gate (toggle via `--no-filter`) | stdlib only |
| `schemas.py` | Pydantic response schemas (`FindingsResponse`, `RootCauseResponse`) used at the LLM call boundary | `pydantic` |
| `llm_client.py` | LangChain wrapper, THREE providers: `ChatAnthropic` (Claude, default), `ChatOllama` (glm-5.2:cloud), or `ChatOpenAI` (GLM/Zhipu, OpenAI-compatible). Also exposes `call_text()` for free-text chat | `langchain`, `langchain-anthropic`, `langchain-ollama`, `langchain-openai` |
| `memory_store.py` | Persistent cross-run incident history (JSON file) — lets root-cause synthesis reference past incidents for the same component | stdlib only |
| `excel_writer.py` | Appends every run's reports to a single accumulating `.xlsx` (Incidents + Run Log sheets, never overwritten) | `openpyxl` |
| `pipeline_runner.py` | Folder-based pipeline engine shared by `main.py` and `app.py` — takes 3 log folders + provider + mode, runs the graph, writes Excel | `langchain` (lazy), `openpyxl` (lazy) |
| `app.py` | Streamlit chat interface — folder-driven run + provider/mode/memory controls + Excel output + follow-up chat stage | `streamlit`, `openpyxl`, `langchain` |
| `agents/system_agent.py` | Per-system (ZC/ES/ATS) LLM analysis agent | `langchain` (via `llm_client.py`) |
| `agents/correlation.py` | Deterministic clustering + LLM root-cause synthesis, now with historical-memory context injected into the prompt | `langchain` (via `llm_client.py`) |
| `agents/report_agent.py` | Deterministic decode (now including the `machine` field) + Markdown report rendering | stdlib only |
| `graph.py` | LangGraph orchestration; accepts `min_severity` (filter toggle) and `memory_store` | `langgraph`, `langchain` |
| `streaming_consumer.py` | Kafka consumer, windowing, dedup | `kafka-python` |
| `observability.py` | LangSmith + OpenTelemetry wiring (automatic LLM tracing via LangChain) | `langsmith`, `opentelemetry-*` |
| `main.py` | Testing/batch CLI entry point — single files OR whole folders (`--zc-dir` etc.), `--no-filter`, `--provider`, `--output-dir`, `--excel-path` | — |
| `stream_main.py` | **Real-time production entry point** — wires Kafka streaming into the pipeline | `kafka-python`, `langchain` |
| `mock_llm_client.py` | Free, no-API-key rule-based fake LLM for testing pipeline wiring (now with `call_text()` for chat) | stdlib only |

## Using Ollama (glm-5.2:cloud) instead of Claude

```bash
python main.py --provider ollama --live
```

Configure via environment variables (see `llm_client.py`):
```bash
export LLM_PROVIDER=ollama            # or pass --provider ollama each time
export OLLAMA_MODEL=glm-5.2:cloud     # default already
export OLLAMA_BASE_URL=http://localhost:11434   # your local `ollama serve`
export OLLAMA_API_KEY=...             # ONLY if your Ollama Cloud setup needs auth
```
Try without `OLLAMA_API_KEY` first (local Ollama, no auth) — only add it if
Ollama rejects the request unauthenticated. This path has been validated
for wiring/structure only against a fake stand-in, NOT against a real
Ollama server — the exact auth parameter name in `_init_ollama()` may need
adjusting against your installed `langchain-ollama` version.

## Using GLM (Zhipu) directly — the third provider

In addition to Claude (anthropic) and Ollama, a third provider `glm` is
available. This talks to Zhipu AI's OpenAI-compatible chat endpoint directly
(via `langchain_openai.ChatOpenAI`), so it uses the same
`with_structured_output()` path Claude uses — no prose-parsing fallback is
needed.

```bash
python main.py --provider glm --live
```

Configure via environment variables (see `llm_client.py`):
```bash
export LLM_PROVIDER=glm
export GLM_MODEL=glm-4.6                 # default; override with your model id
export GLM_API_KEY=...                   # (or ZHIPUAI_API_KEY)
export GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # default already
```

Note the distinction: `--provider ollama` runs the GLM model *through a
local/cloud Ollama server* (model name `glm-5.2:cloud`), while `--provider
glm` hits Zhipu's API directly. Both expose the same `call_structured` /
`call_text` interface, so nothing else in the codebase cares which is active.

## Sending every log line to the LLM (no filter)

```bash
python main.py --no-filter --live
```
Disables the severity pre-filter entirely — every parsed line (including
INFO) goes to the agents. This is a real, supported option, not a
workaround, but it is roughly **40x** the API volume of the default
(measured: 144 of 6,306 lines pass the default filter; all 6,306 would go
through with `--no-filter`). Use deliberately, not by default.

## End-of-day batch mode (whole folders, not single files)

```bash
python main.py --zc-dir "E:\Logs\ZC" --es-dir "E:\Logs\ES" --ats-dir "E:\Logs\ATS" \
                --live --output-dir reports/2026-08-10
```
Every file in each folder is parsed and combined into ONE run — this is
what makes "one consolidated end-of-day report across all of today's log
files" possible instead of running per-file. `--output-dir` writes each
report as its own `.md` file in addition to printing it.

## Excel report output (one accumulating workbook per folder)

Every `--live`/`--mock` run can APPEND its incident reports to a single
`.xlsx` workbook via `--excel-path`. The workbook is created on first run
and **reused on every subsequent run — it is never overwritten**, so it
accumulates a full incident history across days/weeks.

```bash
python main.py --zc-dir "E:\Logs\ZC" --es-dir "E:\Logs\ES" --ats-dir "E:\Logs\ATS" \
               --provider ollama --live \
               --excel-path reports/incident_reports.xlsx --output-dir reports/
```

The workbook (`excel_writer.py`) has two sheets:
- **Incidents** — one row per generated report, fully flattened: `run_id`,
  `generated_at`, `incident_id`, `title`, `severity`, `systems_involved`,
  `affected_components`, `time_window`, `confidence`, `root_cause_summary`,
  `recommended_actions`, `evidence`. Every row is stamped with its `run_id`
  so you can tell which run produced it.
- **Run Log** — one row per run with run-level metadata: mode, provider,
  mapping CSV, file counts per subsystem, events parsed/flagged, report
  count, and the workbook path. This is the "what happened in this run"
  companion to the per-incident detail — including runs that produced zero
  reports (a clean day is still a recorded event, not a silent skip).

This is the **enhanced report persistence capability**: reports are no
longer just printed to stdout / written as loose `.md` files — they land in
a queryable, accumulating Excel audit trail. Persistent memory
(`memory/incident_history.json`) still runs alongside it; the Excel file is
the human-facing cumulative record, the JSON memory is the machine-facing
one the root-cause agent reads on the next run.

## Streamlit chat interface (interactive folder-driven stage)

`app.py` is a Streamlit app that wraps the whole pipeline in a UI:

```bash
streamlit run app.py
```

In the sidebar you point at the three log **folders** (ZC / ES / ATS — one
per subsystem type), choose the **provider** (Claude / Ollama / GLM) and
**mode** (Dry run / Live / Mock), toggle memory, and set the **Excel output
workbook** path. Hitting **Run analysis**:

1. Globs every matching file out of each folder and combines them into one
   run (folder-based, exactly like `main.py --*-dir`).
2. Runs the full pipeline with live stage progress shown in the page.
3. Shows summary metrics (files / events parsed / events flagged / reports)
   and renders each generated report.
4. **Appends** the reports to the configured `.xlsx` workbook (same
   append-never-overwrite semantics as `--excel-path` above).
5. Opens a **chat stage** below the reports: ask follow-up questions about
   the just-generated reports and the chosen provider answers with the
   reports (and recent memory history) as context.

The chat uses the same provider/mode the run used (Claude/Ollama/GLM in live
mode, or the rule-based mock in mock mode). Dry runs make no LLM calls, so
the chat stage is disabled for them. All options/modes that exist in `main.py`
are exposed here as UI controls; `main.py` itself still works unchanged for
CLI/batch use.

## Historical memory (accumulative analysis across days/weeks)

By default, `main.py --live`/`--mock` writes each incident to
`memory/incident_history.json` and, before asking the LLM for root cause,
looks up whether the same component had incidents in the last 14 days —
that history gets added directly into the root-cause prompt, so Claude/GLM
can say "this also happened 3 times last week" instead of treating every
run as isolated. Disable with `--no-memory`, or point elsewhere with
`--memory-path`.

Design note: this is a plain JSON file storing past incident summaries,
looked up deterministically and injected as prompt context — **not** a
LangChain Agent/Memory object handed to the model to manage itself. That
was a deliberate choice (see conversation): it gets you real
cross-run historical analysis while keeping every individual LLM call a
single, structured, validated request — the same reliability guarantee as
the rest of this pipeline. At real production volume, replace the JSON
file with SQLite (`memory_store.py`'s docstring notes this).

## Adding or removing a subsystem

The pipeline is built around `models.System` — a `str, Enum` whose members
(ZC, ES, ATS) are the join key threaded through every layer. Adding or
removing a subsystem is a checklist of small, localized edits, one per
layer, in this order. Each step below names the exact file and what to
change. Nothing here requires touching the deterministic core's logic — only
registration of the new system into each layer's tables.

### To ADD a new subsystem (e.g. a `SCADA` system)

1. **`models.py`** — add a member to the `System` enum:
   ```python
   class System(str, Enum):
       ZC = "ZC"
       ES = "ES"
       ATS = "ATS"
       SCADA = "SCADA"   # new
   ```
   The `.value` string is used everywhere as a dict key and in reports, so
   keep it short and uppercase.

2. **`parsers/scada_parser.py`** (new) — write a regex parser that yields
   `LogEvent(system=System.SCADA, ...)` from the new log format, mirroring
   `parsers/zc_parser.py` / `ats_parser.py` / `es_parser.py`. Export
   `parse_scada_line` and `parse_scada_file` (or `_csv`).

3. **`parsers/__init__.py`** — import and re-export the new parser functions
   and add them to `__all__`.

4. **`severity_rules.py`** — if the new system has its own severity cues
   (like ATS's `<CHANNEL>` tag), extend `classify()` with a branch for
   `System.SCADA`. If it's keyword-based like ZC/ES, the existing default
   path already covers it — no change needed.

5. **`mappings.py`** — nothing code-side to change (encode/decode is generic
   and keyed by `(system, mapping_type, name)`). Just make sure your
   `all_ids_mapping.csv` has rows with `system=SCADA` for whatever
   process/machine/object names you want encoded for correlation. Fields
   not in the table pass through unchanged (no-op on no match) — that's
   correct behavior, not a bug.

6. **`agents/system_agent.py`** — add a `System.SCADA` entry to the
   `_SYSTEM_SPECIFICS` dict with a short role + cues block for the new
   subsystem. `SYSTEM_PROMPTS` is now built automatically from that block
   plus the shared `_SUBTASK_SKELETON` (group → identify → severity →
   summarize → evidence → confidence → output contract), so you do NOT
   re-author the analysis steps — only the system-specific role line.
   `analyze_system_events()` is already generic — it dispatches on the
   `system` argument, so no other change is needed there.

7. **`graph.py`** — add a `node_scada_agent` node (copy `node_zc_agent` and
   swap the `System`), `g.add_node("scada_agent", node_scada_agent)`, and
   wire it into the fan-out/fan-in:
   ```python
   g.add_edge("filter", "scada_agent")
   g.add_edge("scada_agent", "correlate")
   ```
   The `findings` reducer (`operator.add`) already handles N agents writing
   in the same parallel step, so a 4th (or Nth) agent is safe.

8. **`main.py`** — add `--scada` / `--scada-dir` / `--scada-pattern` /
   `--skip-scada` arguments mirroring the existing `--zc*` set, resolve them
   with `_resolve_paths`, and add the parse+classify block in `dry_run()`
   (copy the ZC/ES/ATS branches). This keeps all existing options intact.

9. **`pipeline_runner.py`** — add `scada_dir` / `scada_pattern` parameters
   and the matching collect/parse/classify block, mirroring the ZC block.
   Update `RunResult`'s `scada_files` field and the `run_meta` passed to the
   Excel writer so the Run Log sheet records the new subsystem's file count.

10. **`app.py`** — add a "SCADA logs folder" sidebar input and pass it
    through to `run_pipeline(...)`. The provider/mode/memory/excel controls
    are subsystem-agnostic, so no other UI change is needed.

11. **`excel_writer.py`** — no change required. The Incidents sheet already
    records `systems_involved` per row, so a new system just appears as a new
    value in that column. If you want a per-subsystem file-count column in
    the Run Log sheet, extend `RUN_LOG_HEADERS` + the row in `append_run()`
    (and `run_meta` in the runner).

That's the whole subsystem: enum member → parser → parser export → (maybe)
severity branch → (maybe) mapping CSV rows → system prompt → graph node →
CLI flags → runner param → UI input. Every other piece (correlation,
root-cause synthesis, decode, report rendering, memory, Excel) is generic
and needs no change.

### To REMOVE a subsystem (e.g. drop `ATS`)

Reverse the checklist: delete the parser file, remove it from
`parsers/__init__.py`, delete its `SYSTEM_PROMPTS` entry, delete its graph
node + edges, delete its CLI flags and `dry_run` branch, delete its runner
param + `RunResult` field, delete its sidebar input, and finally remove the
enum member from `models.System`. Remove it from `all_ids_mapping.csv` too
(its rows become dead entries otherwise — harmless, but tidy). The
generic layers (correlation, root cause, decode, report, memory, Excel)
need no change; they simply never receive that system's events again.

> **Tip:** the enum is the single source of truth. If you forget step 1
> (the enum member) nothing downstream will recognize the new system; if
> you forget step 7 (the graph node) the system's events are parsed and
> filtered but never analyzed. A subsystem that's parsed but has no agent
> node is the most common "it compiles but produces no reports" mistake.

## Known gaps / next steps

- **ES severity rules are unvalidated against ES's OWN faults specifically**
  — a real ES fault sample (UDP socket bind failures, error code 10049)
  was reviewed manually during development and the object-name heuristic
  (`_guess_object_ref`) confirmed working correctly, but ES's severity
  keyword classification hasn't been checked against a wide range of ES
  fault types yet.
- **`encode_event()` only encodes EXACT matches against the mapping
  table.** Tested and confirmed: for ATS, `object_ref` (bare class name,
  e.g. "ConfigIntegerManager") encodes correctly, but `module` (the
  compound "Class::Function" string, e.g. "ConfigIntegerManager::Retrieve")
  never matches the table (which only stores bare class names) and passes
  through unencoded. Not a bug — the no-op-on-no-match behavior is
  correct — but "full encoding" isn't complete for every field on every
  system. If you need `module` encoded too, either add compound-name
  entries to `all_ids_mapping.csv`, or split `module` into class+function
  before encoding.
- **Ollama/GLM path is structurally tested only** (see `llm_client.py`
  docstring) — no real network available in the build environment to
  validate against an actual Ollama server or verify glm-5.2:cloud's
  tool-calling reliability with `with_structured_output`.
- **Correlation window (60s default)** was chosen as a reasonable starting
  point, not derived from your systems' actual inter-system latency
  characteristics — tune `DEFAULT_CORRELATION_WINDOW` in
  `agents/correlation.py` against real incident data.
- **No retry/backoff** on LLM API calls yet in `llm_client.py` — add
  before production (matters more for Ollama, which lacks the Anthropic
  SDK's built-in retry handling).
- **memory_store.py is a flat JSON file** — fine at daily-batch scale, but
  will slow down if it grows to tens of thousands of records; the natural
  upgrade is SQLite (noted in the file's docstring).
- **No persistence layer** for reports/incidents shown here — wire
  `on_flush`'s output into whatever you use for tickets/alerts (Slack,
  PagerDuty, a database) at the call site in `streaming_consumer.py`'s example above.
#   A I _ A g e n t _ L o g A n a l y t i c s 
 
 #   A I _ A g e n t _ L o g A n a l y t i c s 
 
 