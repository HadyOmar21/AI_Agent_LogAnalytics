"""
Streamlit chat interface for the log-analytics pipeline -- the interactive,
folder-driven stage.

You point it at three log FOLDERS (one per subsystem: ZC / ES / ATS), pick a
provider (Claude / Ollama / GLM) and a mode (dry / live / mock), hit Run, and:

  1. The agent reads every file out of each folder, combines them into one
     run, and runs the full pipeline (parse -> filter -> agents -> correlate
     -> root cause -> report).
  2. Generated incident reports are shown in the UI AND appended to a single
     accumulating Excel workbook (excel_writer.py) -- the same .xlsx every
     run, never overwritten, so history grows run-over-run.
  3. A chat stage lets you ask follow-up questions about the just-generated
     reports; the chosen provider answers with the reports as context.

Run it:
    streamlit run app.py

All options/modes that exist in main.py are preserved here as UI controls;
main.py itself is unchanged in behavior and still works for CLI use.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Make sibling modules importable when running `streamlit run app.py` from
# the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline_runner import run_pipeline  # noqa: E402
from excel_writer import report_count  # noqa: E402
from agents.report_agent import render_markdown  # noqa: E402

# -- provider / mode menus (kept in sync with main.py + llm_client.py) --
PROVIDERS = {
    "Claude (Anthropic)": "anthropic",
    "Ollama (glm-5.2:cloud)": "ollama",
    "GLM (Zhipu, OpenAI-compatible)": "glm",
}
MODES = {
    "Dry run (no LLM, free)": "dry",
    "Live (real LLM calls)": "live",
    "Mock (free, rule-based fake LLM)": "mock",
}

# Subtask-structured chat prompt: the follow-up Q&A stage is broken into
# ordered subtasks so the model stays scoped to the just-generated reports
# (and any provided memory history) instead of free-associating.
CHAT_SYSTEM_PROMPT = """\
You are a senior rail-signalling reliability engineer assistant. The user just
ran an automated multi-agent log-analytics pipeline over ZC/ES/ATS log folders,
and the incident reports below were generated. Answer follow-up questions
about those reports.

Work through these subtasks IN ORDER:

### Subtask 1 -- Ground yourself
Read the generated reports (and any recent incident history provided). These
are your only source of truth for this conversation.

### Subtask 2 -- Classify the question
Decide whether the question is about the reports (a root cause, a comparison
between incidents, a time window/severity clarification, a recommended next
step) or unrelated to them.

### Subtask 3 -- Answer from the reports
Answer accurately and concisely, citing only what is in the reports. Explain
root causes, compare incidents, suggest deeper investigation steps, or clarify
severity/time windows as asked.

### Subtask 4 -- Stay honest
If a question is unrelated to the reports, say so rather than guessing. Never
invent incidents, evidence, ids, or events that are not in the reports.
"""


def _build_chat_client(mode: str, provider: str):
    """Build the LLM client used by the chat stage. Mirrors pipeline_runner's
    client selection so chat answers come from the same provider the run used."""
    if mode == "mock":
        from mock_llm_client import MockLLMClient
        return MockLLMClient()
    from llm_client import LLMClient
    return LLMClient(provider=provider)


def _reports_context(reports: list) -> str:
    """Compact text rendering of the run's reports, injected into the chat
    system prompt so the model has the reports as context."""
    if not reports:
        return ("(No incident reports were generated in this run -- the logs "
                "may have been clean, or the run was a dry run.)")
    parts = []
    for i, r in enumerate(reports, start=1):
        parts.append(f"### Report {i}\n{render_markdown(r)}")
    return "\n\n".join(parts)


def _memory_context(memory_path: str | None) -> str:
    """Optional: include recent incident history so chat answers can reference
    'this also happened before'. Best-effort -- never blocks the chat."""
    if not memory_path or not Path(memory_path).exists():
        return ""
    try:
        from memory_store import MemoryStore
        store = MemoryStore(memory_path)
        if not store._records:
            return ""
        recent = store._records[-5:][::-1]
        lines = ["Recent incident history (most recent first):"]
        for r in recent:
            lines.append(f"- [{r.get('max_severity','?')}] "
                         f"{r.get('systems_involved','?')}: "
                         f"{r.get('root_cause_summary','')[:160]}")
        return "\n".join(lines)
    except Exception:
        return ""


def _save_uploads_to_tempdir(files, label: str) -> str | None:
    """Save Streamlit UploadedFile objects into a fresh temp dir on the server
    and return that dir's path (or None if nothing was uploaded).

    This is what makes the pipeline work on a REMOTE server: the user uploads
    their ZC/ES/ATS log files through the browser; we materialize them onto the
    server's local filesystem, then the pipeline reads them exactly like any
    other log folder -- no shared/network filesystem required."""
    if not files:
        return None
    tmp = tempfile.mkdtemp(prefix=f"logagent_{label}_")
    for f in files:
        # f.name can carry a browser-side path; keep only the basename.
        out = Path(tmp) / Path(f.name).name
        out.write_bytes(f.getvalue())
    return tmp


def main():
    st.set_page_config(page_title="Log Analytics Agent", page_icon="🛤️",
                       layout="wide")
    st.title("🛤️ Log Analytics Multi-Agent Pipeline")
    st.caption("ZC / ES / ATS log folders → per-system agents → correlate → "
                "root cause → report → Excel + chat")

    # ------------------------------------------------------------------ sidebar
    with st.sidebar:
        st.header("Configuration")
        st.caption("Provide ONE subsystem's logs per field (ZC/ES/ATS). On a "
                   "remote server, upload the files through the browser; on a "
                   "local machine you can point at folders instead.")

        input_mode = st.radio(
            "Input method",
            ["Upload log files", "Local folder paths"],
            index=0, horizontal=True,
            help="Upload when Streamlit runs on a server that can't see your "
                 "log files; Local paths when the logs are on the same machine.")

        # Defaults so both branches leave the variables defined.
        zc_dir = es_dir = ats_dir = ""
        zc_uploads = es_uploads = ats_uploads = None

        if input_mode == "Upload log files":
            st.caption("📁 Pick the log files from your computer — they're "
                       "copied onto the server for this run.")
            zc_uploads = st.file_uploader(
                "ZC log CSV(s)", type=["csv"], accept_multiple_files=True,
                help="Zero-cost Zone Controller logs. One or more .csv files.")
            es_uploads = st.file_uploader(
                "ES log file(s)", type=["log", "txt"], accept_multiple_files=True,
                help="Electronic Interlocking logs. One or more .log/.txt files.")
            ats_uploads = st.file_uploader(
                "ATS trace log(s)", type=["log", "txt"], accept_multiple_files=True,
                help="Automatic Train Supervision traces. One or more .log/.txt files.")
        else:
            zc_dir = st.text_input("ZC logs folder", value="",
                                   placeholder="e.g. E:\\Logs\\ZC  (pattern *.csv)",
                                   help="Folder of ZC log CSVs (glob pattern *.csv)")
            es_dir = st.text_input("ES logs folder", value="",
                                   placeholder="e.g. E:\\Logs\\ES  (pattern *.log)",
                                   help="Folder of ES log files (glob pattern *.log)")
            ats_dir = st.text_input("ATS logs folder", value="",
                                    placeholder="e.g. E:\\Logs\\ATS  (pattern *.log)",
                                    help="Folder of ATS trace logs (glob pattern *.log)")

        mapping_csv = st.text_input("Mapping CSV", value="sample_data/all_ids_mapping.csv")

        excel_path = st.text_input(
            "Excel output workbook", value="reports/incident_reports.xlsx",
            help="Every run APPENDS to this same .xlsx -- it is never "
                 "overwritten, so the workbook accumulates a full history.")

        provider_label = st.selectbox("LLM provider", list(PROVIDERS.keys()),
                                      index=0)
        mode_label = st.radio("Mode", list(MODES.keys()), index=0)
        use_memory = st.checkbox("Use persistent memory (cross-run history)",
                                 value=True)
        no_filter = st.checkbox("Send ALL severities to LLM (no filter, costly)",
                                value=False,
                                help="Disables the cost-saving pre-filter. "
                                     "Much higher API usage -- use deliberately.")

        provider = PROVIDERS[provider_label]
        mode = MODES[mode_label]

        st.divider()
        if excel_path:
            try:
                existing = report_count(excel_path)
                st.caption(f"📊 Workbook `{excel_path}` currently has "
                           f"**{existing}** incident row(s).")
            except Exception:
                st.caption(f"📊 Workbook: `{excel_path}` (new on first run)")

        run_btn = st.button("🚀 Run analysis", type="primary", use_container_width=True)

    # ------------------------------------------------------------- run + report
    if run_btn:
        if input_mode == "Upload log files":
            # Materialize uploaded files onto the server, then run over them.
            zc_dir = _save_uploads_to_tempdir(zc_uploads, "zc")
            es_dir = _save_uploads_to_tempdir(es_uploads, "es")
            ats_dir = _save_uploads_to_tempdir(ats_uploads, "ats")
            # Each temp dir holds ONLY what the user uploaded for that subsystem,
            # so match every file regardless of extension (the uploader's type
            # filter already guarded what was accepted).
            zc_pattern = es_pattern = ats_pattern = "*"
        else:
            zc_pattern, es_pattern, ats_pattern = "*.csv", "*.log", "*.log"
        _do_run(zc_dir, es_dir, ats_dir, mapping_csv, excel_path,
                provider, mode, use_memory, no_filter,
                zc_pattern=zc_pattern, es_pattern=es_pattern,
                ats_pattern=ats_pattern)

    # ------------------------------------------------------------- chat stage
    _render_chat()


def _do_run(zc_dir, es_dir, ats_dir, mapping_csv, excel_path,
            provider, mode, use_memory, no_filter,
            zc_pattern="*.csv", es_pattern="*.log", ats_pattern="*.log"):
    """Execute the folder-based pipeline and render results + Excel write."""
    progress_box = st.container()
    stages = []

    def progress(stage: str, msg: str):
        # Accumulate stage lines into the live status box.
        stages.append(f"**{stage}**: {msg}")
        progress_box.markdown("\n\n".join(stages))

    # Dry runs produce no reports -> don't touch the accumulating workbook.
    # Only live/mock runs append to Excel (matches main.py behavior).
    run_excel = excel_path if mode in ("live", "mock") else None

    try:
        with st.spinner(f"Running pipeline ({mode} / {provider})..."):
            result = run_pipeline(
                zc_dir=zc_dir or None,
                es_dir=es_dir or None,
                ats_dir=ats_dir or None,
                zc_pattern=zc_pattern,
                es_pattern=es_pattern,
                ats_pattern=ats_pattern,
                mapping_csv=mapping_csv,
                mode=mode,
                provider=provider,
                no_filter=no_filter,
                use_memory=use_memory,
                excel_path=run_excel,
                progress=progress,
            )
    except (NotADirectoryError, ValueError) as e:
        st.error(f"Could not start run: {e}")
        return
    except ImportError as e:
        st.error(f"Missing dependency for this mode/provider: {e}\n"
                 f"Install with: pip install -r requirements.txt")
        return
    except Exception as e:
        st.error(f"Run failed: {e}")
        return

    st.session_state["last_result"] = result

    # ---- summary metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files (ZC/ES/ATS)",
              f"{len(result.zc_files)}/{len(result.es_files)}/{len(result.ats_files)}")
    c2.metric("Events parsed", result.events_parsed)
    c3.metric("Events flagged", result.events_flagged)
    c4.metric("Reports", result.reports_count)

    if result.excel_path and result.excel_run_id:
        st.success(f"📊 Appended **{result.reports_count}** report(s) to "
                   f"`{result.excel_path}` (run `{result.excel_run_id}`). "
                   f"The workbook now accumulates every run's history.")
    elif mode == "dry":
        st.info("Dry run — no reports generated and Excel not touched. "
                "Switch to Live or Mock to generate reports.")

    # ---- reports ----
    st.subheader("Generated reports")
    if not result.reports:
        st.write("_No incidents crossed the reporting threshold — the logs "
                 "may be clean. This is expected for a healthy sample._")
    for i, r in enumerate(result.reports, start=1):
        with st.expander(f"Report {i}: {r.title}", expanded=(i == 1)):
            st.markdown(render_markdown(r))

    if result.errors:
        st.warning("Issues during run: " + " | ".join(result.errors))


def _render_chat():
    """The follow-up chat stage. Lives off the most recent run's reports,
    stored in session_state by _do_run()."""
    st.divider()
    st.subheader("💬 Ask about the reports")
    result = st.session_state.get("last_result")
    mode = st.session_state.get("chat_mode")
    provider = st.session_state.get("chat_provider")

    if result is None:
        st.caption("Run an analysis first, then ask follow-up questions here.")
        return
    if result.mode == "dry":
        st.caption("Dry runs produce no reports and make no LLM calls, so the "
                   "chat stage is disabled. Re-run in Live or Mock mode to chat.")
        return

    st.caption(f"Chatting about **{result.reports_count}** report(s) from the "
               f"last run ({result.mode} / {result.provider}).")

    # init message history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    # reset history whenever a new run replaces the report context
    if st.session_state.get("chat_context_id") != id(result):
        st.session_state.chat_messages = []
        st.session_state.chat_context_id = id(result)

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Ask a follow-up question about the reports...")
    if not user_q:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    system_prompt = (CHAT_SYSTEM_PROMPT
                     + "\n\n## Generated reports (context)\n"
                     + _reports_context(result.reports)
                     + ("\n\n## " + _memory_context(result.memory_path)
                        if _memory_context(result.memory_path) else ""))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                client = _build_chat_client(result.mode, result.provider)
                answer = client.call_text(system_prompt, user_q)
            except Exception as e:
                answer = (f"(Could not reach the {result.provider} provider for "
                          f"chat: {e}. Check your API key / endpoint.)")
        st.markdown(answer)
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()