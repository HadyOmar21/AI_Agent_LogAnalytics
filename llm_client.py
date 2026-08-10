"""
LangChain wrapper around the LLM, used for every analysis step in the
pipeline (per-system agents + root-cause synthesis).

Supports TWO providers, chosen at runtime -- nothing else in the codebase
needs to know or care which one is active, since both expose the same
call_structured(system_prompt, user_content, output_schema) interface:

  1. Anthropic Claude (default) -- via langchain_anthropic.ChatAnthropic.
  2. Ollama, for glm-5.2:cloud -- via langchain_ollama.ChatOllama, in
     EITHER of two connection modes:
       a) Local Ollama server (no auth) -- just needs OLLAMA_BASE_URL
          pointing at your running `ollama serve` instance
          (default http://localhost:11434) and the model already pulled
          there (or Ollama resolves the ":cloud" suffix itself).
       b) Ollama Cloud auth via an API key -- set OLLAMA_API_KEY; this is
          sent as a Bearer token header. Whether glm-5.2:cloud needs this
          depends on your Ollama account setup -- try (a) first, and only
          add the key if Ollama rejects the request unauthenticated.

Provider selection: set LLM_PROVIDER=anthropic (default) or
LLM_PROVIDER=ollama, or pass provider= explicitly to LLMClient().

IMPORTANT -- NOT exercised against a real network in the build
environment (no network access there, for either provider). The
Anthropic path was validated in earlier iterations against real
sample data once a real ANTHROPIC_API_KEY was available client-side;
the Ollama path has been validated for wiring/structure only against a
fake stand-in -- verify base_url/model/auth against your actual Ollama
setup before trusting it in production.
"""

from __future__ import annotations

import os
from typing import Type, TypeVar

from pydantic import BaseModel

# -- Anthropic models --
MODEL_MAIN = "claude-sonnet-5"            # per-system agents, root-cause synthesis
MODEL_FAST = "claude-haiku-4-5-20251001"  # optional: cheap secondary severity triage

# -- Ollama defaults --
OLLAMA_DEFAULT_MODEL = "glm-5.2:cloud"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"

# -- GLM (Zhipu) defaults -- the OpenAI-compatible direct API for the GLM
# family, distinct from running the same model through a local Ollama.
GLM_DEFAULT_MODEL = "glm-4.6"
GLM_DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0,
        provider: str | None = None,
    ):
        # "claude" is accepted as a friendlier alias for "anthropic".
        raw = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()
        self.provider = "anthropic" if raw == "claude" else raw

        if self.provider == "anthropic":
            self._init_anthropic(model, api_key, temperature)
        elif self.provider == "ollama":
            self._init_ollama(model, api_key, temperature)
        elif self.provider == "glm":
            self._init_glm(model, api_key, temperature)
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER '{self.provider}' -- use 'anthropic' "
                f"(='claude'), 'ollama', or 'glm'"
            )

    def _init_anthropic(self, model, api_key, temperature):
        from langchain_anthropic import ChatAnthropic  # lazy import -- keeps the
                                                          # deterministic parts of this
                                                          # repo dependency-free
        self.model_name = model or MODEL_MAIN
        self._chat = ChatAnthropic(
            model=self.model_name,
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            temperature=temperature,
        )

    def _init_ollama(self, model, api_key, temperature):
        from langchain_ollama import ChatOllama  # lazy import; `pip install langchain-ollama`

        self.model_name = model or os.environ.get("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)
        base_url = os.environ.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
        key = api_key or os.environ.get("OLLAMA_API_KEY")

        kwargs = {"model": self.model_name, "base_url": base_url, "temperature": temperature}
        if key:
            # Ollama Cloud auth: sent as a Bearer token. If your
            # langchain-ollama version doesn't accept `headers` directly,
            # check its docs for the current auth parameter name --
            # this is the one point in this file most likely to need
            # adjusting against your installed version.
            kwargs["headers"] = {"Authorization": f"Bearer {key}"}

        self._chat = ChatOllama(**kwargs)

    def _init_glm(self, model, api_key, temperature):
        """GLM (Zhipu AI) provider, via its OpenAI-compatible chat endpoint.
        Uses langchain_openai.ChatOpenAI pointed at GLM's API, so the same
        with_structured_output() path Claude uses works here too (GLM's
        OpenAI-compatible API implements tool-calling). The model defaults to
        glm-4.6 but is fully overridable via the GLM_MODEL env var.

        Configure with env vars (or pass model=/api_key= explicitly):
            GLM_MODEL=glm-4.6
            GLM_API_KEY=...            # (or ZHIPUAI_API_KEY)
            GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
        """
        from langchain_openai import ChatOpenAI  # lazy import; `pip install langchain-openai`

        self.model_name = model or os.environ.get("GLM_MODEL", GLM_DEFAULT_MODEL)
        base_url = os.environ.get("GLM_BASE_URL", GLM_DEFAULT_BASE_URL)
        key = api_key or os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")

        self._chat = ChatOpenAI(
            model=self.model_name,
            api_key=key,
            base_url=base_url,
            temperature=temperature,
        )

    def call_structured(self, system_prompt: str, user_content: str, output_schema: Type[T]) -> T:
        """Runs system_prompt + user_content through the configured LLM
        and returns a validated instance of output_schema (a pydantic
        BaseModel from schemas.py). Raises a pydantic ValidationError if
        the model's response doesn't match the schema -- this is the
        "force structured output, never free text" guarantee, and it
        applies identically regardless of which provider is active.

        Note: with_structured_output's DEFAULT method relies on native
        tool-calling, which Claude supports robustly. GLM-5.2 via Ollama
        has been observed to ignore tool-calling entirely and answer in
        free-form markdown prose instead -- and passing
        method="json_schema" through with_structured_output did NOT fix
        this in testing (still got prose back), which means the
        installed langchain-ollama version isn't actually wiring that
        method to Ollama's native structured-output support -- it's
        silently falling back to something else. Rather than depend on
        that wrapper behaving correctly, the Ollama path below talks to
        Ollama's native `format` parameter directly (a raw JSON schema
        passed straight to the API, which Ollama uses to constrain
        token generation at decode time) and parses the result
        ourselves. This is provider-specific on purpose.
        """
        if self.provider == "ollama":
            return self._call_structured_ollama(system_prompt, user_content, output_schema)

        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        structured_llm = self._chat.with_structured_output(output_schema)
        chain = prompt | structured_llm
        return chain.invoke({"input": user_content})

    def call_text(self, system_prompt: str, user_content: str) -> str:
        """Free-text chat call -- the counterpart to call_structured(). Used by
        the Streamlit chat stage to let an engineer ask follow-up questions
        about the generated reports without forcing the response into a
        schema. Returns the model's plain text answer.

        Works identically across all three providers: the LangChain chat
        object's .invoke() returns a message whose .content we extract. For
        Ollama the content can be a list of part dicts, so we reuse the same
        _extract_text helper the structured path uses.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        response = self._chat.invoke(messages)
        return self._extract_text(response.content)

    def _call_structured_ollama(self, system_prompt: str, user_content: str, output_schema: Type[T]) -> T:
        """Ollama-specific structured output path. Bypasses
        with_structured_output entirely and passes the JSON schema via
        Ollama's native `format` field, which constrains generation at
        the token level rather than relying on the model choosing to
        emit a tool call. `format` is accepted as a bind()-time kwarg
        by ChatOllama and forwarded to the underlying ollama client's
        chat() call.

        Confirmed in testing against Ollama Cloud + glm-5.2:cloud: the
        `format=` schema constraint is NOT honored -- the model still
        returns free markdown prose, no JSON at all. This is a
        real limitation (likely: cloud-hosted models proxy to backend
        infra that doesn't implement Ollama's local grammar-constrained
        decoding), not a wiring bug -- confirmed because auth, base
        URL, and the request itself all succeed; only the format
        constraint is ignored. So below we don't trust format= to do
        the work. Instead: (1) explicit "JSON only" instructions baked
        into the prompt, schema included inline, and (2) if that still
        parses as non-JSON, one self-repair round-trip where we show
        the model its own bad output plus the exact parse error and
        ask it to fix it into valid JSON. This costs one extra call in
        the failure case but is much more reliable against a model
        that won't structurally guarantee JSON.
        """
        import json
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        schema = output_schema.model_json_schema()
        json_instructions = (
            f"{system_prompt}\n\n"
            "IMPORTANT: Respond with ONLY a single raw JSON object -- "
            "no markdown headers, no bullet points, no code fences, no "
            "commentary before or after. The JSON must validate against "
            f"this JSON Schema:\n{json.dumps(schema)}"
        )

        messages = [
            SystemMessage(content=json_instructions),
            HumanMessage(content=user_content),
        ]
        response = self._chat.invoke(messages)
        raw = self._extract_text(response.content)

        parsed = self._try_parse(raw)
        if parsed is not None:
            return output_schema.model_validate(parsed)

        # Self-repair round-trip: show the model its own output + the
        # error, ask it to fix it.
        repair_prompt = (
            "Your previous response was not valid JSON. Here is what "
            f"you sent:\n\n{raw}\n\n"
            "Convert this into ONLY a single raw JSON object matching "
            f"this JSON Schema, with no other text:\n{json.dumps(schema)}"
        )
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content=repair_prompt))
        response = self._chat.invoke(messages)
        raw2 = self._extract_text(response.content)

        parsed = self._try_parse(raw2)
        if parsed is not None:
            return output_schema.model_validate(parsed)

        raise ValueError(
            f"Ollama model '{self.model_name}' did not return valid JSON "
            f"even after a repair retry. First attempt:\n{raw[:300]}\n\n"
            f"Repair attempt:\n{raw2[:300]}"
        )

    @staticmethod
    def _extract_text(content) -> str:
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return content

    @staticmethod
    def _try_parse(raw: str) -> dict | None:
        """Try direct JSON parse; if that fails, try pulling the first
        {...} block out of surrounding prose/code fences as a fallback."""
        import json
        import re

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None