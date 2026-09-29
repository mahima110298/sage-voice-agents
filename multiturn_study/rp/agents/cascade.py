"""Cascade adapter: ASR transcript -> text LLM with tool calling (OpenRouter chat
completions, non-streaming). The second agent architecture under study, next to
the speech-to-speech adapters (Gemini Live, OpenAI Realtime): same shared loop
(`rp.agents.base.run_case`), same interceptor/backend wiring, same per-turn
record shape -- only how user input arrives and model output leaves differs.

  - user input: `rp.asr.transcribe(wav)` (run_case already calls this once per
    turn into `state.current.user_text`, the same cached transcript SAGE and
    every other agent see -- the cascade makes no separate ASR call);
  - tools = rp.tools.schemas_for(domain), wrapped as OpenAI-style function
    tools: [{"type": "function", "function": schema} for schema in schemas];
  - instructions = prompt + interceptor.prompt_suffix() + backend.session_context()
    (system_instruction(), from rp.agents.base -- identical to both speech
    adapters), sent once as the system message; the message list then persists
    for the whole case (conversation memory carries across turns, same as a
    Realtime/Live session);
  - every function call goes through handle_call (the Interceptor) before the
    backend, exactly like the speech adapters: a blocked call gets the
    Decision's message back as its tool result, never the backend's;
  - MAX_TOOL_RESPONSE_ROUNDS = 4, mirroring both speech adapters: a tripped
    valve is flagged usage["safety_valve"] = "tool_rounds" (not an error) and
    the turn ends without requesting a further completion, though every tool
    call already made is still answered so the message list stays well-formed
    for the next turn;
  - latency_ms = wall time from turn start (the ASR transcript is already
    cached by the time run_turn is called) to the first completion's return.
    OpenRouter's chat/completions endpoint is called non-streaming here, so
    "first content token or tool call" and "the first completion's return"
    are the same instant -- there is nothing earlier to time against.
  - usage: each completion's `usage` dict (prompt_tokens, completion_tokens,
    total_tokens, ...) is summed into the turn's usage via the same
    flatten-nested-dicts-to-dotted-keys helper the OpenAI Realtime adapter
    uses, so usage keys are directly comparable across agents where the
    underlying fields match (they will not always match: it's a different
    endpoint/provider).

CASCADE_LLM ("openai/gpt-4.1-mini") was chosen from OpenRouter's public model
listing (https://openrouter.ai/api/v1/models, fetched 2026-09-28; no API key
needed, no cost) as a current mid-tier, non-Anthropic model whose
`supported_parameters` includes "tools", distinct from the verifier's model
(google/gemini-2.5-flash-lite, rp.verifier.VERIFIER_MODEL). Pricing on that
listing: $0.40 / 1M input tokens, $1.60 / 1M output tokens (see task-16-report.md
for the full listing excerpt this was chosen from). Temperature is left unset
(provider default), same as both speech agents.

Open cascade (Task 16b; spec section 5): CascadeAdapter(llm_model, asr=..., llm_base_url=...)
  - asr="harness" (default): the LLM hears rp.asr.transcribe(wav), as above.
  - asr="groq-whisper": the LLM hears Groq `whisper-large-v3`'s transcript of the
    turn wav (rp.asr.whisper_transcribe: POST {GROQ}/audio/transcriptions; cached
    next to the wav as input_turn{n}.whisper.json by the same atomic cache as the
    harness ASR; warm with `python3 -m rp.asr --whisper --split ...`, and the
    runner refuses to start open-cascade without it). SAGE and the verifier still read
    state.current.user_text = the harness transcript (run_case sets it), so only
    the agent's own "ears" change. Whisper is cached like the harness ASR and
    its cost ($0.111 per audio hour, 10 s minimum per request; pricing.json) is
    NOT added to the turn usage (an unknown usage key would make the runner's
    cost None); latency_ms likewise excludes it (turn start is after ASR, same
    as the closed cascade).
  - llm_base_url=OPENROUTER | GROQ picks the OpenAI-compatible chat endpoint and
    its key (OPENROUTER_API_KEY / GROQ_API_KEY from multiturn_study/.env). Groq
    serves openai/gpt-oss-120b (open weights) with tool calling.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import requests

from rp import asr
from rp.agents.base import run_case

_STUDY = Path(__file__).resolve().parents[2]
_SCRIPTS = str(_STUDY / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from env_utils import load_env_file  # noqa: E402

OPENROUTER = "https://openrouter.ai/api/v1"
GROQ = "https://api.groq.com/openai/v1"
OPENROUTER_URL = OPENROUTER + "/chat/completions"
KEY_ENV = {OPENROUTER: "OPENROUTER_API_KEY", GROQ: "GROQ_API_KEY"}
CASCADE_LLM = "openai/gpt-4.1-mini"
ASR_MODES = ("harness", "groq-whisper")
MAX_TOOL_RESPONSE_ROUNDS = 4  # same cap as both speech adapters
TURN_TIMEOUT_S = 60


def _http_error(r) -> RuntimeError:
    # status + a short body excerpt (the provider's error message); never the request headers
    return RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")


class ChatClient:
    """Minimal sync client for an OpenAI-compatible chat/completions endpoint
    (OpenRouter or Groq), normalized to the small dict shape the adapters need:
    {"tool_calls": [...], "content": str, "usage": {...}}. A fake with this same
    `.chat` signature drives the offline tests; this class itself is only
    exercised by the live smoke tests."""

    def __init__(self, base_url: str = OPENROUTER):
        if base_url not in KEY_ENV:
            raise ValueError(f"unknown llm_base_url {base_url!r}")
        self.url = base_url + "/chat/completions"
        self.key_env = KEY_ENV[base_url]

    def chat(self, model: str, messages: list[dict], tools: list[dict]) -> dict:
        load_env_file()
        r = requests.post(self.url, timeout=TURN_TIMEOUT_S,
                          headers={"Authorization": f"Bearer {os.environ[self.key_env]}"},
                          json={"model": model, "messages": messages, "tools": tools})
        if r.status_code != 200:
            raise _http_error(r)
        d = r.json()
        if "error" in d and not d.get("choices"):
            raise RuntimeError(f"provider error: {str(d['error'])[:200]}")
        msg = (d.get("choices") or [{}])[0].get("message") or {}
        calls = [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                 for c in (msg.get("tool_calls") or [])]
        return {"tool_calls": calls, "content": msg.get("content") or "", "usage": d.get("usage") or {}}


OpenRouterChatClient = ChatClient  # the pre-16b name (OpenRouter is the default endpoint)


def add_usage(acc: dict, usage: dict, prefix: str = "") -> None:
    """Sum one completion's usage into the turn's totals; nested dicts become
    dotted keys (same helper as rp.agents.openai_realtime)."""
    for k, v in (usage or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            acc[prefix + k] = acc.get(prefix + k, 0) + v
        elif isinstance(v, dict):
            add_usage(acc, v, f"{prefix}{k}.")


class _Session:
    """Holds the persistent message list (and the OpenAI-style tools payload)
    for one case; the same instance is passed to run_turn for every turn."""

    def __init__(self, messages: list[dict], tools: list[dict]):
        self.messages = messages
        self.tools = tools


async def chat_loop(client, model: str, session: _Session, cur, on_call, out) -> None:
    """One user turn's completion/tool loop, shared by CascadeAdapter and
    rp.agents.audio_chat.AudioChatAdapter. The caller has already appended the
    turn's user message to session.messages."""
    rounds = 0
    while True:
        t0 = time.perf_counter()
        resp = client.chat(model, session.messages, session.tools)
        if cur.latency_ms is None:
            cur.latency_ms = (time.perf_counter() - t0) * 1000
        add_usage(cur.usage, resp.get("usage") or {})
        calls = resp.get("tool_calls") or []
        content = resp.get("content") or ""
        assistant_msg = {"role": "assistant", "content": content}
        if calls:
            assistant_msg["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
        session.messages.append(assistant_msg)
        piece = content.strip()
        if piece:
            # keep every utterance the turn produces (this is what a cascade user
            # would hear, one round at a time) -- a single space between pieces,
            # same as the OpenAI Realtime adapter's between-responses separator.
            out.text = f"{out.text} {piece}" if out.text else piece
        if not calls:
            return
        for c in calls:
            try:
                args = json.loads(c["arguments"] or "{}")
            except ValueError:
                args = {"_raw": c["arguments"]}
            result = on_call(c["name"], args)
            session.messages.append({"role": "tool", "tool_call_id": c["id"],
                                     "content": json.dumps(result)})
        if rounds >= MAX_TOOL_RESPONSE_ROUNDS:
            # safety valve: every pending call has already been answered above (so the
            # message list stays well-formed for the next turn) but no further
            # completion is requested this turn. Flagged, not an error -- the case
            # continues, same as both speech adapters.
            cur.usage["safety_valve"] = "tool_rounds"
            return
        rounds += 1


class CascadeAdapter:
    def __init__(self, llm_model: str = CASCADE_LLM, client=None, asr: str = "harness",
                 llm_base_url: str = OPENROUTER):
        if asr not in ASR_MODES:
            raise ValueError(f"asr must be one of {ASR_MODES}, got {asr!r}")
        self.model = llm_model
        self.asr = asr
        self.client = client or ChatClient(llm_base_url)

    @asynccontextmanager
    async def connect(self, case, tools, system_instruction):
        payload_tools = [{"type": "function", "function": t} for t in tools]
        yield _Session([{"role": "system", "content": system_instruction}], payload_tools)

    async def run_turn(self, session: _Session, wav, on_call, state, out) -> None:
        cur = state.current
        # cur.user_text is the harness ASR (what SAGE sees); with asr="groq-whisper"
        # the agent itself hears Whisper's transcript instead.
        heard = asr.whisper_transcribe(wav) if self.asr == "groq-whisper" else cur.user_text
        session.messages.append({"role": "user", "content": heard})
        await chat_loop(self.client, self.model, session, cur, on_call, out)

    async def run(self, case: dict, interceptor, system_prompt: str):
        return await run_case(self, case, interceptor, system_prompt,
                              case_timeout_s=TURN_TIMEOUT_S * len(case["steps"]) + 30)
