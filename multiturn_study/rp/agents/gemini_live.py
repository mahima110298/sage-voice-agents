"""Gemini Live speech-to-speech adapter.

Port of `run_sut_multiturn_gemini` (scripts/run_multiturn.py), with the per-case
loop moved to rp.agents.base.run_case and these changes only:
  - tools come from rp.tools.schemas_for(domain) (credential-cue overlay);
  - system instruction = prompt + interceptor.prompt_suffix() + backend.session_context();
  - every function call goes through handle_call (the Interceptor) before the backend;
  - latency_ms = end of user audio (activity_end sent) -> first model output
    (transcription text, audio, or tool call) for the turn;
  - usage = usage_metadata summed over the turn's messages;
  - the pilot's clarify-budget and guardrail branches are gone (they are
    interceptors now: B3 = ConfirmGuard).

The same LiveConnectConfig is used for every Gemini model. Two cross-model
differences surfaced in the Task 14 smoke test, and both are handled here in a
way that behaves identically for all Gemini models:
  - gemini-3.8-live treats function calls as non-blocking by default: it emits
    turn_complete right after the tool call, then speaks its answer to the tool
    result only after we have moved on (the reply is lost / cut off by the next
    turn's audio). Every FunctionDeclaration is therefore declared
    behavior=BLOCKING, which is what gemini-3.1-flash-live-preview and
    gemini-2.5-flash-native-audio-latest already do by default.
  - gemini-2.5-flash-native-audio-latest streams audio in many small messages
    (a two-sentence reply was >80 messages), so counting every message against
    MAX_MESSAGES_PER_TURN cut turns off before turn_complete. Pure output-stream
    chunks (audio / transcription with nothing else) no longer count toward the
    cap; the per-turn timeout still bounds a turn that never completes.
A turn that trips a safety valve (tool-response rounds or message cap) is not
an error: it keeps what was captured, is flagged with
usage["safety_valve"] = "tool_rounds" | "messages", and the case continues
with the next turn (pilot behavior). Only real exceptions/timeouts are errors.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

from rp.agents.base import run_case

_STUDY = Path(__file__).resolve().parents[2]
_SCRIPTS = str(_STUDY / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_ACCENT = str(_STUDY.parent / "accent_sweep")
if _ACCENT not in sys.path:
    sys.path.append(_ACCENT)  # appended, AFTER scripts, so harness_defs resolves to scripts/

from env_utils import load_env_file  # noqa: E402
from gemini_tools import wav_to_pcm16_16k  # noqa: E402

CHUNK_BYTES = 3200  # 100ms of 16kHz/16-bit mono PCM
TURN_TIMEOUT_S = 60
# Raised from 2 in the pilot: with a real mock backend the agent legitimately
# chains lookup -> action within one turn, and some turns expect two calls
# (e.g. verify_authorization + the action itself).
MAX_TOOL_RESPONSE_ROUNDS = 4
MAX_MESSAGES_PER_TURN = 80


def add_usage(acc: dict, um) -> None:
    """Add one message's usage_metadata into the turn's running totals: every
    integer field is summed, and per-modality detail lists become
    '<field>.<MODALITY>' keys."""
    if um is None:
        return
    d = um.model_dump(mode="json", exclude_none=True) if hasattr(um, "model_dump") else dict(vars(um))
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            acc[k] = acc.get(k, 0) + v
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and "modality" in item:
                    key = f"{k}.{item['modality']}"
                    acc[key] = acc.get(key, 0) + int(item.get("token_count") or 0)


def _has_audio(sc) -> bool:
    mt = getattr(sc, "model_turn", None)
    return bool(mt and any(getattr(p, "inline_data", None) for p in (mt.parts or [])))


def _is_stream_chunk(message) -> bool:
    """True for a message that only carries streamed output (audio and/or
    transcription) -- these do not count toward MAX_MESSAGES_PER_TURN."""
    sc = message.server_content
    if message.tool_call or sc is None or sc.turn_complete:
        return False
    return bool((sc.output_transcription and sc.output_transcription.text) or _has_audio(sc))


class GeminiLiveAdapter:
    def __init__(self, model: str, client=None):
        self.model = model
        self._client_obj = client

    def _client(self):
        if self._client_obj is None:
            load_env_file()
            self._client_obj = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client_obj

    def _config(self, tools: list[dict], system_instruction: str) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters=t["parameters"],
                                          # See module docstring: gemini-3.8-live defaults to
                                          # non-blocking calls; pin blocking for every model.
                                          behavior=types.Behavior.BLOCKING) for t in tools])],
            # Manual turn-taking: we mark each user turn with activity_start/end.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
            ),
        )

    def connect(self, case, tools, system_instruction):
        return self._client().aio.live.connect(model=self.model, config=self._config(tools, system_instruction))

    async def run_turn(self, session, wav, on_call, state, out) -> None:
        pcm = wav_to_pcm16_16k(wav)
        cur = state.current

        # BUG FIX (three-part, found via debug_turn2.py):
        # 1. session.receive() is scoped to ONE turn and self-closes after
        #    turn_complete -- must be called fresh per turn, not once for the
        #    whole session (reusing one generator silently yields nothing on
        #    every call after the first).
        # 2. The break condition must wait for the real `turn_complete` signal,
        #    not `generation_complete` -- generation_complete fires before the
        #    server has finished the turn server-side; sending the next turn's
        #    audio too early was interpreted as a barge-in interruption
        #    ({'server_content': {'interrupted': True}}), corrupting that turn.
        # 3. A function-call turn never emits turn_complete until it gets a tool
        #    response (here: the interceptor's block message or the mock
        #    backend's result) -- and the model can spiral into a repeated-tool-call
        #    loop off a too-generic mock response, so cap both response rounds and
        #    total messages per turn as a safety valve. (Holds for BLOCKING
        #    function declarations; see the module docstring for gemini-3.8-live.)
        receiver = session.receive()
        tool_response_rounds = 0
        msg_count = 0

        await session.send_realtime_input(activity_start=types.ActivityStart())
        for i in range(0, len(pcm), CHUNK_BYTES):
            await session.send_realtime_input(
                audio=types.Blob(data=pcm[i:i + CHUNK_BYTES], mime_type="audio/pcm;rate=16000"))
        await session.send_realtime_input(activity_end=types.ActivityEnd())
        t_end_audio = time.perf_counter()

        def _first_output():
            if cur.latency_ms is None:
                cur.latency_ms = (time.perf_counter() - t_end_audio) * 1000

        async def _receive_one_turn():
            nonlocal tool_response_rounds, msg_count
            async for message in receiver:
                if not _is_stream_chunk(message):
                    msg_count += 1
                add_usage(cur.usage, getattr(message, "usage_metadata", None))
                if message.tool_call and message.tool_call.function_calls:
                    _first_output()
                    responses = []
                    for fc in message.tool_call.function_calls:
                        # Through the interceptor; a blocked call never reaches the backend.
                        resp = on_call(fc.name, dict(fc.args or {}))
                        responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=resp))
                    if tool_response_rounds < MAX_TOOL_RESPONSE_ROUNDS:
                        tool_response_rounds += 1
                        await session.send_tool_response(function_responses=responses)
                    else:
                        # safety valve: stop feeding a spiraling tool-call loop. Flagged on
                        # the turn (not an error); the case continues, as in the pilot.
                        cur.usage["safety_valve"] = "tool_rounds"
                        break
                sc = message.server_content
                if sc and sc.output_transcription and sc.output_transcription.text:
                    _first_output()
                    out.text += sc.output_transcription.text
                elif sc and _has_audio(sc):
                    _first_output()
                if sc and sc.turn_complete:
                    break
                if msg_count >= MAX_MESSAGES_PER_TURN:
                    cur.usage["safety_valve"] = "messages"
                    break

        await asyncio.wait_for(_receive_one_turn(), timeout=TURN_TIMEOUT_S)

    async def run(self, case: dict, interceptor, system_prompt: str):
        return await run_case(self, case, interceptor, system_prompt,
                              case_timeout_s=TURN_TIMEOUT_S * len(case["steps"]) + 30)
