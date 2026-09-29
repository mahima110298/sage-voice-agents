"""OpenAI Realtime speech-to-speech adapter (gpt-realtime-2.1 / -2.1-mini / -1.5).

Plugs into rp.agents.base.run_case exactly like the Gemini Live adapter:
  - tools = rp.tools.schemas_for(domain), for every config;
  - instructions = prompt + interceptor.prompt_suffix() + backend.session_context();
  - every function call goes through handle_call (the Interceptor) before the backend;
  - latency_ms = end of user audio (commit + response.create sent) -> first model
    output (audio transcript/text delta, audio delta, or a completed tool call);
  - usage = every response.done usage in the turn, summed (nested token-detail
    objects flattened to dotted keys, e.g. "output_token_details.audio_tokens");
  - MAX_TOOL_RESPONSE_ROUNDS = 4, TURN_TIMEOUT_S = 60; a tripped valve is flagged
    usage["safety_valve"] = "tool_rounds" (not an error) and the case continues.

Protocol (GA Realtime API; one websocket per case). Event names and shapes were
taken from the openai-python 2.46.0 types (openai/types/realtime/*, generated
from OpenAI's OpenAPI spec -- the platform.openai.com reference pages returned
403 to the fetcher on 2026-09-28). The model list there includes
"gpt-realtime-2.1", "gpt-realtime-2.1-mini" and "gpt-realtime-1.5".

  connect:  wss://api.openai.com/v1/realtime?model=<model>
            header  Authorization: Bearer $OPENAI_API_KEY   (no OpenAI-Beta header -> GA)
            server  -> {"type": "session.created", ...}
  client    {"type": "session.update", "session": {
                "type": "realtime",
                "instructions": <str>,
                "output_modalities": ["audio"],      # audio + its transcript
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                    "turn_detection": null}},   # manual turns
                "tools": [{"type": "function", "name", "description", "parameters"}],
                "tool_choice": "auto"}}
            server  -> {"type": "session.updated", "session": {...}}
  per turn  {"type": "input_audio_buffer.append", "audio": <base64 PCM16 24 kHz mono>}  (chunks)
            {"type": "input_audio_buffer.commit"}
            {"type": "response.create"}
  server    response.output_audio_transcript.delta {delta}   the model's own speech, as text
            response.output_audio.delta {delta}              audio (only used for latency)
            response.output_text.delta {delta}               (text modality; not used)
            response.function_call_arguments.done {call_id, name, arguments: JSON str, item_id}
            response.done {response: {status: completed|cancelled|incomplete|failed,
                           status_details: {type, reason, error}, output: [...],
                           usage: {input_tokens, output_tokens, total_tokens,
                                   input_token_details: {text_tokens, audio_tokens, image_tokens,
                                       cached_tokens, cached_tokens_details: {...}},
                                   output_token_details: {text_tokens, audio_tokens}}}}
            error {error: {type, code, message, param, event_id}}
  tool      {"type": "conversation.item.create",
             "item": {"type": "function_call_output", "call_id": <id>, "output": <JSON str>}}
            then {"type": "response.create"} again so the model answers the result.

Beta-era names (response.audio_transcript.delta, response.audio.delta,
response.text.delta) are also accepted by parse_event.

Turn loop: calls are answered (via handle_call) as they arrive and their
function_call_output items are sent once that response is done; a response.done
with calls answered triggers another response.create (at most
MAX_TOOL_RESPONSE_ROUNDS times per turn); a response.done with no pending calls
ends the turn. When the valve trips, the outputs are still added to the
conversation (so it stays well-formed for the next turn) but no further
response is requested. There is no message-count valve (Gemini's exists for its
Live message stream): the tool-round cap and the turn timeout bound a turn.
An `error` event or a response.done with status "failed" is a turn error; any
other non-"completed" status (incomplete, cancelled) is recorded as
usage["response_status"] and the turn continues normally. Latency stops only on
a non-empty transcript/text delta, an audio delta, or a completed tool call
(response.created / rate_limits.updated / empty deltas do not count).
connect() checks the session.updated payload: audio.input.turn_detection must
be null or absent, otherwise (server VAD on) the case fails with a session error.
"""
from __future__ import annotations

import asyncio
import audioop  # audioop-lts backport on Python 3.13
import base64
import json
import os
import sys
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from rp.agents.base import run_case

_STUDY = Path(__file__).resolve().parents[2]
_SCRIPTS = str(_STUDY / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from env_utils import load_env_file  # noqa: E402

URL = "wss://api.openai.com/v1/realtime?model="
SAMPLE_RATE = 24000
CHUNK_BYTES = 4800  # 100ms of 24kHz/16-bit mono PCM
TURN_TIMEOUT_S = 60
MAX_TOOL_RESPONSE_ROUNDS = 4  # same as the Gemini adapter
MAX_WS_MESSAGE_BYTES = 16 * 1024 * 1024

_TEXT = ("response.output_audio_transcript.delta", "response.audio_transcript.delta",
         "response.output_text.delta", "response.text.delta")
_AUDIO = ("response.output_audio.delta", "response.audio.delta")


def parse_event(ev: dict):
    """Normalize one server event (GA or beta name) to (kind, payload)."""
    t = ev.get("type", "")
    if t in _TEXT:
        return "text", {"delta": ev.get("delta", "")}
    if t in _AUDIO:
        return "audio", {}
    if t == "response.function_call_arguments.done":
        try:
            args = json.loads(ev.get("arguments") or "{}")
        except ValueError:
            args = {"_raw": ev.get("arguments")}
        return "call", {"call_id": ev["call_id"], "name": ev["name"], "args": args}
    if t == "response.done":
        r = ev.get("response") or {}
        return "done", {"usage": r.get("usage") or {}, "status": r.get("status"),
                        "status_details": str(r.get("status_details"))[:300]}
    if t == "error":
        return "error", {"message": str(ev.get("error"))[:300]}
    return "other", {}


def add_usage(acc: dict, usage: dict, prefix: str = "") -> None:
    """Sum one response's usage into the turn's totals; nested dicts become dotted keys."""
    for k, v in (usage or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            acc[prefix + k] = acc.get(prefix + k, 0) + v
        elif isinstance(v, dict):
            add_usage(acc, v, f"{prefix}{k}.")


def wav_to_pcm16(wav_path: Path, rate: int = SAMPLE_RATE) -> bytes:
    """Read a WAV (any rate / width / mono or stereo) -> raw PCM16 mono at `rate`."""
    with wave.open(str(wav_path), "rb") as wf:
        ch, width, fr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raw = audioop.lin2lin(raw, width, 2)
    if ch == 2:
        raw = audioop.tomono(raw, 2, 0.5, 0.5)
    elif ch != 1:
        raise ValueError(f"{wav_path}: unsupported channel count {ch}")
    if fr != rate:
        raw, _ = audioop.ratecv(raw, 2, 1, fr, rate, None)
    return raw


def _ws_connect(url: str, headers: dict):
    import websockets
    return websockets.connect(url, additional_headers=headers, max_size=MAX_WS_MESSAGE_BYTES)


class _Session:
    def __init__(self, ws):
        self.ws = ws

    async def send(self, ev: dict) -> None:
        await self.ws.send(json.dumps(ev))

    async def recv(self) -> dict:
        return json.loads(await self.ws.recv())

    async def wait_for(self, type_: str) -> dict:
        while True:
            ev = await self.recv()
            if ev.get("type") == type_:
                return ev
            if ev.get("type") == "error":
                raise RuntimeError(f"realtime error: {str(ev.get('error'))[:300]}")


class OpenAIRealtimeAdapter:
    def __init__(self, model: str, ws_connect=None, api_key: str | None = None):
        self.model = model
        self._ws_connect = ws_connect or _ws_connect
        self._api_key = api_key

    def _key(self) -> str:
        if self._api_key is None:
            load_env_file()
            self._api_key = os.environ["OPENAI_API_KEY"]
        return self._api_key

    @staticmethod
    def session_config(tools: list[dict], system_instruction: str) -> dict:
        return {
            "type": "realtime",
            "instructions": system_instruction,
            "output_modalities": ["audio"],
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                                "turn_detection": None}},
            "tools": [{"type": "function", "name": t["name"], "description": t["description"],
                       "parameters": t["parameters"]} for t in tools],
            "tool_choice": "auto",
        }

    @asynccontextmanager
    async def connect(self, case, tools, system_instruction):
        headers = {"Authorization": f"Bearer {self._key()}"}
        async with self._ws_connect(URL + self.model, headers) as ws:
            s = _Session(ws)
            await asyncio.wait_for(s.wait_for("session.created"), timeout=TURN_TIMEOUT_S)
            await s.send({"type": "session.update",
                          "session": self.session_config(tools, system_instruction)})
            upd = await asyncio.wait_for(s.wait_for("session.updated"), timeout=TURN_TIMEOUT_S)
            td = (((upd.get("session") or {}).get("audio") or {}).get("input") or {}).get("turn_detection")
            if td is not None:  # absent or null = off; anything else means server VAD is on
                raise RuntimeError(f"session.updated reports turn_detection={str(td)[:100]}; "
                                   "manual turns need it disabled (null)")
            yield s

    async def run_turn(self, session, wav, on_call, state, out) -> None:
        pcm = wav_to_pcm16(wav, SAMPLE_RATE)
        cur = state.current
        for i in range(0, len(pcm), CHUNK_BYTES):
            await session.send({"type": "input_audio_buffer.append",
                                "audio": base64.b64encode(pcm[i:i + CHUNK_BYTES]).decode("ascii")})
        await session.send({"type": "input_audio_buffer.commit"})
        await session.send({"type": "response.create"})
        t_end_audio = time.perf_counter()

        def _first_output():
            if cur.latency_ms is None:
                cur.latency_ms = (time.perf_counter() - t_end_audio) * 1000

        async def _receive_one_turn():
            rounds = 0
            pending: list[tuple[str, dict]] = []
            new_response = False
            while True:
                kind, v = parse_event(await session.recv())
                if kind == "text":
                    if not v["delta"]:
                        continue  # an empty delta is not model output (same as Gemini)
                    _first_output()
                    if new_response and out.text and not out.text[-1].isspace():
                        out.text += " "  # separate speech from consecutive responses
                    new_response = False
                    out.text += v["delta"]
                elif kind == "audio":
                    _first_output()
                elif kind == "call":
                    _first_output()
                    # Through the interceptor; a blocked call never reaches the backend.
                    pending.append((v["call_id"], on_call(v["name"], v["args"])))
                elif kind == "error":
                    raise RuntimeError(f"realtime error: {v['message']}")
                elif kind == "done":
                    add_usage(cur.usage, v["usage"])
                    if v["status"] == "failed":
                        raise RuntimeError(f"response failed: {v['status_details']}")
                    if v["status"] and v["status"] != "completed":
                        cur.usage["response_status"] = v["status"]  # e.g. incomplete / cancelled
                    if not pending:
                        return
                    for call_id, resp in pending:
                        await session.send({"type": "conversation.item.create", "item": {
                            "type": "function_call_output", "call_id": call_id,
                            "output": json.dumps(resp)}})
                    pending = []
                    if rounds >= MAX_TOOL_RESPONSE_ROUNDS:
                        # safety valve: stop feeding a spiraling tool-call loop. Flagged on
                        # the turn (not an error); the case continues, as with Gemini.
                        cur.usage["safety_valve"] = "tool_rounds"
                        return
                    rounds += 1
                    new_response = True
                    await session.send({"type": "response.create"})

        await asyncio.wait_for(_receive_one_turn(), timeout=TURN_TIMEOUT_S)

    async def run(self, case: dict, interceptor, system_prompt: str):
        return await run_case(self, case, interceptor, system_prompt,
                              case_timeout_s=TURN_TIMEOUT_S * len(case["steps"]) + 30)
