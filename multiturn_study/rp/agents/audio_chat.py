"""Audio-in chat adapter: the turn's WAV goes straight into an OpenRouter chat
completion (non-streaming) as an `input_audio` content part, and the model answers
in text with tool calls. The third architecture under study (spec section 5,
"audio-in chat": qwen3.8-omni-flash, voxtral-small-24b; nemotron-3-nano-omni was dropped),
next to live speech-to-speech (Gemini Live, OpenAI Realtime) and the cascade.

Same shared loop (rp.agents.base.run_case) and the same completion/tool loop as
the cascade (rp.agents.cascade.chat_loop): OpenAI-style function tools from
rp.tools.schemas_for(domain); every call through handle_call (a blocked call's
tool message is the Decision's message); assistant message with tool_calls
appended before its tool messages; MAX_TOOL_RESPONSE_ROUNDS = 4 with the
usage["safety_valve"] = "tool_rounds" valve (not an error); agent_text = every
assistant content piece of the turn joined by a single space; latency_ms = turn
start -> first completion's return (non-streaming, so not comparable with the
live agents' first-audio latency); usage summed per turn with nested dicts as
dotted keys -- OpenRouter reports prompt_tokens_details.audio_tokens (Voxtral
reports its audio there; Qwen folds audio into prompt_tokens and reports 0);
rp.runner.price_usage bills those at the pricing.json audio rate.

User turn = [{"type": "text", "text": AUDIO_TURN_TEXT},
             {"type": "input_audio", "input_audio": {"data": <base64 WAV>, "format": "wav"}}].
The agent never receives the transcript; state.current.user_text (what SAGE and
the verifier read) is still the harness ASR, set by run_case.

Confirmed 2026-09-28:
  - OpenRouter docs, https://openrouter.ai/docs/features/multimodal/audio: the
    content part is {"type": "input_audio", "input_audio": {"data", "format"}};
    audio must be base64 (no URLs); formats include wav, mp3, flac, ogg, m4a,
    aac, aiff, pcm16, pcm24; per-model format support varies; the page states no
    limit on the number of audio parts.
  - Probe on dev_v2 ecommerce_mt_04 (24 kHz mono 16-bit WAV): a two-turn message
    list with BOTH turns as audio parts is accepted by qwen/qwen3.8-omni-flash
    (Alibaba) and mistralai/voxtral-small-24b-2507 (Mistral; audio_tokens rose
    7 -> 12), so earlier user turns are kept as audio parts (no transcript
    fallback is needed or implemented).
  - nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free (Nvidia, only endpoint)
    returns 200 but ignores the audio (prompt_tokens 22 for a text+audio message
    asking for a transcript; "I can't hear the audio"), for 24 kHz and 16 kHz WAV.
    Dropped from the study (spec section 5; task-16b-report.md).
"""
from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path

from rp.agents.base import run_case
from rp.agents.cascade import (MAX_TOOL_RESPONSE_ROUNDS, TURN_TIMEOUT_S, ChatClient, OPENROUTER,  # noqa: F401
                               _Session, chat_loop)

AUDIO_TURN_TEXT = "(The customer's spoken turn is attached as audio.)"


def audio_user_message(wav: Path) -> dict:
    data = base64.b64encode(Path(wav).read_bytes()).decode("ascii")
    return {"role": "user", "content": [
        {"type": "text", "text": AUDIO_TURN_TEXT},
        {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
    ]}


class AudioChatAdapter:
    def __init__(self, model: str, client=None):
        self.model = model
        self.client = client or ChatClient(OPENROUTER)

    @asynccontextmanager
    async def connect(self, case, tools, system_instruction):
        payload_tools = [{"type": "function", "function": t} for t in tools]
        yield _Session([{"role": "system", "content": system_instruction}], payload_tools)

    async def run_turn(self, session: _Session, wav, on_call, state, out) -> None:
        session.messages.append(audio_user_message(wav))
        await chat_loop(self.client, self.model, session, state.current, on_call, out)

    async def run(self, case: dict, interceptor, system_prompt: str):
        return await run_case(self, case, interceptor, system_prompt,
                              case_timeout_s=TURN_TIMEOUT_S * len(case["steps"]) + 30)
