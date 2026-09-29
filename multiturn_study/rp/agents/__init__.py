"""Agent adapters. `AGENTS` maps each agent name under study to a zero-arg
factory returning an adapter with `async run(case, interceptor, system_prompt)
-> list[TurnRecord]`.
"""
from __future__ import annotations


def _gemini(model: str):
    def make():
        from rp.agents.gemini_live import GeminiLiveAdapter
        return GeminiLiveAdapter(model)
    return make


def _openai(model: str):
    def make():
        from rp.agents.openai_realtime import OpenAIRealtimeAdapter
        return OpenAIRealtimeAdapter(model)
    return make


def _cascade():
    from rp.agents.cascade import CascadeAdapter
    return CascadeAdapter()


def _audio_chat(model: str):
    def make():
        from rp.agents.audio_chat import AudioChatAdapter
        return AudioChatAdapter(model)
    return make


def _open_cascade():
    from rp.agents.cascade import GROQ, CascadeAdapter
    return CascadeAdapter("openai/gpt-oss-120b", asr="groq-whisper", llm_base_url=GROQ)


AGENTS = {
    "gemini-3.8-live": _gemini("gemini-3.8-live"),
    "gemini-3.1-flash-live-preview": _gemini("gemini-3.1-flash-live-preview"),
    "gemini-2.5-flash-native-audio-latest": _gemini("gemini-2.5-flash-native-audio-latest"),
    "gpt-realtime-2.1": _openai("gpt-realtime-2.1"),
    "gpt-realtime-2.1-mini": _openai("gpt-realtime-2.1-mini"),
    "gpt-realtime-1.5": _openai("gpt-realtime-1.5"),
    "cascade": _cascade,
    # open weights (Task 16b; spec section 5). nemotron-3-nano-omni (:free) was dropped:
    # it ignored input_audio in its smoke (task-16b-report.md).
    "qwen3.8-omni-flash": _audio_chat("qwen/qwen3.8-omni-flash"),
    "voxtral-small-24b": _audio_chat("mistralai/voxtral-small-24b-2507"),
    "open-cascade": _open_cascade,
}
