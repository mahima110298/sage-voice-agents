"""Helpers for feeding this project's existing tool schemas and case audio
into the Gemini Live API: converting OpenAI-style function schemas to
Gemini's FunctionDeclaration format, and converting the project's WAV files
(gpt-4o-mini-tts output) to the raw 16kHz mono PCM16 the Live API expects for
realtime audio input.
"""
from __future__ import annotations

import audioop  # stdlib; deprecated since 3.11 but still present through 3.12/3.13
import wave
from pathlib import Path

LIVE_INPUT_SAMPLE_RATE = 16000  # Gemini Live's expected input rate


def openai_tool_to_gemini_function_declaration(tool: dict) -> dict:
    """Convert one TOOL_SCHEMAS[...] entry (OpenAI `{"type": "function",
    "function": {...}}` shape) into a Gemini FunctionDeclaration dict. Gemini's
    google-genai SDK accepts plain dicts for `types.FunctionDeclaration(**d)`,
    and its schema fields are (for current SDK versions) a JSON-Schema-like
    subset compatible with what's already in TOOL_SCHEMAS — no property-level
    translation needed, just dropping the outer OpenAI envelope.
    """
    fn = tool["function"]
    return {
        "name": fn["name"],
        "description": fn["description"],
        "parameters": fn["parameters"],
    }


def domain_tools_to_gemini_tool(tool_schemas: dict, domain_tool_names: list[str]):
    """Build a single google.genai.types.Tool covering every tool available
    in one case's domain. Import is deferred so this module (and the parts of
    the pipeline that only need the WAV conversion) can be imported/tested
    without google-genai installed.
    """
    from google.genai import types  # noqa: PLC0415

    decls = [
        types.FunctionDeclaration(**openai_tool_to_gemini_function_declaration(tool_schemas[name]))
        for name in domain_tool_names
    ]
    return types.Tool(function_declarations=decls)


def wav_to_pcm16_16k(wav_path: Path) -> bytes:
    """Read a WAV file (any sample rate/channel count/sample width — OpenAI's
    gpt-4o-mini-tts WAV output is what this project produces) and return raw
    PCM16 mono bytes at 16kHz, the format Gemini Live's send_realtime_input
    expects for audio/pcm input.
    """
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        # audioop.lin2lin handles 1/2/4-byte widths; normalize to 16-bit first.
        raw = audioop.lin2lin(raw, sample_width, 2)
        sample_width = 2

    if n_channels == 2:
        raw = audioop.tomono(raw, sample_width, 0.5, 0.5)
        n_channels = 1
    elif n_channels != 1:
        raise ValueError(f"{wav_path}: unsupported channel count {n_channels}")

    if frame_rate != LIVE_INPUT_SAMPLE_RATE:
        raw, _ = audioop.ratecv(raw, sample_width, n_channels, frame_rate, LIVE_INPUT_SAMPLE_RATE, None)

    return raw
