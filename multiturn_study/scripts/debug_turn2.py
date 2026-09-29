#!/usr/bin/env python3
"""Diagnostic: stream finance_mt_01 turns 1 and 2 into one Gemini Live
session, printing every raw message received during EACH turn (not just the
fields we currently parse), to find out why turn 2+ come back empty."""
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "accent_sweep"))
sys.path.insert(0, str(HERE))

from env_utils import load_env_file  # noqa: E402
load_env_file()
import os  # noqa: E402

from harness_defs import SUT_SYSTEM, TOOL_SCHEMAS_REALTIME, DOMAIN_TOOLS  # noqa: E402
from gemini_tools import wav_to_pcm16_16k  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

CASE_DIR = HERE.parent.parent / "data" / "multiturn_pilot_cases" / "finance_mt_01"
CHUNK_BYTES = 3200
SUT_MODEL = "gemini-3.1-flash-live-preview"


async def main():
    meta = json.loads((CASE_DIR / "metadata.json").read_text())
    tool_names = DOMAIN_TOOLS.get(meta["domain"], [])
    tools = [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=TOOL_SCHEMAS_REALTIME[n]["name"],
            description=TOOL_SCHEMAS_REALTIME[n]["description"],
            parameters=TOOL_SCHEMAS_REALTIME[n]["parameters"],
        ) for n in tool_names if n in TOOL_SCHEMAS_REALTIME
    ])]
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=SUT_SYSTEM,
        tools=tools,
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True),
        ),
    )

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    async with client.aio.live.connect(model=SUT_MODEL, config=config) as session:
        for turn in (1, 2, 3, 4):
            receiver = session.receive()  # fresh per turn -- it closes itself after turn_complete
            print(f"\n{'='*20} SENDING TURN {turn} {'='*20}")
            wav_path = CASE_DIR / f"input_turn{turn}.wav"
            pcm = wav_to_pcm16_16k(wav_path)
            await session.send_realtime_input(activity_start=types.ActivityStart())
            for i in range(0, len(pcm), CHUNK_BYTES):
                await session.send_realtime_input(audio=types.Blob(data=pcm[i:i+CHUNK_BYTES], mime_type="audio/pcm;rate=16000"))
            await session.send_realtime_input(activity_end=types.ActivityEnd())
            print(f"  sent {len(pcm)} bytes of audio for turn {turn}, now receiving...")

            msg_count = 0
            async def _drain():
                nonlocal msg_count
                async for message in receiver:
                    msg_count += 1
                    sc = message.server_content
                    d = message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else str(message)
                    s = str(d)
                    if len(s) > 300:
                        s = s[:150] + " ...<TRUNCATED>... " + s[-100:]
                    print(f"  [msg {msg_count}] {s}")
                    if message.tool_call and message.tool_call.function_calls:
                        print(f"    -> FUNCTION CALL: {[(fc.name, dict(fc.args or {})) for fc in message.tool_call.function_calls]}")
                        # The model won't emit turn_complete until it gets a tool response back
                        # (we're a passive observer with no live backend -- send a generic mock
                        # result so the turn can close cleanly before the next turn's audio).
                        await session.send_tool_response(function_responses=[
                            types.FunctionResponse(id=fc.id, name=fc.name, response={"result": "ok"})
                            for fc in message.tool_call.function_calls
                        ])
                    if sc and sc.turn_complete:  # wait for the REAL server-side turn closure
                        break

            try:
                await asyncio.wait_for(_drain(), timeout=30)
            except asyncio.TimeoutError:
                print(f"  !! TIMEOUT waiting for turn {turn} response after {msg_count} messages")
            print(f"  turn {turn}: received {msg_count} messages total")


if __name__ == "__main__":
    asyncio.run(main())
