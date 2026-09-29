#!/usr/bin/env python3
"""Synthesize one wav per turn for every multi-turn case in this folder,
reusing the exact pipeline from data/multiturn_pilot_cases/_synthesize_audio.py
(gpt-4o-mini-tts -- the cheap OpenAI TTS tier -- with a naturalness prefix and
a per-case speaker profile shared across all turns in that case, since it's
the same person speaking throughout the conversation).

Requires OPENAI_API_KEY set in the environment (loaded from .env in this
project's directories via python-dotenv if you run this from a notebook that
already loaded it; standalone, export it or prefix the command with it).

Run:  OPENAI_API_KEY=sk-... python3 _synthesize_audio.py
"""
import json
import os
import struct
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_DATA_DIR = HERE.parent.parent.parent / "data"
sys.path.insert(0, str(MAIN_DATA_DIR))
from speaker_profiles import assign_profiles  # noqa: E402

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini-tts"

NATURALNESS_PREFIX = (
    "Speak naturally and conversationally, the way a real person talks on a phone "
    "call — with the natural rhythm, breathing, and pitch variation of real speech. "
    "Avoid a flat, monotone, or robotic delivery."
)


def fix_streaming_wav_header(data: bytes) -> bytes:
    data = bytearray(data)
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return bytes(data)
    i = 12
    while i < len(data) - 8:
        cid = bytes(data[i:i + 4])
        csize = struct.unpack("<I", data[i + 4:i + 8])[0]
        if cid == b"data" and csize == 0xFFFFFFFF:
            real_size = len(data) - (i + 8)
            data[i + 4:i + 8] = struct.pack("<I", real_size)
            data[4:8] = struct.pack("<I", len(data) - 8)
            break
        if csize == 0xFFFFFFFF:
            break
        i += 8 + csize + (csize % 2)
    return bytes(data)


def synthesize(text: str, voice: str, instructions: str, out_path: Path, max_retries=3):
    payload = json.dumps({
        "model": MODEL,
        "input": text,
        "voice": voice,
        "instructions": instructions,
        "response_format": "wav",
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/audio/speech",
        data=payload,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                out_path.write_bytes(fix_streaming_wav_header(data))
                return True, None
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode()[:300]}"
            if e.code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            break
        except Exception as e:
            last_err = str(e)[:300]
            time.sleep(2)
    return False, last_err


def main():
    case_dirs = sorted(p for p in HERE.iterdir() if p.is_dir())
    case_ids = [d.name for d in case_dirs]
    manifest = assign_profiles(case_ids)
    (HERE / "speaker_manifest.json").write_text(json.dumps(manifest, indent=2))

    ok, skipped, failed = 0, 0, []
    total_turns = sum(len(json.loads((d / "metadata.json").read_text())["steps"]) for d in case_dirs)
    done = 0

    for d in case_dirs:
        case_id = d.name
        meta = json.loads((d / "metadata.json").read_text())
        profile = manifest[case_id]

        for step in meta["steps"]:
            turn = step["turn"]
            out_path = d / f"input_turn{turn}.wav"
            if out_path.exists():
                done += 1
                skipped += 1
                print(f"[{done}/{total_turns}] SKIP {case_id} turn {turn}  (already synthesized)")
                continue
            instructions = (
                f"{NATURALNESS_PREFIX} Speak as {profile['age_descriptor']} speaker with "
                f"{profile['accent_descriptor']} accent. {step.get('acting_notes', 'Speak naturally.')}"
            )
            success, err = synthesize(step["user"], profile["voice"], instructions, out_path)
            done += 1
            if success:
                ok += 1
                print(f"[{done}/{total_turns}] OK   {case_id} turn {turn}  "
                      f"({profile['voice']}, {profile['age_bracket']}, {profile['accent']})")
            else:
                failed.append((case_id, turn, err))
                print(f"[{done}/{total_turns}] FAIL {case_id} turn {turn}: {err}")

    print(f"\nSynthesized {ok} new turn audio files ({skipped} already existed, "
          f"{total_turns} total) across {len(case_dirs)} cases")
    if failed:
        print(f"{len(failed)} failures:")
        for cid, turn, err in failed:
            print(f"  {cid} turn {turn}: {err}")


if __name__ == "__main__":
    main()
