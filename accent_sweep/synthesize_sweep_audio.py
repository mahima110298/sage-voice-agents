#!/usr/bin/env python3
"""Resynthesize every severe_failure_cases case's user utterance in all 12
accents from data/speaker_profiles.ACCENTS, holding voice and age bracket
fixed at the case's existing speaker_manifest.json assignment — so accent is
the only thing that varies across the 12 versions of a given case.

This is the "12-accent sweep": 139 cases x 12 accents = 1,668 audio files,
letting the same task be scored under 12 different accents instead of the
single accent each case currently has baked in.

Requires OPENAI_API_KEY (same gpt-4o-mini-tts pipeline as the rest of the
project's audio).

Run:  OPENAI_API_KEY=sk-... python3 synthesize_sweep_audio.py
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "data"
# Full 139-case set: 39-case training set + 100-case held-out set.
CASE_SUBDIRS = ["severe_failure_cases", "test_cases"]
OUT_DIR = HERE / "sweep_audio"

sys.path.insert(0, str(REPO_ROOT / "data"))
from speaker_profiles import ACCENTS  # noqa: E402

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


def synthesize(text: str, voice: str, instructions: str, out_path: Path, max_retries: int = 3):
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


def load_case_dirs() -> dict[str, Path]:
    """case_id -> case folder path (folders are '<case_id>_<hash>'), across both
    the training set and the held-out set."""
    dirs = {}
    for subdir in CASE_SUBDIRS:
        for p in (DATA_DIR / subdir).iterdir():
            if p.is_dir():
                case_id = p.name.rsplit("_", 1)[0]
                assert case_id not in dirs, f"duplicate case_id {case_id} across subdirs"
                dirs[case_id] = p
    return dirs


def load_manifest() -> dict:
    """speaker_manifest.json is written per-subdir; merge them (case IDs are
    unique across the combined pool, per speaker_profiles.py's design)."""
    manifest = {}
    for subdir in CASE_SUBDIRS:
        manifest.update(json.loads((DATA_DIR / subdir / "speaker_manifest.json").read_text()))
    return manifest


def main():
    case_dirs = load_case_dirs()
    manifest = load_manifest()

    accent_keys = sorted(ACCENTS.keys())  # stable order across runs
    total = len(case_dirs) * len(accent_keys)
    done = ok = skipped = 0
    failed = []

    print(f"{len(case_dirs)} cases x {len(accent_keys)} accents = {total} audio files")

    for case_id, case_dir in sorted(case_dirs.items()):
        meta = json.loads((case_dir / "metadata.json").read_text())
        user_text = meta["dialogue"][0]["user"]
        profile = manifest[case_id]  # existing voice + age assignment, held fixed

        case_out_dir = OUT_DIR / case_id
        case_out_dir.mkdir(parents=True, exist_ok=True)

        for accent_key in accent_keys:
            done += 1
            out_path = case_out_dir / f"{accent_key}.wav"
            if out_path.exists():
                skipped += 1
                print(f"[{done}/{total}] SKIP {case_id} / {accent_key} (already synthesized)")
                continue

            instructions = (
                f"{NATURALNESS_PREFIX} Speak as {profile['age_descriptor']} speaker with "
                f"{ACCENTS[accent_key]} accent. "
                f"{meta.get('acting_notes', 'Speak naturally.')}"
            )
            success, err = synthesize(user_text, profile["voice"], instructions, out_path)
            if success:
                ok += 1
                print(f"[{done}/{total}] OK   {case_id} / {accent_key}  (voice={profile['voice']}, age={profile['age_bracket']})")
            else:
                failed.append((case_id, accent_key, err))
                print(f"[{done}/{total}] FAIL {case_id} / {accent_key}: {err}")

    # Write a flat index the next stage (run_sweep.py) reads to know exactly
    # which (case_id, accent) audio files exist and where.
    index = {
        f"{cid}/{acc}": str((OUT_DIR / cid / f"{acc}.wav").relative_to(REPO_ROOT))
        for cid in case_dirs
        for acc in accent_keys
        if (OUT_DIR / cid / f"{acc}.wav").exists()
    }
    (OUT_DIR / "sweep_index.json").write_text(json.dumps(index, indent=2))

    print(f"\nSynthesized {ok} new files ({skipped} already existed, {total} total)")
    print(f"Index written: {OUT_DIR / 'sweep_index.json'} ({len(index)} entries)")
    if failed:
        print(f"\n{len(failed)} failures:")
        for cid, acc, err in failed:
            print(f"  {cid} / {acc}: {err}")


if __name__ == "__main__":
    main()
