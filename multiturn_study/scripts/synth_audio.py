#!/usr/bin/env python3
"""Synthesize missing turn audio for a split, with the pilot's TTS pipeline.
Run: python3 scripts/synth_audio.py --split test   (or dev_v2)"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "data" / "cases"))
sys.path.insert(0, str(HERE.parent.parent / "data")); sys.path.insert(0, str(HERE.parent))
from env_utils import load_env_file  # noqa: E402
load_env_file()
from _synthesize_audio import synthesize, NATURALNESS_PREFIX  # noqa: E402


def instructions(profile: dict, step: dict) -> str:
    return (f"{NATURALNESS_PREFIX} Speak as {profile['age_descriptor']} speaker with "
            f"{profile['accent_descriptor']} accent. {step.get('acting_notes', 'Speak naturally.')}")


def synth_folder(base: Path, manifest: dict) -> list:
    failed = []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and (p / "metadata.json").exists()):
        meta = json.loads((d / "metadata.json").read_text())
        prof = manifest.get(d.name) or manifest.get(meta["id"])
        for step in meta["steps"]:
            out = d / f"input_turn{step['turn']}.wav"
            if out.exists():
                continue
            ok, err = synthesize(step["user"], prof["voice"], instructions(prof, step), out)
            if not ok:
                failed.append((meta["id"], step["turn"], err))
    return failed


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--split", choices=["test", "dev_v2"], required=True)
    a = ap.parse_args()
    from rp.cases import SPLITS
    base = SPLITS[a.split][0]
    if a.split == "test":
        manifest = json.loads((base / "speaker_manifest.json").read_text())
    else:
        manifest = json.loads((base / "speaker_manifest.json").read_text())
    failed = synth_folder(base, manifest)
    print("failed:", failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
