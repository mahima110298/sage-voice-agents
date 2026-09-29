"""Warm the harness ASR and Groq Whisper caches for only the locked-test-set
80-case subset (`data/test_subset_v3.json`), not the full 119-case split.

`rp.asr`'s warm CLI (`python3 -m rp.asr --split test [--whisper]`) warms every
case in a split, which would spend the ASR/Whisper budget on the 39 cases this
task doesn't need. This script reuses the exact same cached functions
(`rp.asr.transcribe` / `rp.asr.whisper_transcribe`, same cache files, same
atomic-write format -- rp/asr.py is not modified) but restricts the case set
to the subset file.

Usage:
    python3 -m scripts.warm_subset_cache            # harness ASR only
    python3 -m scripts.warm_subset_cache --whisper   # Groq Whisper only
    python3 -m scripts.warm_subset_cache --both      # both caches
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rp import asr, cases as rp_cases

# Groq's whisper-large-v3 on-demand tier is rate-limited (observed: 20
# requests/minute). Pace whisper calls and retry on 429 rather than treating
# a rate limit as a hard failure -- rp/asr.py's cache format/API is untouched.
_WHISPER_PACING_S = 3.2
_WHISPER_MAX_RETRIES = 6

STUDY_DIR = Path(__file__).resolve().parent.parent
SUBSET_PATH = STUDY_DIR / "data" / "test_subset_v3.json"


def load_subset_cases() -> list[dict]:
    subset_ids = set(json.loads(SUBSET_PATH.read_text())["case_ids"])
    all_cases = rp_cases.load_split("test")
    selected = [c for c in all_cases if c["id"] in subset_ids]
    missing = subset_ids - {c["id"] for c in selected}
    if missing:
        raise SystemExit(f"subset case ids not found in split 'test': {sorted(missing)}")
    return selected


def _fill_with_retry(fill, wav, whisper: bool) -> None:
    """Call `fill(wav)`; for whisper, retry with backoff on a 429 (rate limit)
    rather than failing immediately."""
    if not whisper:
        fill(wav)
        return
    last = None
    for attempt in range(_WHISPER_MAX_RETRIES):
        try:
            fill(wav)
            return
        except Exception as e:  # noqa: BLE001 -- retry only rate limits
            last = e
            if "429" not in str(e):
                raise
            time.sleep(_WHISPER_PACING_S * (attempt + 2))
    raise last


def warm_subset(whisper: bool) -> dict:
    """Same cache semantics as rp.asr.warm, restricted to the subset's cases."""
    check, fill = (asr.has_whisper_cache, asr.whisper_transcribe) if whisper else (asr.has_cache, asr.transcribe)
    transcribed = cached = failed = 0
    failures: list[str] = []
    for case in load_subset_cases():
        for wav in sorted(case["_folder"].glob("input_turn*.wav")):
            if check(wav):
                cached += 1
                continue
            try:
                _fill_with_retry(fill, wav, whisper)
                transcribed += 1
            except Exception as e:  # noqa: BLE001 -- report and keep going
                failed += 1
                failures.append(f"{wav}: {e}")
            if whisper:
                time.sleep(_WHISPER_PACING_S)
    return {"transcribed": transcribed, "cached": cached, "failed": failed, "failures": failures}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python3 -m scripts.warm_subset_cache")
    ap.add_argument("--whisper", action="store_true", help="warm the Groq Whisper cache only")
    ap.add_argument("--both", action="store_true", help="warm both the harness ASR and Whisper caches")
    args = ap.parse_args(argv)

    if args.both:
        print("harness ASR:", warm_subset(whisper=False))
        print("groq whisper:", warm_subset(whisper=True))
    else:
        print(warm_subset(whisper=args.whisper))


if __name__ == "__main__":
    main()
