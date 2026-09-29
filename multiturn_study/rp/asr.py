"""Cached transcripts of the turn WAVs.

  - harness ASR (gpt-4o-mini-transcribe): <wav>.asr.json -- what SAGE, the
    verifier and every text-reading agent see (`transcribe`).
  - Groq whisper-large-v3: <wav>.whisper.json -- only what the open cascade's
    LLM hears (`whisper_transcribe`); SAGE still reads the harness transcript.

Both caches are written atomically (temp file in the same folder, then
os.replace); a corrupt cache is re-transcribed. Warm before a run:

  python3 -m rp.asr --split {dev_v2,test}
  python3 -m rp.asr --whisper --split {dev_v2,test}
"""
from __future__ import annotations
import json, os, sys, tempfile
from pathlib import Path
from openai import OpenAI
from . import cases

ASR_MODEL = "gpt-4o-mini-transcribe"
WHISPER_MODEL = "whisper-large-v3"
GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
WHISPER_TIMEOUT_S = 60
_ASR_SUFFIX, _WHISPER_SUFFIX = ".asr.json", ".whisper.json"


def _call_api(wav: Path) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with open(wav, "rb") as f:
        r = client.audio.transcriptions.create(model=ASR_MODEL, file=f)
    return r.text


def _whisper_api(wav: Path) -> str:
    import requests
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from env_utils import load_env_file
    load_env_file()
    with open(wav, "rb") as f:
        r = requests.post(GROQ_TRANSCRIPTIONS_URL, timeout=WHISPER_TIMEOUT_S,
                          headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
                          files={"file": (Path(wav).name, f, "audio/wav")},
                          data={"model": WHISPER_MODEL, "response_format": "json", "temperature": "0"})
    if r.status_code != 200:
        # status + a short body excerpt (the provider's message); never the request headers
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["text"].strip()


def _read_cache(cache: Path) -> str | None:
    try:
        data = json.loads(cache.read_text())
    except (OSError, ValueError):
        return None
    return data["text"] if isinstance(data, dict) and "text" in data else None


def _cached(wav: Path, suffix: str, model: str, api) -> str:
    wav = Path(wav)
    cache = wav.with_suffix(suffix)
    text = _read_cache(cache)
    if text is not None:
        return text
    text = api(wav)
    with tempfile.NamedTemporaryFile(mode="w", dir=wav.parent, delete=False,
                                     suffix=".tmp", encoding="utf-8") as f:
        json.dump({"model": model, "text": text}, f)
        tmp_path = f.name
    try:
        os.replace(tmp_path, cache)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return json.loads(cache.read_text())["text"]


def has_cache(wav: Path) -> bool:
    """Check if a valid harness-ASR cache file exists for this wav file."""
    return _read_cache(Path(wav).with_suffix(_ASR_SUFFIX)) is not None


def has_whisper_cache(wav: Path) -> bool:
    """Check if a valid Groq Whisper cache file exists for this wav file."""
    return _read_cache(Path(wav).with_suffix(_WHISPER_SUFFIX)) is not None


def transcribe(wav: Path) -> str:
    """Harness transcript of a wav file, using the cache if valid, otherwise calling the API."""
    return _cached(wav, _ASR_SUFFIX, ASR_MODEL, lambda w: _call_api(w))


def whisper_transcribe(wav: Path) -> str:
    """Groq whisper-large-v3 transcript of a wav file (cached as <stem>.whisper.json)."""
    return _cached(wav, _WHISPER_SUFFIX, WHISPER_MODEL, lambda w: _whisper_api(w))


def warm(split: str, whisper: bool = False) -> dict:
    """Warm the harness (or, with whisper=True, the Groq Whisper) cache for a
    split; returns counts of transcribed and already-cached files."""
    check, fill = (has_whisper_cache, whisper_transcribe) if whisper else (has_cache, transcribe)
    transcribed = cached = 0
    for case in cases.load_split(split):
        for wav in sorted(case["_folder"].glob("input_turn*.wav")):
            if check(wav):
                cached += 1
            else:
                fill(wav)
                transcribed += 1
    return {"transcribed": transcribed, "cached": cached}


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog="python3 -m rp.asr")
    ap.add_argument("--split", required=True, help="dev_v2 or test")
    ap.add_argument("--whisper", action="store_true",
                    help="warm the Groq whisper-large-v3 caches (open cascade) instead of the harness ASR")
    args = ap.parse_args(argv)
    print(warm(args.split, whisper=args.whisper))


if __name__ == "__main__":
    main()
