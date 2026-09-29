import pytest
from rp import asr, cases
import json


def test_cache(tmp_path, monkeypatch):
    wav = tmp_path / "input_turn1.wav"; wav.write_bytes(b"RIFF")
    n = {"calls": 0}
    def fake(path):
        n["calls"] += 1; return "send two hundred dollars"
    monkeypatch.setattr(asr, "_call_api", fake)
    assert asr.transcribe(wav) == "send two hundred dollars"
    assert asr.transcribe(wav) == "send two hundred dollars"
    assert n["calls"] == 1


def test_corrupt_cache_is_retranscribed(tmp_path, monkeypatch):
    wav = tmp_path / "input_turn1.wav"; wav.write_bytes(b"RIFF")
    cache = wav.with_suffix(".asr.json")

    # Write corrupt JSON to cache
    cache.write_text("{not json")

    n = {"calls": 0}
    def fake(path):
        n["calls"] += 1; return "recovered text"
    monkeypatch.setattr(asr, "_call_api", fake)

    # Should re-transcribe and return the new text
    result = asr.transcribe(wav)
    assert result == "recovered text"
    assert n["calls"] == 1

    # Cache should now be valid JSON
    cached = json.loads(cache.read_text())
    assert cached["text"] == "recovered text"
    assert cached["model"] == asr.ASR_MODEL


def test_no_temp_files_left(tmp_path, monkeypatch):
    wav = tmp_path / "input_turn1.wav"; wav.write_bytes(b"RIFF")
    n = {"calls": 0}
    def fake(path):
        n["calls"] += 1; return "some text"
    monkeypatch.setattr(asr, "_call_api", fake)

    # Transcribe once
    asr.transcribe(wav)

    # No temp files should be left
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Found temp files: {tmp_files}"


def test_warm_counts(tmp_path, monkeypatch):
    # Create fake cases with wav files
    case1_folder = tmp_path / "case1"; case1_folder.mkdir()
    case2_folder = tmp_path / "case2"; case2_folder.mkdir()

    wav1_t1 = case1_folder / "input_turn1.wav"; wav1_t1.write_bytes(b"RIFF")
    wav1_t2 = case1_folder / "input_turn2.wav"; wav1_t2.write_bytes(b"RIFF")
    wav2_t1 = case2_folder / "input_turn1.wav"; wav2_t1.write_bytes(b"RIFF")

    fake_cases = [
        {"_folder": case1_folder},
        {"_folder": case2_folder},
    ]

    n = {"calls": 0}
    def fake(path):
        n["calls"] += 1; return f"text_{n['calls']}"
    monkeypatch.setattr(asr, "_call_api", fake)
    monkeypatch.setattr(cases, "load_split", lambda split: fake_cases)

    # First warm: should transcribe all 3 files
    result1 = asr.warm("test")
    assert result1["transcribed"] == 3
    assert result1["cached"] == 0
    assert n["calls"] == 3

    # Second warm: should use cache
    result2 = asr.warm("test")
    assert result2["transcribed"] == 0
    assert result2["cached"] == 3
    assert n["calls"] == 3  # No new calls


def test_has_cache(tmp_path):
    wav = tmp_path / "input_turn1.wav"; wav.write_bytes(b"RIFF")
    cache = wav.with_suffix(".asr.json")

    # No cache initially
    assert asr.has_cache(wav) is False

    # Write valid cache
    cache.write_text(json.dumps({"model": asr.ASR_MODEL, "text": "some text"}))
    assert asr.has_cache(wav) is True

    # Corrupt cache should return False
    cache.write_text("{not json")
    assert asr.has_cache(wav) is False


# ---------------------------------------------------------------- Task 16b fix round: Groq Whisper cache

def test_whisper_cache_is_atomic_and_reused(tmp_path, monkeypatch):
    wav = tmp_path / "input_turn2.wav"; wav.write_bytes(b"RIFF")
    n = {"calls": 0}
    def fake(path):
        n["calls"] += 1; return "cancel it"
    monkeypatch.setattr(asr, "_whisper_api", fake)
    assert not asr.has_whisper_cache(wav)
    assert asr.whisper_transcribe(wav) == "cancel it"
    assert asr.whisper_transcribe(wav) == "cancel it"
    assert n["calls"] == 1 and asr.has_whisper_cache(wav)
    cache = tmp_path / "input_turn2.whisper.json"
    assert json.loads(cache.read_text()) == {"model": "whisper-large-v3", "text": "cancel it"}
    assert not list(tmp_path.glob("*.tmp"))
    assert not wav.with_suffix(".asr.json").exists()        # separate from the harness cache
    cache.write_text("{corrupt")
    assert not asr.has_whisper_cache(wav)
    assert asr.whisper_transcribe(wav) == "cancel it" and n["calls"] == 2


def test_warm_whisper_counts(tmp_path, monkeypatch):
    folder = tmp_path / "case1"; folder.mkdir()
    for t in (1, 2):
        (folder / f"input_turn{t}.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(cases, "load_split", lambda split: [{"_folder": folder}])
    monkeypatch.setattr(asr, "_call_api", lambda p: pytest.fail("harness ASR must not be called"))
    monkeypatch.setattr(asr, "_whisper_api", lambda p: "w")
    assert asr.warm("dev_v2", whisper=True) == {"transcribed": 2, "cached": 0}
    assert asr.warm("dev_v2", whisper=True) == {"transcribed": 0, "cached": 2}


def test_cli_whisper_flag(tmp_path, monkeypatch, capsys):
    seen = {}
    monkeypatch.setattr(asr, "warm", lambda split, whisper=False: seen.update(split=split, whisper=whisper) or {})
    asr.main(["--whisper", "--split", "dev_v2"])
    assert seen == {"split": "dev_v2", "whisper": True}
    asr.main(["--split", "test"])
    assert seen == {"split": "test", "whisper": False}
    with pytest.raises(SystemExit):
        asr.main(["--whisper"])
