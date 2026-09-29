import importlib
import sys


def test_skips_existing_and_builds_instructions(tmp_path, monkeypatch):
    # Hermetic: _synthesize_audio.py reads OPENAI_API_KEY at import time, so
    # a dummy key must be set before the (possibly first) import, and any
    # cached module dropped so a prior test run's real import doesn't linger.
    monkeypatch.setenv("OPENAI_API_KEY", "test-dummy")
    sys.modules.pop("synth_audio", None)
    sys.modules.pop("_synthesize_audio", None)
    sa = importlib.import_module("synth_audio")
    case = tmp_path / "c1"; case.mkdir()
    (case / "metadata.json").write_text('{"id":"c1","steps":[{"turn":1,"user":"hi","acting_notes":"calm"},{"turn":2,"user":"yes"}]}')
    (case / "input_turn1.wav").write_bytes(b"x")
    calls = []

    def fake_synthesize(text, voice, instr, out):
        calls.append((text, voice, instr))
        out.write_bytes(b"w")
        return True, None

    monkeypatch.setattr(sa, "synthesize", fake_synthesize)
    sa.synth_folder(tmp_path, {"c1": {"voice": "verse", "age_descriptor": "an adult", "accent_descriptor": "a General American"}})
    assert [c[0] for c in calls] == ["yes"]
    assert "a General American accent" in calls[0][2]
