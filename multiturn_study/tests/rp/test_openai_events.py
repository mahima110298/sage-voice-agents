"""Offline tests for the OpenAI Realtime adapter: event parsing (GA + beta
names) and full turns against a fake Realtime websocket (no network)."""
import asyncio
import base64
import json
import wave
from contextlib import asynccontextmanager

import pytest

from rp.agents.openai_realtime import parse_event
from rp.interceptors import build_config


# ---------------------------------------------------------------- parse_event

def test_transcript_delta_both_names():
    assert parse_event({"type": "response.output_audio_transcript.delta", "delta": "Hi"}) == ("text", {"delta": "Hi"})
    assert parse_event({"type": "response.audio_transcript.delta", "delta": "Hi"}) == ("text", {"delta": "Hi"})


def test_text_delta_both_names():
    assert parse_event({"type": "response.output_text.delta", "delta": "x"}) == ("text", {"delta": "x"})
    assert parse_event({"type": "response.text.delta", "delta": "x"}) == ("text", {"delta": "x"})


def test_audio_delta_both_names():
    assert parse_event({"type": "response.output_audio.delta", "delta": "AAAA"})[0] == "audio"
    assert parse_event({"type": "response.audio.delta", "delta": "AAAA"})[0] == "audio"


def test_function_call_done():
    k, v = parse_event({"type": "response.function_call_arguments.done", "call_id": "c1", "name": "sign_lease",
                        "arguments": "{\"unit_id\": \"3B\"}"})
    assert k == "call" and v == {"call_id": "c1", "name": "sign_lease", "args": {"unit_id": "3B"}}


def test_function_call_bad_json_kept_raw():
    k, v = parse_event({"type": "response.function_call_arguments.done", "call_id": "c1", "name": "f",
                        "arguments": "{oops"})
    assert k == "call" and v["args"] == {"_raw": "{oops"}


def test_response_done_usage():
    k, v = parse_event({"type": "response.done", "response": {"usage": {"input_tokens": 5, "output_tokens": 7}}})
    assert k == "done" and v["usage"]["output_tokens"] == 7


def test_response_done_status():
    k, v = parse_event({"type": "response.done", "response": {"status": "failed",
                        "status_details": {"type": "failed", "error": {"code": "server_error"}}}})
    assert k == "done" and v["status"] == "failed" and "server_error" in v["status_details"]


def test_error_event():
    k, v = parse_event({"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}})
    assert k == "error" and "bad" in v["message"]


def test_other_events():
    for t in ("session.updated", "rate_limits.updated", "response.output_item.added",
              "response.function_call_arguments.delta"):
        assert parse_event({"type": t}) == ("other", {})


def test_add_usage_flattens_and_sums():
    from rp.agents.openai_realtime import add_usage
    acc = {}
    u = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
         "input_token_details": {"audio_tokens": 6, "cached_tokens": 0,
                                 "cached_tokens_details": {"text_tokens": 0}},
         "output_token_details": {"audio_tokens": 3, "text_tokens": 1}}
    add_usage(acc, u)
    add_usage(acc, u)
    assert acc["input_tokens"] == 20 and acc["total_tokens"] == 28
    assert acc["input_token_details.audio_tokens"] == 12
    assert acc["input_token_details.cached_tokens_details.text_tokens"] == 0
    assert acc["output_token_details.text_tokens"] == 2


# ---------------------------------------------------------------- resampling

def _write_wav(path, rate, n_frames, channels=1):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x01\x00" * n_frames * channels)


def test_wav_to_pcm16_resamples_to_24k_mono(tmp_path):
    from rp.agents.openai_realtime import wav_to_pcm16
    p = tmp_path / "a.wav"
    _write_wav(p, 16000, 16000, channels=2)  # 1 s stereo at 16 kHz
    pcm = wav_to_pcm16(p, 24000)
    assert abs(len(pcm) - 24000 * 2) <= 4  # ~1 s of 24 kHz 16-bit mono


# ---------------------------------------------------------------- adapter against a fake websocket

def _case(tmp_path, n=2):
    for i in range(1, n + 1):
        _write_wav(tmp_path / f"input_turn{i}.wav", 24000, 12000)  # 0.5 s
    return {"id": "fake", "domain": "ecommerce_support", "_folder": tmp_path, "auth_passcode": "1111",
            "mock_data": {"list_orders": [{"order_id": "A"}], "client_id": "C9"},
            "steps": [{"turn": i, "user": f"u{i}"} for i in range(1, n + 1)]}


@pytest.fixture
def fake_asr(monkeypatch):
    import rp.asr
    monkeypatch.setattr(rp.asr, "transcribe", lambda wav: f"asr:{wav.name}")


def T(s):
    return {"type": "response.output_audio_transcript.delta", "delta": s}


def A():
    return {"type": "response.output_audio.delta", "delta": "AAAA"}


def C(name, args, cid):
    return {"type": "response.function_call_arguments.done", "call_id": cid, "name": name,
            "item_id": "i" + cid, "arguments": json.dumps(args)}


def D(inp=10, outp=5):
    return {"type": "response.done", "response": {"status": "completed", "usage": {
        "input_tokens": inp, "output_tokens": outp, "total_tokens": inp + outp,
        "output_token_details": {"audio_tokens": outp, "text_tokens": 0}}}}


class FakeWS:
    """Fake Realtime server: answers session.update with session.updated, and
    every response.create with the next scripted list of server events."""
    def __init__(self, responses, updated_session=None):
        self.responses, self.sent, self.queue = list(responses), [], [{"type": "session.created"}]
        self.updated_session = updated_session  # override what session.updated reports

    async def send(self, msg):
        ev = json.loads(msg)
        self.sent.append(ev)
        if ev["type"] == "session.update":
            self.queue.append({"type": "session.updated",
                               "session": self.updated_session or ev["session"]})
        elif ev["type"] == "response.create":
            self.queue.extend(self.responses.pop(0) if self.responses else [])

    async def recv(self):
        if not self.queue:
            await asyncio.sleep(3600)  # nothing more from the server: hang (timeout path)
        ev = self.queue.pop(0)
        while isinstance(ev, float):  # a scripted server-side delay
            await asyncio.sleep(ev)
            ev = self.queue.pop(0)
        return json.dumps(ev)

    def of(self, t):
        return [e for e in self.sent if e["type"] == t]


def _adapter(ws, seen=None):
    import rp.agents.openai_realtime as oar

    @asynccontextmanager
    async def ws_connect(url, headers):
        if seen is not None:
            seen.update(url=url, headers=headers)
        yield ws
    return oar.OpenAIRealtimeAdapter("gpt-realtime-test", ws_connect=ws_connect, api_key="sk-test")


def test_turn_with_tool_call_routes_through_handle_call(tmp_path, fake_asr):
    import rp.agents.openai_realtime as oar
    ws = FakeWS([
        [C("list_orders", {}, "c1"), D()],                    # turn 1, response 1: a lookup
        [T("You have "), A(), T("order A."), D()],            # turn 1, response 2: speaks the result
        [C("cancel_order", {"order_id": "A"}, "c2"), D()],    # turn 2: cancel -> SAGE blocks it
        [T("I need you to confirm first."), D()],
    ])
    seen = {}
    turns = asyncio.run(_adapter(ws, seen).run(_case(tmp_path), build_config("SAGE", None), "BASE"))
    assert [t.error for t in turns] == [None, None]
    assert turns[0].agent_text == "You have order A." and turns[0].calls[0].function == "list_orders"
    assert turns[1].blocked and not turns[1].calls
    assert turns[1].agent_text == "I need you to confirm first."
    outs = [e["item"] for e in ws.of("conversation.item.create")]
    assert outs[0] == {"type": "function_call_output", "call_id": "c1",
                       "output": json.dumps({"result": [{"order_id": "A"}]})}
    assert outs[1]["call_id"] == "c2" and json.loads(outs[1]["output"])["error"] == "BLOCKED_BY_POLICY"
    # usage summed over both responses of turn 1; latency set
    assert turns[0].usage["input_tokens"] == 20 and turns[0].usage["output_token_details.audio_tokens"] == 10
    assert all(t.latency_ms is not None and t.latency_ms >= 0 for t in turns)
    # session.update: instructions, tools, 24 kHz PCM in, turn detection off, audio out
    s = ws.of("session.update")[0]["session"]
    assert s["type"] == "realtime" and s["instructions"].startswith("BASE") and "client_id=C9" in s["instructions"]
    assert s["output_modalities"] == ["audio"]
    assert s["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert s["audio"]["input"]["turn_detection"] is None
    assert {t["name"] for t in s["tools"]} >= {"list_orders", "cancel_order"}
    assert all(t["type"] == "function" for t in s["tools"])
    # audio: appended in chunks then committed, then response.create
    kinds = [e["type"] for e in ws.sent]
    first_commit = kinds.index("input_audio_buffer.commit")
    assert kinds[first_commit + 1] == "response.create"
    appends = ws.of("input_audio_buffer.append")
    audio = b"".join(base64.b64decode(e["audio"]) for e in appends)
    assert len(audio) == 2 * 12000 * 2 and len(base64.b64decode(appends[0]["audio"])) == oar.CHUNK_BYTES
    assert len(ws.of("response.create")) == 4
    assert "model=gpt-realtime-test" in seen["url"] and seen["headers"]["Authorization"] == "Bearer sk-test"


def test_tool_round_valve_flags_turn_and_continues(tmp_path, fake_asr):
    import rp.agents.openai_realtime as oar
    loop = [[C("list_orders", {}, f"c{i}"), D()] for i in range(oar.MAX_TOOL_RESPONSE_ROUNDS + 1)]
    ws = FakeWS(loop + [[T("next turn ok"), D()]])
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path), build_config("B0", None), "BASE"))
    # 1 initial response.create + 4 follow-ups for turn 1, then 1 for turn 2
    assert len(ws.of("response.create")) == 1 + oar.MAX_TOOL_RESPONSE_ROUNDS + 1
    assert turns[0].error is None and turns[0].usage["safety_valve"] == "tool_rounds"
    assert len(turns[0].calls) == oar.MAX_TOOL_RESPONSE_ROUNDS + 1
    assert turns[1].error is None and turns[1].agent_text == "next turn ok"
    assert "safety_valve" not in turns[1].usage


def test_error_event_fails_turn_and_skips_rest(tmp_path, fake_asr):
    from rp.agents.base import SKIPPED
    ws = FakeWS([[T("part"), {"type": "error", "error": {"message": "boom"}}]])
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path), build_config("B0", None), "BASE"))
    assert "boom" in turns[0].error and turns[0].agent_text == "part"
    assert turns[1].error == SKIPPED


def test_failed_response_is_an_error(tmp_path, fake_asr):
    ws = FakeWS([[{"type": "response.done", "response": {"status": "failed", "status_details": {
        "type": "failed", "error": {"code": "server_error"}}}}]])
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path, n=1), build_config("B0", None), "BASE"))
    assert "failed" in turns[0].error


def test_turn_timeout(tmp_path, fake_asr, monkeypatch):
    import rp.agents.openai_realtime as oar
    monkeypatch.setattr(oar, "TURN_TIMEOUT_S", 0.2)
    ws = FakeWS([[T("hm")]])  # never sends response.done
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path, n=1), build_config("B0", None), "BASE"))
    assert turns[0].error == "timeout" and turns[0].agent_text == "hm"


def test_agents_registry_openai():
    from rp.agents import AGENTS
    from rp.agents.openai_realtime import OpenAIRealtimeAdapter
    for name in ("gpt-realtime-2.1", "gpt-realtime-2.1-mini", "gpt-realtime-1.5"):
        ad = AGENTS[name]()
        assert isinstance(ad, OpenAIRealtimeAdapter) and ad.model == name


# ---------------------------------------------------------------- fix round 1: latency, status, VAD

def _lifecycle():
    return [{"type": "response.created", "response": {"status": "in_progress"}},
            {"type": "rate_limits.updated", "rate_limits": []}, 0.05]


def test_latency_ignores_lifecycle_events(tmp_path, fake_asr):
    ws = FakeWS([_lifecycle() + [T("Hello"), D()]])
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path, n=1), build_config("B0", None), "BASE"))
    assert turns[0].error is None and turns[0].latency_ms >= 50


def test_empty_delta_does_not_stop_latency_clock(tmp_path, fake_asr):
    ws = FakeWS([[{"type": "response.created"}, T(""), 0.05, T("Hi"), D()]])
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path, n=1), build_config("B0", None), "BASE"))
    assert turns[0].agent_text == "Hi" and turns[0].latency_ms >= 50


def test_audio_and_call_still_stop_latency_clock(tmp_path, fake_asr):
    ws = FakeWS([[A(), 0.05, T("x"), D()],
                 [C("list_orders", {}, "c1"), 0.05, D()], [T("done"), D()]])
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path), build_config("B0", None), "BASE"))
    assert turns[0].latency_ms < 50 and turns[1].latency_ms < 50


def _done_status(status):
    return {"type": "response.done", "response": {"status": status, "usage": {"output_tokens": 1}}}


@pytest.mark.parametrize("status", ["incomplete", "cancelled"])
def test_non_completed_status_recorded(tmp_path, fake_asr, status):
    ws = FakeWS([[T("cut"), _done_status(status)], [T("ok"), D()]])
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path), build_config("B0", None), "BASE"))
    assert turns[0].error is None and turns[0].usage["response_status"] == status
    assert turns[1].error is None and "response_status" not in turns[1].usage


def test_completed_status_not_recorded(tmp_path, fake_asr):
    ws = FakeWS([[T("ok"), _done_status("completed")]])
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path, n=1), build_config("B0", None), "BASE"))
    assert "response_status" not in turns[0].usage


def test_server_vad_enabled_is_a_session_error(tmp_path, fake_asr):
    vad = {"type": "realtime", "audio": {"input": {"turn_detection": {"type": "server_vad"}}}}
    ws = FakeWS([[T("x"), D()]], updated_session=vad)
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path, n=1), build_config("B0", None), "BASE"))
    assert turns[0].error.startswith("session_error: ") and "turn_detection" in turns[0].error
    assert not ws.of("input_audio_buffer.append")


def test_turn_detection_absent_is_accepted(tmp_path, fake_asr):
    ws = FakeWS([[T("x"), D()]], updated_session={"type": "realtime", "audio": {"input": {}}})
    turns = asyncio.run(_adapter(ws).run(_case(tmp_path, n=1), build_config("B0", None), "BASE"))
    assert turns[0].error is None and turns[0].agent_text == "x"
