"""Offline tests for the audio-in chat adapter (rp.agents.audio_chat), driven by a
fake chat client (no network)."""
import asyncio
import base64

import pytest

from rp.agents.audio_chat import AudioChatAdapter, AUDIO_TURN_TEXT, MAX_TOOL_RESPONSE_ROUNDS
from rp.interceptors import build_config


def _case(tmp_path, n=1, domain="ecommerce_support"):
    for i in range(1, n + 1):
        (tmp_path / f"input_turn{i}.wav").write_bytes(f"RIFF-turn{i}".encode())
    return {"id": "c", "domain": domain, "mock_data": {"list_orders": [{"order_id": "A"}],
            "client_id": "C9"}, "auth_passcode": "1111", "_folder": tmp_path,
            "steps": [{"turn": i, "user": f"u{i}"} for i in range(1, n + 1)]}


@pytest.fixture
def fake_asr(monkeypatch):
    import rp.asr as asr
    monkeypatch.setattr(asr, "transcribe", lambda wav: "what are my orders")


class Fake:
    """Scripted client: script[n] returns for the n-th chat() call (1-indexed);
    falls back to the last entry once exhausted."""
    def __init__(self, script):
        self.script, self.n, self.calls = script, 0, []

    def chat(self, model, messages, tools):
        self.n += 1
        self.calls.append((model, [dict(m) for m in messages], tools))
        step = self.script[min(self.n, len(self.script)) - 1]
        if isinstance(step, Exception):
            raise step
        return step


def _audio_parts(msg):
    return [p for p in msg["content"] if p.get("type") == "input_audio"]


def test_user_turn_is_sent_as_audio_part_not_transcript(tmp_path, fake_asr):
    fake = Fake([{"tool_calls": [], "content": "Hi.", "usage": {}}])
    case = _case(tmp_path)
    turns = asyncio.run(AudioChatAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    assert turns[0].error is None
    model, msgs, tools = fake.calls[0]
    assert model == "m"
    user = msgs[-1]
    assert user["role"] == "user" and isinstance(user["content"], list)
    parts = _audio_parts(user)
    assert len(parts) == 1
    assert parts[0]["input_audio"]["format"] == "wav"
    assert base64.b64decode(parts[0]["input_audio"]["data"]) == b"RIFF-turn1"
    texts = [p["text"] for p in user["content"] if p.get("type") == "text"]
    assert texts == [AUDIO_TURN_TEXT]
    assert "what are my orders" not in str(user)            # the transcript is never sent
    # SAGE's view of the turn is still the harness transcript
    assert turns[0].user_text == "what are my orders"
    assert tools and all(t["type"] == "function" for t in tools)
    assert any(t["function"]["name"] == "list_orders" for t in tools)


def test_earlier_turns_stay_as_audio(tmp_path, fake_asr):
    fake = Fake([{"tool_calls": [], "content": "first reply", "usage": {}},
                 {"tool_calls": [], "content": "second reply", "usage": {}}])
    case = _case(tmp_path, n=2)
    asyncio.run(AudioChatAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    msgs = fake.calls[1][1]
    assert msgs[0]["role"] == "system" and "client_id=C9" in msgs[0]["content"]
    users = [m for m in msgs if m["role"] == "user"]
    assert [base64.b64decode(_audio_parts(u)[0]["input_audio"]["data"]) for u in users] == \
        [b"RIFF-turn1", b"RIFF-turn2"]
    assert {"role": "assistant", "content": "first reply"} in msgs


def test_tool_call_routed_through_handle_call(tmp_path, fake_asr):
    fake = Fake([
        {"tool_calls": [{"id": "t1", "name": "list_orders", "arguments": "{}"}], "content": "", "usage": {}},
        {"tool_calls": [], "content": "You have one order.", "usage": {}},
    ])
    case = _case(tmp_path)
    turns = asyncio.run(AudioChatAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    assert turns[0].error is None
    assert [c.function for c in turns[0].calls] == ["list_orders"]
    msgs = fake.calls[1][1]
    assistant, tool = msgs[-2], msgs[-1]
    assert assistant["role"] == "assistant" and assistant["tool_calls"][0]["id"] == "t1"
    assert tool == {"role": "tool", "tool_call_id": "t1", "content": '{"result": [{"order_id": "A"}]}'}
    assert turns[0].agent_text == "You have one order."
    assert turns[0].latency_ms is not None and turns[0].latency_ms >= 0


def test_blocked_call_returns_decision_message(tmp_path, fake_asr):
    fake = Fake([
        {"tool_calls": [{"id": "t1", "name": "cancel_order", "arguments": '{"order_id": "A"}'}],
         "content": "", "usage": {}},
        {"tool_calls": [], "content": "Okay, I will not cancel it.", "usage": {}},
    ])
    case = _case(tmp_path)
    turns = asyncio.run(AudioChatAdapter("m", client=fake).run(case, build_config("SAGE", None), "sys"))
    assert turns[0].error is None
    assert turns[0].blocked and not turns[0].calls
    tool_msg = fake.calls[1][1][-1]
    assert tool_msg["role"] == "tool" and '"error": "BLOCKED_BY_POLICY"' in tool_msg["content"]


def test_valve_trips_after_max_rounds(tmp_path, fake_asr):
    always = {"tool_calls": [{"id": "t1", "name": "list_orders", "arguments": "{}"}], "content": "", "usage": {}}
    fake = Fake([always])
    case = _case(tmp_path, n=2)
    turns = asyncio.run(AudioChatAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    assert turns[0].error is None
    assert turns[0].usage["safety_valve"] == "tool_rounds"
    assert len(turns[0].calls) == MAX_TOOL_RESPONSE_ROUNDS + 1
    assert turns[1].error is None                     # the case continues after a valve


def test_api_error_marks_turn_error_and_skips_rest(tmp_path, fake_asr):
    fake = Fake([RuntimeError("402 payment required")])
    case = _case(tmp_path, n=3)
    turns = asyncio.run(AudioChatAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    assert [t.error for t in turns] == ["402 payment required", "skipped_after_prior_turn_error",
                                        "skipped_after_prior_turn_error"]


def test_content_joined_by_single_space_and_usage_summed(tmp_path, fake_asr):
    fake = Fake([
        {"tool_calls": [{"id": "t1", "name": "list_orders", "arguments": "{}"}],
         "content": "Let me check. ", "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                               "prompt_tokens_details": {"audio_tokens": 7}}},
        {"tool_calls": [], "content": "  It ships tomorrow.",
         "usage": {"prompt_tokens": 20, "completion_tokens": 3, "prompt_tokens_details": {"audio_tokens": 7}}},
    ])
    case = _case(tmp_path)
    turns = asyncio.run(AudioChatAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    assert turns[0].agent_text == "Let me check. It ships tomorrow."
    assert turns[0].usage["prompt_tokens"] == 30 and turns[0].usage["completion_tokens"] == 5
    assert turns[0].usage["prompt_tokens_details.audio_tokens"] == 14


def test_open_agents_registered():
    from rp.agents import AGENTS
    for name, model in (("qwen3.8-omni-flash", "qwen/qwen3.8-omni-flash"),
                        ("voxtral-small-24b", "mistralai/voxtral-small-24b-2507")):
        ad = AGENTS[name]()
        assert isinstance(ad, AudioChatAdapter) and ad.model == model
    assert "nemotron-3-nano-omni" not in AGENTS          # dropped after its smoke (spec section 5)


def test_open_models_priced():
    import json
    from pathlib import Path
    prices = json.loads((Path(__file__).resolve().parents[2] / "rp" / "pricing.json").read_text())
    for m in ("qwen/qwen3.8-omni-flash", "mistralai/voxtral-small-24b-2507", "openai/gpt-oss-120b"):
        e = prices[m]
        assert e["input_per_mtok"] is not None and e["output_per_mtok"] is not None
        assert e["source"] and e["retrieved"]
    assert prices["whisper-large-v3"]["per_hour_usd"] > 0
    assert not any("nemotron" in k for k in prices)
