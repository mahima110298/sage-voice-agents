"""Offline tests for the cascade adapter (rp.agents.cascade), driven by a fake
chat client (no network)."""
import asyncio

import pytest

from rp.agents.cascade import CascadeAdapter, MAX_TOOL_RESPONSE_ROUNDS
from rp.interceptors import build_config


def _case(tmp_path, n=1, domain="ecommerce_support"):
    return {"id": "c", "domain": domain, "mock_data": {"list_orders": [{"order_id": "A"}],
            "client_id": "C9"}, "auth_passcode": "1111", "_folder": tmp_path,
            "steps": [{"turn": i, "user": f"u{i}"} for i in range(1, n + 1)]}


@pytest.fixture
def fake_asr(monkeypatch):
    import rp.asr as asr
    monkeypatch.setattr(asr, "transcribe", lambda wav: "what are my orders")


class Fake:
    """Scripted client: script[n] returns for the n-th call() invocation (1-indexed).
    Falls back to the last entry once the script is exhausted."""
    def __init__(self, script):
        self.script, self.n, self.calls = script, 0, []

    def chat(self, model, messages, tools):
        self.n += 1
        self.calls.append((model, [dict(m) for m in messages], tools))
        step = self.script[min(self.n, len(self.script)) - 1]
        if isinstance(step, Exception):
            raise step
        return step


def test_cascade_loop(tmp_path, fake_asr):
    """The brief's loop test: one tool call, then final text."""
    class Loop:
        def __init__(self): self.n = 0
        def chat(self, model, messages, tools):
            self.n += 1
            if self.n == 1:
                return {"tool_calls": [{"id": "t1", "name": "list_orders", "arguments": "{}"}],
                        "content": "", "usage": {}}
            return {"tool_calls": [], "content": "You have one order.", "usage": {}}

    case = _case(tmp_path)
    turns = asyncio.run(CascadeAdapter("m", client=Loop()).run(case, build_config("B0", None), "sys"))
    assert turns[0].calls[0].function == "list_orders"
    assert "one order" in turns[0].agent_text
    assert turns[0].error is None
    assert turns[0].latency_ms is not None and turns[0].latency_ms >= 0


def test_cascade_content_with_tool_call_is_space_joined(tmp_path, fake_asr):
    """A round that returns content AND a tool call, followed by a round with more
    content: agent_text keeps both utterances, joined by a single space (matching
    openai_realtime.py's between-responses separator) rather than run together."""
    fake = Fake([
        {"tool_calls": [{"id": "t1", "name": "list_orders", "arguments": "{}"}],
         "content": "Sure, let me check that for you.", "usage": {}},
        {"tool_calls": [], "content": "Your order ships tomorrow.", "usage": {}},
    ])
    case = _case(tmp_path)
    turns = asyncio.run(CascadeAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    assert turns[0].error is None
    assert turns[0].agent_text == "Sure, let me check that for you. Your order ships tomorrow."


def test_cascade_content_with_tool_call_no_leading_space_when_first_empty(tmp_path, fake_asr):
    """When the first round's content is empty, the joined text has no leading space."""
    fake = Fake([
        {"tool_calls": [{"id": "t1", "name": "list_orders", "arguments": "{}"}],
         "content": "", "usage": {}},
        {"tool_calls": [], "content": "Your order ships tomorrow.", "usage": {}},
    ])
    case = _case(tmp_path)
    turns = asyncio.run(CascadeAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    assert turns[0].error is None
    assert turns[0].agent_text == "Your order ships tomorrow."


def test_cascade_blocked_call_returns_decision_message(tmp_path, fake_asr):
    """A gated call SAGE blocks: the tool message the model sees is the Decision's
    message (BLOCKED_BY_POLICY), and it never reaches the backend as an executed call."""
    fake = Fake([
        {"tool_calls": [{"id": "t1", "name": "cancel_order", "arguments": '{"order_id": "A"}'}],
         "content": "", "usage": {}},
        {"tool_calls": [], "content": "Okay, I will not cancel it.", "usage": {}},
    ])
    case = _case(tmp_path)
    turns = asyncio.run(CascadeAdapter("m", client=fake).run(case, build_config("SAGE", None), "sys"))
    assert turns[0].error is None
    assert turns[0].blocked and not turns[0].calls
    tool_msg = fake.calls[1][1][-1]
    assert tool_msg["role"] == "tool"
    assert '"error": "BLOCKED_BY_POLICY"' in tool_msg["content"]


def test_cascade_valve_trips_after_max_rounds(tmp_path, fake_asr):
    """The model keeps calling a tool forever: the loop stops after
    MAX_TOOL_RESPONSE_ROUNDS and flags the valve instead of erroring."""
    always_calls = {"tool_calls": [{"id": "t1", "name": "list_orders", "arguments": "{}"}],
                    "content": "", "usage": {}}

    class AlwaysCalls:
        def __init__(self): self.n = 0
        def chat(self, model, messages, tools):
            self.n += 1
            return dict(always_calls)

    fake = AlwaysCalls()
    case = _case(tmp_path)
    turns = asyncio.run(CascadeAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    assert turns[0].error is None
    assert turns[0].usage["safety_valve"] == "tool_rounds"
    assert len(turns[0].calls) == MAX_TOOL_RESPONSE_ROUNDS + 1
    assert fake.n == MAX_TOOL_RESPONSE_ROUNDS + 1


def test_cascade_api_error_marks_turn_error_and_skips_rest(tmp_path, fake_asr):
    fake = Fake([RuntimeError("connection refused")])
    case = _case(tmp_path, n=3)
    turns = asyncio.run(CascadeAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    assert [t.turn for t in turns] == [1, 2, 3]
    assert turns[0].error == "connection refused"
    assert turns[1].error == "skipped_after_prior_turn_error"
    assert turns[2].error == "skipped_after_prior_turn_error"


def test_cascade_messages_persist_across_turns(tmp_path, fake_asr):
    """Conversation memory carries across turns inside one case: the system prompt
    is sent once and every prior user/assistant turn stays in the message list."""
    fake = Fake([
        {"tool_calls": [], "content": "first reply", "usage": {}},
        {"tool_calls": [], "content": "second reply", "usage": {}},
    ])
    case = _case(tmp_path, n=2)
    asyncio.run(CascadeAdapter("m", client=fake).run(case, build_config("B0", None), "sys"))
    second_call_messages = fake.calls[1][1]
    roles_and_text = [(m["role"], m.get("content")) for m in second_call_messages]
    assert roles_and_text[0][0] == "system" and roles_and_text[0][1].startswith("sys")
    assert "client_id=C9" in roles_and_text[0][1]
    assert ("assistant", "first reply") in roles_and_text
    assert roles_and_text.count(("user", "what are my orders")) == 2


def test_cascade_registered_in_agents(tmp_path):
    from rp.agents import AGENTS
    ad = AGENTS["cascade"]()
    assert type(ad).__name__ == "CascadeAdapter"


# ---------------------------------------------------------------- Task 16b: open cascade

def test_open_cascade_agent_hears_whisper_sage_sees_harness_asr(tmp_path, fake_asr, monkeypatch):
    """asr="groq-whisper": the LLM's user message is the Whisper transcript, while
    state.user_text (what SAGE and the verifier read) stays the harness ASR."""
    import rp.agents.cascade as cascade
    import rp.asr as asr
    monkeypatch.setattr(asr, "_whisper_api", lambda wav: f"whisper says {wav.stem}")
    (tmp_path / "input_turn1.wav").write_bytes(b"RIFF")
    fake = Fake([{"tool_calls": [], "content": "ok", "usage": {}}])
    case = _case(tmp_path)
    ad = CascadeAdapter("openai/gpt-oss-120b", client=fake, asr="groq-whisper", llm_base_url=cascade.GROQ)
    turns = asyncio.run(ad.run(case, build_config("B0", None), "sys"))
    assert turns[0].error is None
    assert turns[0].user_text == "what are my orders"
    assert fake.calls[0][1][-1] == {"role": "user", "content": "whisper says input_turn1"}
    assert fake.calls[0][0] == "openai/gpt-oss-120b"


def test_cascade_default_asr_is_harness(tmp_path, fake_asr, monkeypatch):
    import rp.asr as asr
    monkeypatch.setattr(asr, "_whisper_api", lambda wav: pytest.fail("whisper must not be called"))
    fake = Fake([{"tool_calls": [], "content": "ok", "usage": {}}])
    turns = asyncio.run(CascadeAdapter("m", client=fake).run(_case(tmp_path), build_config("B0", None), "sys"))
    assert turns[0].error is None
    assert fake.calls[0][1][-1] == {"role": "user", "content": "what are my orders"}


def test_cascade_rejects_unknown_asr():
    with pytest.raises(ValueError):
        CascadeAdapter("m", client=object(), asr="nope")


def test_llm_base_url_selects_endpoint_and_key():
    import rp.agents.cascade as cascade
    groq = CascadeAdapter("openai/gpt-oss-120b", llm_base_url=cascade.GROQ).client
    assert groq.url == "https://api.groq.com/openai/v1/chat/completions" and groq.key_env == "GROQ_API_KEY"
    orr = CascadeAdapter("m").client
    assert orr.url == "https://openrouter.ai/api/v1/chat/completions" and orr.key_env == "OPENROUTER_API_KEY"


def test_open_cascade_registered():
    from rp.agents import AGENTS
    import rp.agents.cascade as cascade
    ad = AGENTS["open-cascade"]()
    assert isinstance(ad, CascadeAdapter)
    assert ad.model == "openai/gpt-oss-120b" and ad.asr == "groq-whisper"
    assert ad.client.url.startswith(cascade.GROQ)
    assert AGENTS["cascade"]().model == cascade.CASCADE_LLM
