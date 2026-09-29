"""Offline tests for the agent-adapter interface (rp.agents.base) and the Gemini
Live adapter's message handling (driven by a fake Live session, no network)."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import pytest

from rp.agents.base import handle_call, run_case, SKIPPED
from rp.backend import MockBackend
from rp.interceptors import build_config
from rp.state import SessionState


def test_handle_call_blocks_and_records():
    b = MockBackend({"auth_passcode": "1111", "mock_data": {}})
    s = SessionState(b); s.begin_turn(1, "sign the lease for 3B")
    resp, executed = handle_call("sign_lease", {"unit_id": "3B"}, s, build_config("SAGE", None), b)
    assert not executed and resp["error"] == "BLOCKED_BY_POLICY"
    assert s.current.blocked and not s.current.calls and s.current.guard_ms >= 0


def test_handle_call_executes_lookup():
    b = MockBackend({"mock_data": {"list_orders": [{"order_id": "A"}]}})
    s = SessionState(b); s.begin_turn(1, "my orders")
    resp, executed = handle_call("list_orders", {}, s, build_config("B0", None), b)
    assert executed and resp["result"][0]["order_id"] == "A"


def test_handle_call_counts_verifier_calls():
    class V:
        def check(self, call, state, policy):
            return {"verdict": "ok", "reason": "", "candidates": []}
    b = MockBackend({"mock_data": {}})
    s = SessionState(b); s.begin_turn(1, "cancel order A")
    handle_call("cancel_order", {"order_id": "A"}, s, build_config("B4", V()), b)
    assert s.current.verifier_calls == 1


# ---------------------------------------------------------------- run_case with a fake adapter

def _case(tmp_path, n=3, domain="ecommerce_support"):
    return {"id": "fake", "domain": domain, "_folder": tmp_path, "auth_passcode": "1111",
            "mock_data": {"list_orders": [{"order_id": "A"}], "client_id": "C9"},
            "steps": [{"turn": i, "user": f"u{i}"} for i in range(1, n + 1)]}


class FakeAdapter:
    """Scripted adapter: script[turn] = list of ("call", name, args) / ("text", s) / ("raise", exc)."""
    def __init__(self, script):
        self.script, self.seen_instruction, self.seen_tools, self.responses = script, None, None, []

    @asynccontextmanager
    async def connect(self, case, tools, system_instruction):
        self.seen_instruction, self.seen_tools = system_instruction, tools
        yield "session"

    async def run_turn(self, session, wav, on_call, state, out):
        for step in self.script.get(state.current.turn, []):
            if step[0] == "call":
                self.responses.append(on_call(step[1], step[2]))
            elif step[0] == "text":
                out.text += step[1]
            elif step[0] == "raise":
                raise step[1]


@pytest.fixture
def fake_asr(monkeypatch):
    import rp.asr
    monkeypatch.setattr(rp.asr, "transcribe", lambda wav: f"asr:{wav.name}")


def test_run_case_happy_path(tmp_path, fake_asr):
    ad = FakeAdapter({1: [("call", "list_orders", {}), ("text", "you have order A")],
                      2: [("call", "cancel_order", {"order_id": "A"}), ("text", "confirm?")],
                      3: [("text", "ok")]})
    turns = asyncio.run(run_case(ad, _case(tmp_path), build_config("SAGE", None), "BASE"))
    assert [t.turn for t in turns] == [1, 2, 3] and all(t.error is None for t in turns)
    assert turns[0].user_text == "asr:input_turn1.wav" and turns[0].agent_text == "you have order A"
    assert turns[0].calls[0].function == "list_orders"
    assert turns[1].blocked and not turns[1].calls          # SAGE blocked the cancel
    assert ad.responses[1]["error"] == "BLOCKED_BY_POLICY"  # and the model got the block message
    # system instruction = prompt + interceptor suffix + backend session context
    assert ad.seen_instruction.startswith("BASE") and "client_id=C9" in ad.seen_instruction
    assert {t["name"] for t in ad.seen_tools} >= {"list_orders", "cancel_order"}


def test_run_case_uses_interceptor_prompt_suffix(tmp_path, fake_asr):
    ad = FakeAdapter({})
    icp = build_config("B1", None)
    asyncio.run(run_case(ad, _case(tmp_path, n=1), icp, "BASE"))
    assert ad.seen_instruction.startswith("BASE" + icp.prompt_suffix())


def test_run_case_turn_error_skips_rest_and_keeps_partial(tmp_path, fake_asr):
    ad = FakeAdapter({1: [("text", "hi")],
                      2: [("call", "list_orders", {}), ("text", "partial"), ("raise", RuntimeError("boom"))]})
    turns = asyncio.run(run_case(ad, _case(tmp_path, n=4), build_config("B0", None), "BASE"))
    assert [t.turn for t in turns] == [1, 2, 3, 4]
    assert turns[0].error is None
    assert turns[1].error == "boom" and turns[1].agent_text == "partial" and turns[1].calls
    assert turns[2].error == SKIPPED and turns[3].error == SKIPPED


def test_run_case_timeout_is_marked(tmp_path, fake_asr):
    ad = FakeAdapter({1: [("raise", asyncio.TimeoutError())]})
    turns = asyncio.run(run_case(ad, _case(tmp_path, n=2), build_config("B0", None), "BASE"))
    assert turns[0].error == "timeout" and turns[1].error == SKIPPED


def test_run_case_connect_failure_marks_all_turns(tmp_path, fake_asr):
    class Broken(FakeAdapter):
        @asynccontextmanager
        async def connect(self, case, tools, system_instruction):
            raise ConnectionError("refused")
            yield  # pragma: no cover
    turns = asyncio.run(run_case(Broken({}), _case(tmp_path, n=2), build_config("B0", None), "BASE"))
    assert turns[0].error.startswith("session_error: ") and "refused" in turns[0].error
    assert turns[1].error == SKIPPED and len(turns) == 2


# ---------------------------------------------------------------- Gemini adapter against a fake Live session

def _fc(name, args, id_="c1"):
    return NS(tool_call=NS(function_calls=[NS(id=id_, name=name, args=args)]), server_content=None,
              usage_metadata=None)


def _txt(s, complete=False, usage=None):
    return NS(tool_call=None, usage_metadata=usage,
              server_content=NS(output_transcription=NS(text=s) if s else None, model_turn=None,
                                turn_complete=complete))


class FakeSession:
    def __init__(self, per_turn):
        self.per_turn, self.turn_i, self.sent, self.tool_responses = per_turn, -1, [], []

    async def send_realtime_input(self, **kw):
        self.sent.append(kw)
        if "activity_start" in kw:
            self.turn_i += 1

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(function_responses)

    def receive(self):
        async def gen():  # lazy, like the SDK: messages flow once the turn's audio is sent
            for m in self.per_turn[self.turn_i]:
                yield m
        return gen()


def test_gemini_adapter_routes_calls_and_collects(tmp_path, fake_asr, monkeypatch):
    from google.genai import types
    import rp.agents.gemini_live as gl
    monkeypatch.setattr(gl, "wav_to_pcm16_16k", lambda wav: b"\x00" * 7000)
    usage = types.UsageMetadata(prompt_token_count=10, response_token_count=5, total_token_count=15,
                                response_tokens_details=[types.ModalityTokenCount(
                                    modality=types.MediaModality.AUDIO, token_count=5)])
    sess = FakeSession([
        [_fc("list_orders", {}), _txt("You have "), _txt("order A.", complete=True, usage=usage)],
        [_fc("cancel_order", {"order_id": "A"}), _txt("Confirm cancel?", usage=usage), _txt("", complete=True,
                                                                                           usage=usage)],
    ])

    class FakeClient:
        def __init__(self):
            self.aio = NS(live=NS(connect=self.connect))
            self.config = None

        @asynccontextmanager
        async def connect(self, model, config):
            self.config = config
            yield sess

    client = FakeClient()
    ad = gl.GeminiLiveAdapter("gemini-test", client=client)
    turns = asyncio.run(ad.run(_case(tmp_path, n=2), build_config("SAGE", None), "BASE"))
    assert [t.error for t in turns] == [None, None]
    assert turns[0].agent_text == "You have order A." and turns[0].calls[0].function == "list_orders"
    assert turns[1].blocked and not turns[1].calls
    # the block message is what the model receives as the function response
    assert sess.tool_responses[1][0].response["error"] == "BLOCKED_BY_POLICY"
    assert sess.tool_responses[0][0].response == {"result": [{"order_id": "A"}]}
    # audio chunked at CHUNK_BYTES between activity_start and activity_end
    audio = [kw["audio"] for kw in sess.sent if "audio" in kw]
    assert len(audio) == 3 * 2 and len(audio[0].data) == gl.CHUNK_BYTES
    assert all(t.latency_ms is not None and t.latency_ms >= 0 for t in turns)
    assert turns[0].usage["prompt_token_count"] == 10
    assert turns[1].usage["total_token_count"] == 30  # summed across two messages
    assert turns[1].usage["response_tokens_details.AUDIO"] == 10
    assert "client_id=C9" in client.config.system_instruction
    fnames = {f.name for f in client.config.tools[0].function_declarations}
    assert "cancel_order" in fnames


def test_gemini_adapter_caps_tool_rounds(tmp_path, fake_asr, monkeypatch):
    import rp.agents.gemini_live as gl
    monkeypatch.setattr(gl, "wav_to_pcm16_16k", lambda wav: b"\x00" * 100)
    loop = [_fc("list_orders", {}, id_=f"c{i}") for i in range(10)]
    sess = FakeSession([loop, [_txt("next turn ok", complete=True)]])

    @asynccontextmanager
    async def connect(model, config):
        yield sess
    ad = gl.GeminiLiveAdapter("gemini-test", client=NS(aio=NS(live=NS(connect=connect))))
    turns = asyncio.run(ad.run(_case(tmp_path, n=2), build_config("B0", None), "BASE"))
    assert len(sess.tool_responses) == gl.MAX_TOOL_RESPONSE_ROUNDS
    # flagged, not an error; calls so far kept; the following turn still runs
    assert turns[0].error is None and turns[0].usage["safety_valve"] == "tool_rounds"
    assert len(turns[0].calls) == gl.MAX_TOOL_RESPONSE_ROUNDS + 1
    assert turns[1].error is None and turns[1].agent_text == "next turn ok"
    assert "safety_valve" not in turns[1].usage


def test_gemini_adapter_message_cap_flags_turn(tmp_path, fake_asr, monkeypatch):
    import rp.agents.gemini_live as gl
    monkeypatch.setattr(gl, "wav_to_pcm16_16k", lambda wav: b"\x00" * 100)
    control = [_txt("") for _ in range(gl.MAX_MESSAGES_PER_TURN + 5)]  # non-stream messages
    sess = FakeSession([[_txt("partial")] + control, [_txt("t2", complete=True)]])

    @asynccontextmanager
    async def connect(model, config):
        yield sess
    ad = gl.GeminiLiveAdapter("gemini-test", client=NS(aio=NS(live=NS(connect=connect))))
    turns = asyncio.run(ad.run(_case(tmp_path, n=2), build_config("B0", None), "BASE"))
    assert turns[0].error is None and turns[0].usage["safety_valve"] == "messages"
    assert turns[0].agent_text == "partial" and turns[1].agent_text == "t2" and turns[1].error is None


def test_gemini_adapter_stream_chunks_do_not_hit_message_cap(tmp_path, fake_asr, monkeypatch):
    # gemini-2.5 native audio streams a short reply in >80 small messages
    import rp.agents.gemini_live as gl
    monkeypatch.setattr(gl, "wav_to_pcm16_16k", lambda wav: b"\x00" * 100)
    chunks = [_txt("w ") for _ in range(gl.MAX_MESSAGES_PER_TURN + 20)] + [_txt("", complete=True)]
    sess = FakeSession([chunks])

    @asynccontextmanager
    async def connect(model, config):
        yield sess
    ad = gl.GeminiLiveAdapter("gemini-test", client=NS(aio=NS(live=NS(connect=connect))))
    turns = asyncio.run(ad.run(_case(tmp_path, n=1), build_config("B0", None), "BASE"))
    assert turns[0].error is None and turns[0].agent_text.count("w") == gl.MAX_MESSAGES_PER_TURN + 20


def test_gemini_config_pins_blocking_function_calls():
    from google.genai import types
    import rp.agents.gemini_live as gl
    cfg = gl.GeminiLiveAdapter("m", client=object())._config(
        [{"name": "f", "description": "d", "parameters": {"type": "object", "properties": {}}}], "SI")
    assert cfg.tools[0].function_declarations[0].behavior == types.Behavior.BLOCKING
    assert cfg.realtime_input_config.automatic_activity_detection.disabled is True


def test_agents_registry():
    from rp.agents import AGENTS
    assert set(AGENTS) == {"gemini-3.8-live", "gemini-3.1-flash-live-preview",
                           "gemini-2.5-flash-native-audio-latest", "gpt-realtime-2.1",
                           "gpt-realtime-2.1-mini", "gpt-realtime-1.5", "cascade",
                           "qwen3.8-omni-flash", "voxtral-small-24b", "open-cascade"}
    from rp.agents.gemini_live import GeminiLiveAdapter
    ad = AGENTS["gemini-3.8-live"]()
    assert isinstance(ad, GeminiLiveAdapter) and ad.model == "gemini-3.8-live"
