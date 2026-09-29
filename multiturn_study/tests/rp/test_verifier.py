from rp.backend import MockBackend
from rp.state import SessionState
from rp.types import Call
from rp.verifier import triggers, Verifier


def st(user, results=()):
    s = SessionState(MockBackend({"mock_data": {}})); s.begin_turn(1, user)
    for c, r in results:
        s.record_call(c, r)
    return s


def test_trigger_on_high_gate():
    assert "gated_action" in triggers(Call("cancel_card", {"card_id": "C1"}), st("cancel card C1"), "confirm")


def test_trigger_on_self_correction():
    t = triggers(Call("transfer_funds", {"to_account": "8821", "amount": 200}),
                 st("send two hundred to 8821 -- no wait, 8281"), "authorize")
    assert "self_correction" in t


def test_trigger_on_multiple_candidates():
    s = st("track my order", [(Call("list_orders"), {"result": [{"order_id": "A"}, {"order_id": "B"}]})])
    assert "multiple_candidates" in triggers(Call("track_order", {"order_id": "A"}), s, "execute")


def test_multiple_candidates_requires_call_target_in_list():
    s = st("check my accounts", [(Call("list_accounts"), {"result": [{"account_id": "A1"}, {"account_id": "A2"}]})])
    unrelated = triggers(Call("pay_bill", {"biller": "x", "amount": 50}), s, "execute")
    assert "multiple_candidates" not in unrelated
    related = triggers(Call("transfer_funds", {"to_account": "A1", "amount": 50}), s, "execute")
    assert "multiple_candidates" in related


def test_trigger_on_unmentioned_value():
    assert "unmentioned_value" in triggers(Call("add_to_cart", {"product_id": "ZZ9", "quantity": 1}), st("add it"), "execute")


def test_no_trigger_for_clean_low_call():
    assert triggers(Call("log_mood_entry", {"note": "good day"}), st("log that I had a good day"), "execute") == []


class FakeClient:
    def __init__(self, content): self.content = content
    def complete(self, model, messages): return {"content": self.content, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_verifier_parses_json():
    v = Verifier("m", client=FakeClient('{"verdict":"ambiguous","candidates":["A","B"],"reason":"two orders"}'))
    out = v.check(Call("cancel_order", {"order_id": "A"}), st("cancel my order"), "policy")
    assert out["verdict"] == "ambiguous" and out["candidates"] == ["A", "B"]


def test_verifier_fails_safe_on_bad_json():
    v = Verifier("m", client=FakeClient("not json"))
    assert v.check(Call("cancel_order", {"order_id": "A"}), st("x"), "p")["verdict"] == "ok"
    assert v.check(Call("cancel_order", {"order_id": "A"}), st("x"), "p")["reason"].startswith("unparseable:")


class RaisingClient:
    def complete(self, model, messages):
        raise ConnectionError("timed out")


def test_verifier_network_error_returns_ok_unparseable():
    v = Verifier("m", client=RaisingClient())
    out = v.check(Call("cancel_order", {"order_id": "A"}), st("cancel my order"), "policy")
    assert out["verdict"] == "ok"
    assert out["reason"].startswith("unparseable:")


class CapturingClient:
    def __init__(self):
        self.messages = None

    def complete(self, model, messages):
        self.messages = messages
        return {"content": '{"verdict":"ok","candidates":[],"reason":"fine"}', "usage": {}}


def test_prompt_has_no_labels():
    client = CapturingClient()
    v = Verifier("m", client=client)
    s = st("cancel my order")
    s.record_call(Call("list_orders"), {"result": [{"order_id": "A"}]})
    v.check(Call("cancel_order", {"order_id": "A"}), s, "some policy text")
    assert client.messages is not None
    content = client.messages[0]["content"]
    forbidden = ["gate_expected", "severity", "policy_rule_id", "expected_tool_calls",
                 "trap_tool_calls", "must_not", "failure_mode"]
    for label in forbidden:
        assert label not in content


def test_verifier_model_pinned_and_not_anthropic():
    from rp.verifier import VERIFIER_MODEL
    assert "REPLACE" not in VERIFIER_MODEL and not VERIFIER_MODEL.startswith("anthropic/")
