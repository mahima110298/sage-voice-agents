from rp.backend import MockBackend
from rp.interceptors import Sage, build_config, CONFIGS
from rp.prompts import policy_suffix
from rp.state import SessionState
from rp.types import Call


class V:
    def __init__(self, verdict="ok", cands=()): self.v, self.c, self.n = verdict, list(cands), 0
    def check(self, call, state, policy): self.n += 1; return {"verdict": self.v, "candidates": self.c, "reason": "", "latency_ms": 1.0, "usage": {}}


def st(user, prev="", mock=None, code="7429"):
    s = SessionState(MockBackend({"auth_passcode": code, "mock_data": mock or {}}))
    s.begin_turn(1, "x"); s.end_turn(prev); s.begin_turn(2, user); return s


XFER = Call("transfer_funds", {"to_account": "9999", "amount": 200})


def test_blocks_unconfirmed_authorize_with_credential_request():
    d = Sage(policy_text="p", verifier_obj=V()).on_call(XFER, st("send it"))
    assert not d.allow and d.gate == "authorize"
    assert "passcode" in d.message["detail"].lower()


def test_authorize_message_asks_for_passcode_and_verify():
    d = Sage(policy_text="p", verifier_obj=V()).on_call(XFER, st("send it"))
    detail = d.message["detail"].lower()
    assert "passcode" in detail and "verify_authorization" in detail


def test_allows_after_readback_and_passcode():
    s = st("seven four two nine", prev="Sending $200 to account 9999 needs your passcode.")
    s.record_call(Call("verify_authorization", {"passcode": "7429"}), {"result": {"authorized": True}})
    assert Sage(policy_text="p", verifier_obj=V()).on_call(XFER, s).allow


def test_verifier_ambiguity_blocks_with_clarify():
    v = V("ambiguous", ["QJ881", "RT552"])
    d = Sage(policy_text="p", verifier_obj=v).on_call(Call("cancel_order", {"order_id": "QJ881"}),
                                                      st("yes", prev="Cancel order QJ881?"))
    assert not d.allow and d.gate == "clarify" and "RT552" in d.message["detail"]


def test_routing_skips_verifier_on_clean_low_call():
    v = V()
    Sage(policy_text="p", verifier_obj=v).on_call(Call("log_mood_entry", {"note": "good day"}), st("log a good day"))
    assert v.n == 0


def test_no_routing_calls_verifier_on_every_state_change():
    v = V()
    Sage(policy_text="p", verifier_obj=v, routing=False).on_call(Call("log_mood_entry", {"note": "good day"}), st("log a good day"))
    assert v.n == 1


def test_ablation_gate_types_always_confirm():
    d = Sage(policy_text="p", verifier_obj=V(), gate_types=False).on_call(XFER, st("send it"))
    assert d.gate == "confirm" and "passcode" not in d.message["detail"].lower()


def test_exemption_passes_crisis_line():
    assert Sage(policy_text="p", verifier_obj=V()).on_call(Call("escalate_to_crisis_line", {}), st("I have a plan")).allow


def test_no_exemption_sends_crisis_to_verifier():
    v = V("ambiguous", [])
    d = Sage(policy_text="p", verifier_obj=v, exemption=False).on_call(Call("escalate_to_crisis_line", {}), st("rough week"))
    assert v.n == 1 and not d.allow


def test_decision_records_verifier_called():
    v = V()
    d = Sage(policy_text="p", verifier_obj=v).on_call(XFER, st("send it"))
    assert d.verifier_called is True and v.n == 1

    v2 = V()
    d2 = Sage(policy_text="p", verifier_obj=v2).on_call(Call("log_mood_entry", {"note": "good day"}), st("log a good day"))
    assert d2.verifier_called is False and v2.n == 0


# --- Carried-over fixes from Task 12's review -------------------------------

class FixedV:
    """A fake verifier that always returns the same verdict/candidates/reason,
    with an invocation counter -- for asserting the cache reuses the verdict."""
    def __init__(self, verdict="ambiguous", cands=(), reason=""):
        self.verdict, self.cands, self.reason, self.n = verdict, list(cands), reason, 0

    def check(self, call, state, policy):
        self.n += 1
        return {"verdict": self.verdict, "candidates": self.cands, "reason": self.reason,
                "latency_ms": 1.0, "usage": {}}


def test_verdict_cache_reused_within_turn_and_refreshed_on_new_turn():
    v = FixedV("ambiguous", ["QJ881", "RT552"])
    call = Call("cancel_order", {"order_id": "QJ881"})
    sage = Sage(policy_text="p", verifier_obj=v)
    s = st("yes", prev="Cancel order QJ881?")
    for _ in range(3):
        d = sage.on_call(call, s)
        assert not d.allow and d.gate == "clarify"
    assert v.n == 1  # 3 identical calls in one turn -> invoked once

    s.end_turn("Which order do you mean?")
    s.begin_turn(3, "QJ881")
    sage.on_call(call, s)
    assert v.n == 2  # a new turn -> invoked again


def test_clarify_message_ambiguous_names_candidates():
    v = FixedV("ambiguous", ["QJ881", "RT552"])
    d = Sage(policy_text="p", verifier_obj=v).on_call(Call("cancel_order", {"order_id": "QJ881"}),
                                                       st("yes", prev="Cancel order QJ881?"))
    detail = d.message["detail"]
    assert "QJ881" in detail and "RT552" in detail and "which one" in detail.lower()


def test_clarify_message_missing_value_asks_for_it():
    v = FixedV("missing_value", reason="no amount was given")
    d = Sage(policy_text="p", verifier_obj=v).on_call(XFER, st("send money to 9999"))
    detail = d.message["detail"].lower()
    assert not d.allow and d.gate == "clarify"
    assert "missing" in detail and "ask the user for the missing value" in detail
    assert "no amount was given" in detail


def test_clarify_message_wrong_target_names_ask_and_what_was_asked():
    v = FixedV("wrong_target", reason="the user asked to cancel RT552, not QJ881")
    d = Sage(policy_text="p", verifier_obj=v).on_call(Call("cancel_order", {"order_id": "QJ881"}),
                                                       st("yes", prev="Cancel order QJ881?"))
    detail = d.message["detail"].lower()
    assert not d.allow and d.gate == "clarify"
    assert "which one" in detail
    assert "the user asked to cancel rt552, not qj881" in detail


def test_no_unused_t0_attribute():
    # Regression: Sage.on_call no longer computes an unused perf-counter timestamp.
    import inspect
    src = inspect.getsource(Sage.on_call)
    assert "t0" not in src


def test_regression_authorize_block_persists_across_retries_until_next_turn():
    s = st("send it")
    sage = Sage(policy_text="p", verifier_obj=V())
    for _ in range(4):
        d = sage.on_call(XFER, s)
        assert not d.allow and d.gate == "authorize"
    s.end_turn("Sending $200 to account 9999 needs your passcode.")
    s.begin_turn(3, "seven four two nine")
    s.record_call(Call("verify_authorization", {"passcode": "7429"}), {"result": {"authorized": True}})
    assert sage.on_call(XFER, s).allow


def test_regression_two_turn_authorize_flow():
    sage = Sage(policy_text="p", verifier_obj=V())
    s = st("send 200 to 9999")
    d1 = sage.on_call(XFER, s)
    assert not d1.allow and d1.gate == "authorize"
    assert "passcode" in d1.message["detail"].lower()

    s.end_turn("Sending $200 to account 9999 needs your passcode.")
    s.begin_turn(3, "seven four two nine")
    s.record_call(Call("verify_authorization", {"passcode": "7429"}), {"result": {"authorized": True}})
    d2 = sage.on_call(XFER, s)
    assert d2.allow


def test_regression_clarify_resolves_on_later_turn():
    v = FixedV("ambiguous", ["QJ881", "RT552"])
    call = Call("cancel_order", {"order_id": "QJ881"})
    sage = Sage(policy_text="p", verifier_obj=v)
    s = st("cancel the order")
    d1 = sage.on_call(call, s)
    assert not d1.allow and d1.gate == "clarify"

    s.end_turn("Which order do you mean, QJ881 or RT552?")
    s.begin_turn(3, "yes, go ahead")
    v.verdict = "ok"
    d2 = sage.on_call(call, s)
    assert d2.allow


# --- Task 25: SAGE carries the policy prompt; SAGE-no_policy ablation ------

def test_sage_prompt_suffix_equals_b2():
    """Every SAGE config except SAGE-no_policy has a prompt_suffix() byte-identical
    to B2 PolicyPrompt's."""
    b2_suffix = build_config("B2", None).prompt_suffix()
    assert b2_suffix == policy_suffix()
    for c in CONFIGS:
        if c == "SAGE-no_policy" or not c.startswith("SAGE"):
            continue
        sage = build_config(c, verifier_obj=None)
        assert sage.prompt_suffix() == b2_suffix, f"{c} prompt_suffix() diverged from B2's"


def test_sage_no_policy_has_empty_suffix_and_same_enforcement():
    """SAGE-no_policy carries no prompt suffix, but still blocks the same call
    the same way as full SAGE (only the prompt changes, not enforcement)."""
    no_policy = build_config("SAGE-no_policy", verifier_obj=None)
    assert no_policy.prompt_suffix() == ""

    sage = Sage(policy_text="p", verifier_obj=V())
    no_policy_sage = Sage(policy_text="p", verifier_obj=V(), policy=False)
    d1 = sage.on_call(XFER, st("send it"))
    d2 = no_policy_sage.on_call(XFER, st("send it"))
    assert not d1.allow and not d2.allow
    assert d1.gate == d2.gate == "authorize"
    assert no_policy_sage.name == "SAGE-no_policy"


def test_configs_include_no_policy():
    assert "SAGE-no_policy" in CONFIGS
    cfg = build_config("SAGE-no_policy", verifier_obj=None)
    assert cfg.name == "SAGE-no_policy"
