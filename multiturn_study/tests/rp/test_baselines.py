import re
from pathlib import Path

from rp.interceptors import build_config, CONFIGS
from rp.prompts import policy_suffix
from rp.backend import MockBackend
from rp.state import SessionState
from rp.types import Call

GATE_POLICY_MD = Path(__file__).resolve().parent.parent.parent / "data" / "test_cases_v3" / "GATE_POLICY.md"
LABEL_TERMS = ("gate_expected", "expected_tool_calls", "trap_tool_calls", "must_not",
               "failure_mode", "policy_rule_id", "auth_passcode", "Label conventions")


def _rule_ids() -> list[str]:
    text = GATE_POLICY_MD.read_text()
    rules = text[text.index("## Rules"):text.index("## Label conventions")]
    return re.findall(r"`(G-[A-Z0-9-]+)`", rules)


def st():
    s = SessionState(MockBackend({"mock_data": {}}))
    s.begin_turn(1, "sign it")
    return s


def test_all_configs_build():
    for c in CONFIGS:
        assert build_config(c, verifier_obj=None).name == c


def test_b2_suffix_contains_policy_rules_not_labels():
    s = policy_suffix()
    assert "G-XFER-EXT" in s and "gate_expected" not in s
    for term in LABEL_TERMS:
        assert term not in s, f"policy_suffix() leaked label vocabulary: {term!r}"
    for rid in _rule_ids():
        assert rid in s, f"policy_suffix() is missing rule id {rid!r}"


def test_verifier_policy_text_is_bounded():
    class V:
        def check(self, call, state, policy_text):
            self.policy_text = policy_text
            return {"verdict": "ok", "candidates": [], "reason": "", "latency_ms": 1, "usage": {}}

    v = V()
    build_config("SAGE", v).on_call(Call("sign_lease", {"unit_id": "3B"}), st())
    for term in LABEL_TERMS:
        assert term not in v.policy_text, f"verifier policy_text leaked label vocabulary: {term!r}"


def test_b0_b1_b2_never_block():
    for c in ("B0", "B1", "B2"):
        assert build_config(c, None).on_call(Call("sign_lease", {"unit_id": "3B"}), st()).allow


def test_b3_blocks_high_tier_once_with_confirm_message():
    g = build_config("B3", None)
    s = st()
    d1 = g.on_call(Call("sign_lease", {"unit_id": "3B"}), s)
    assert not d1.allow and "confirm" in d1.message["detail"]
    assert g.on_call(Call("sign_lease", {"unit_id": "3B"}), s).allow  # pilot rule: second attempt passes


def test_b4_calls_verifier_on_every_state_change():
    class V:
        n = 0

        def check(self, *a):
            V.n += 1
            return {"verdict": "ok", "candidates": [], "reason": "", "latency_ms": 1, "usage": {}}

    build_config("B4", V()).on_call(Call("log_mood_entry", {"note": "x"}), st())
    assert V.n == 1


def test_b3_allows_exempt_tool():
    g = build_config("B3", None)
    assert g.on_call(Call("escalate_to_crisis_line", {}), st()).allow


def test_b3_allows_when_verify_authorization_already_called_this_turn():
    g = build_config("B3", None)
    s = st()
    s.record_call(Call("verify_authorization", {"passcode": "1234"}), {"result": {"authorized": True}})
    assert g.on_call(Call("sign_lease", {"unit_id": "3B"}), s).allow


def test_b1_has_safety_suffix_b2_has_policy_suffix():
    b1 = build_config("B1", None)
    b2 = build_config("B2", None)
    assert "Safety" in b1.prompt_suffix()
    assert "G-XFER-EXT" in b2.prompt_suffix()


def test_baselines_suffixes_unchanged_by_task_25():
    """Task 25 only touches SAGE's prompt_suffix() -- B0-B4's are untouched."""
    b0 = build_config("B0", None)
    b1 = build_config("B1", None)
    b2 = build_config("B2", None)
    b3 = build_config("B3", None)
    b4 = build_config("B4", None)
    assert b0.prompt_suffix() == ""
    assert "Safety" in b1.prompt_suffix()
    assert b2.prompt_suffix() == policy_suffix()
    assert b3.prompt_suffix() == ""
    assert b4.prompt_suffix() == ""


def test_b4_blocks_on_non_ok_verdict_with_generic_message():
    class V:
        def check(self, *a):
            return {"verdict": "ambiguous", "candidates": ["a", "b"], "reason": "two records fit",
                    "latency_ms": 1, "usage": {}}

    d = build_config("B4", V()).on_call(Call("sign_lease", {"unit_id": "3B"}), st())
    assert not d.allow and "check with the user" in d.message["detail"].lower()
