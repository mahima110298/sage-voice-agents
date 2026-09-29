#!/usr/bin/env python3
"""Sanity check (not a test): run rp.policy.required_gate on every dev_v2 turn's
pending/expected state-changing call and compare to that turn's own
policy_rule_id label, printing agree/total per rule id.

This is a one-way check ON the dev_v2 labels FROM this script -- required_gate
itself (rp/policy.py) never reads gate_expected, severity, policy_rule_id,
expected_tool_calls, trap_tool_calls, must_not or failure_mode; it only sees
runtime state (mock_data via MockBackend, and the call under test).

Only covers the per-tool rules in scope for Task 9 (turn-level rules G-AMBIG,
G-SLOT, G-CRISIS, G-DISTRESS, etc. are the verifier's job, Task 11, and are
skipped here). Hard-coded to the dev_v2 split -- refuses to run on "test"
(the locked split) since that would mean reading test-set labels, which is
never allowed.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rp.backend import MockBackend
from rp.policy import required_gate
from rp.state import SessionState
from rp.tools import is_read_only
from rp.types import Call
from rp.cases import load_split

PER_TOOL_RULES = {
    "G-READ", "G-ROUTINE", "G-REVERSIBLE",
    "G-XFER-OWN", "G-XFER-OWN-LARGE", "G-XFER-EXT",
    "G-BILL", "G-BILL-LARGE",
    "G-ORDER", "G-ORDER-LARGE",
    "G-REFUND", "G-REFUND-LARGE",
    "G-CANCEL-REFUNDABLE", "G-CANCEL-NONREF",
    "G-BINDING", "G-IRREV",
    "G-DROP-OPEN", "G-DROP-CLOSED",
}


def _pending_call(step: dict) -> Call | None:
    """The turn's state-changing candidate call: the non-read-only,
    non-verify_authorization entry in expected_tool_calls (whether executed
    now or held pending -- both are "the call the gate applies to")."""
    for c in step.get("expected_tool_calls") or []:
        fn = c["function"]
        if fn == "verify_authorization" or is_read_only(fn):
            continue
        return Call(fn, c.get("args") or {})
    return None


def main() -> int:
    if len(sys.argv) > 1:
        print(f"refusing: this script only runs on dev_v2, got argument {sys.argv[1]!r}", file=sys.stderr)
        return 1

    cases = load_split("dev_v2")
    agree: dict[str, int] = {}
    total: dict[str, int] = {}

    for case in cases:
        mock_data = case.get("mock_data") or {}
        for step in case.get("steps", []):
            rule_id = step.get("policy_rule_id")
            if rule_id not in PER_TOOL_RULES:
                continue
            call = _pending_call(step)
            if call is None:
                continue
            state = SessionState(MockBackend({"mock_data": mock_data}))
            state.begin_turn(step.get("turn", 1), "")
            _, got_rule = required_gate(call, state)
            total[rule_id] = total.get(rule_id, 0) + 1
            if got_rule == rule_id:
                agree[rule_id] = agree.get(rule_id, 0) + 1

    print(f"{'rule':<24}{'agree/total':>14}")
    grand_agree = grand_total = 0
    for rule_id in sorted(total):
        a, t = agree.get(rule_id, 0), total[rule_id]
        grand_agree += a
        grand_total += t
        print(f"{rule_id:<24}{a}/{t}")
    print(f"{'TOTAL':<24}{grand_agree}/{grand_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
