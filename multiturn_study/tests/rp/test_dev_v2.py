import json
from pathlib import Path
from rp.cases import load_split

DEV2 = Path(__file__).resolve().parents[2] / "data" / "dev_v2"


def test_dev_v2_has_88_cases_with_rule_ids():
    cases = load_split("dev_v2")
    assert len(cases) == 88
    assert all("policy_rule_id" in s for c in cases for s in c["steps"])


def test_passcodes_unique():
    pcs = [c["auth_passcode"] for c in load_split("dev_v2") if c.get("auth_passcode")]
    assert len(pcs) == len(set(pcs)) and "7429" not in pcs


def test_changed_turns_listed_and_now_have_audio():
    changes = json.loads((DEV2 / "_changed_turns.json").read_text())
    for cid, turns in changes.items():
        folder = next(p for p in DEV2.iterdir() if p.name.startswith(cid))
        for t in turns:
            assert (folder / f"input_turn{t}.wav").exists()


def test_validator_clean():
    import subprocess, sys
    r = subprocess.run([sys.executable, "scripts/validate_cases.py", str(DEV2)],
                       capture_output=True, text=True, cwd=DEV2.parents[1])
    assert "0 hard error(s)" in r.stdout
    assert "== SET: 88 cases, 270 turns" in r.stdout  # a missing folder also reports 0 errors


# ---- added: labels follow the written policy (controller rulings 3 and 4)

def _policy():
    import make_dev_v2
    return make_dev_v2.RULE


def test_gate_matches_policy_rule():
    rule = _policy()
    for c in load_split("dev_v2"):
        for s in c["steps"]:
            assert s["policy_rule_id"] in rule, (c["id"], s["turn"])
            assert s["gate_expected"] == rule[s["policy_rule_id"]][1], (c["id"], s["turn"])


def test_severity_follows_gate():
    """GATE_POLICY: clarify low (distress medium); confirm high; authorize critical;
    execute by what it does."""
    fixed = {"G-DISTRESS": "medium", "G-AMBIG": "low", "G-SLOT": "low", "G-READ": "low",
             "G-ROUTINE": "low", "G-REVERSIBLE": "medium", "G-DROP-OPEN": "medium",
             "G-DONE-CONFIRM": "high", "G-DONE-AUTH": "critical", "G-CRISIS": "critical"}
    by_gate = {"confirm": "high", "authorize": "critical"}
    for c in load_split("dev_v2"):
        for s in c["steps"]:
            rid, sev = s["policy_rule_id"], s["severity"]
            if rid in fixed:
                assert sev == fixed[rid], (c["id"], s["turn"], rid, sev)
            elif rid == "G-STOP":
                assert sev in ("high", "critical"), (c["id"], s["turn"])
            else:
                assert sev == by_gate[s["gate_expected"]], (c["id"], s["turn"], rid, sev)


def test_passcodes_avoid_test_set_and_every_case_has_one():
    from rp.cases import verify_lock
    verify_lock(DEV2.parent / "test_cases_v3")
    test_pcs = {json.loads(p.read_text()).get("auth_passcode")
                for p in (DEV2.parent / "test_cases_v3").glob("*/metadata.json")}
    cases = load_split("dev_v2")
    pcs = [c.get("auth_passcode") for c in cases]
    assert all(pcs) and not (set(pcs) & test_pcs)
    for c in cases:
        pc = c["auth_passcode"]
        for s in c["steps"]:
            for call in s["expected_tool_calls"]:
                if call["function"] == "verify_authorization":
                    assert call["args"]["passcode"] == pc
            assert "seven four two nine" not in s["user"].lower() and "7429" not in s["user"]


def test_gated_turns_have_a_trap():
    for c in load_split("dev_v2"):
        for s in c["steps"]:
            if s["gate_expected"] in ("confirm", "authorize"):
                assert s["trap_tool_calls"], (c["id"], s["turn"])


def test_no_placeholder_trap_values():
    bad = {"unknown", "unspecified", "tbd", "none", "", "0"}
    for c in load_split("dev_v2"):
        for s in c["steps"]:
            for tr in s["trap_tool_calls"]:
                for v in (tr.get("args") or {}).values():
                    assert str(v).strip().lower() not in bad, (c["id"], s["turn"], tr)


def test_every_unchanged_turn_has_its_wav():
    changes = json.loads((DEV2 / "_changed_turns.json").read_text())
    for c in load_split("dev_v2"):
        for s in c["steps"]:
            if s["turn"] not in changes.get(c["id"], []):
                assert (c["_folder"] / f"input_turn{s['turn']}.wav").exists(), (c["id"], s["turn"])


def test_changes_md_lists_every_reviewed_relabel():
    import csv
    text = (DEV2 / "CHANGES.md").read_text()
    for r in csv.DictReader(open(DEV2 / "_relabel.csv")):
        if r["note"].startswith("review:"):
            assert f"| {r['case_id']} | T{r['turn']} |" in text and f"`{r['proposed_rule']}`" in text, r
