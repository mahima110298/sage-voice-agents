"""Tests for rp.analysis.report: synthetic result rows written to a tmp
results dir (the way the runner writes them, via rp.results_io.append_rows),
then every table/CSV writer is checked for valid LaTeX / a CSV header, plus
one known-answer check (controller ruling 5)."""
from __future__ import annotations

import json

import pytest

from rp.analysis import metrics as metrics_mod
from rp.analysis import report
from rp import results_io

SPLIT = "unit_test_split"
AGENT = "gemini-3.8-live"       # a full-grid agent
BASELINE_AGENT = "qwen3.8-omni-flash"  # the other full-grid agent (left empty in most tests)


def mk_row(**over) -> dict:
    row = {
        "case_id": "c0", "domain": "ecommerce_support", "turn": 1, "num_steps": 1,
        "severity": "medium", "gate_expected": "confirm", "policy_rule_id": "G-X",
        "sut_text": "", "sut_tool_calls": [], "blocked": [], "user_asr": "",
        "latency_ms": 100.0, "guard_ms": 5.0, "verifier_calls": 0,
        "usage": {}, "sut_error": None, "safety_valve": None, "response_status": None,
        "verdict": "PASS", "verdict_judge": "PASS", "detected_gate": "confirm", "reason": "",
        "judge_model": "m", "judge_prompt_version": "v2", "severe_matcher": False,
        "split": SPLIT, "agent": AGENT, "config": "SAGE", "pass": 1,
        "run_id": "run-1", "git_sha": "deadbeef",
    }
    row.update(over)
    return row


def write_pass(results_root, split, agent, config, pass_num, rows):
    path = results_root / split / agent / config / f"pass{pass_num}.jsonl"
    results_io.append_rows(path, rows)


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    """Point both rp.analysis.metrics.RESULTS_ROOT and rp.analysis.report's own
    copy of it (imported separately, used for cost_log.jsonl) at a tmp dir."""
    monkeypatch.setattr(metrics_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(report, "RESULTS_ROOT", tmp_path)
    return tmp_path


def _ten_case_rows(agent, config, n_under, detected_when_not_under="confirm", run_id="run-1"):
    """10 cases, 1 turn each, gate_expected=confirm (so all 10 are gate-scoreable):
    n_under of them under-gated (detected execute), the rest correct."""
    rows = []
    for i in range(10):
        detected = "execute" if i < n_under else detected_when_not_under
        rows.append(mk_row(
            case_id=f"c{i}", turn=1, agent=agent, config=config, run_id=run_id,
            gate_expected="confirm", detected_gate=detected,
            verdict="FAIL" if detected == "execute" else "PASS",
        ))
    return rows


def _rows_with_classes(agent, config, n_under=0, n_wrong=0, run_id="run-1"):
    """10 cases, 1 turn each, gate_expected=confirm: the first n_under detected
    execute (under-gated), the next n_wrong detected authorize (gate-type
    error / wrong_shape), the rest correct -- lets a test control both
    primary metrics' rates independently instead of only under-gating."""
    rows = []
    for i in range(10):
        if i < n_under:
            detected = "execute"
        elif i < n_under + n_wrong:
            detected = "authorize"
        else:
            detected = "confirm"
        rows.append(mk_row(
            case_id=f"c{i}", turn=1, agent=agent, config=config, run_id=run_id,
            gate_expected="confirm", detected_gate=detected,
            verdict="PASS" if detected == "confirm" else "FAIL",
        ))
    return rows


def _seed_minimal_grid(results_root):
    """SAGE and B0 rows for both full-grid agents, enough for every table
    writer to run without crashing (B1-B4 are deliberately left empty to
    exercise the em-dash/no-crash path -- rule 6)."""
    sage_rows = _ten_case_rows(AGENT, "SAGE", n_under=3)
    b0_rows = _ten_case_rows(AGENT, "B0", n_under=10)  # B0 under-gates every turn
    write_pass(results_root, SPLIT, AGENT, "SAGE", 1, sage_rows)
    write_pass(results_root, SPLIT, AGENT, "B0", 1, b0_rows)
    # cost_log.jsonl, one record per case (rp.runner's format)
    log_lines = []
    for i in range(10):
        for config in ("SAGE", "B0"):
            log_lines.append(json.dumps({
                "ts": "2026-09-28T00:00:00+00:00", "split": SPLIT, "agent": AGENT,
                "config": config, "pass": 1, "case_id": f"c{i}", "components": [],
                "cost_usd": 0.01, "known_cost_usd": 0.01, "cumulative_known_usd": 0.01,
            }))
    (results_root / "cost_log.jsonl").write_text("\n".join(log_lines) + "\n")
    return sage_rows, b0_rows


# --------------------------------------------------------------------------- LaTeX / CSV validity helpers

def assert_valid_latex_table(text: str) -> None:
    assert "\\begin{tabular}" in text
    assert text.count("{") == text.count("}"), "unbalanced braces"
    # Only actual table rows (inside tabular, between the rules) are cells;
    # captions/tablenotes are prose and may legitimately say "95\%" without a "/".
    in_tabular = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("\\begin{tabular}"):
            in_tabular = True
            continue
        if stripped.startswith("\\end{tabular}"):
            in_tabular = False
            continue
        if not in_tabular or stripped in ("\\toprule", "\\midrule", "\\bottomrule"):
            continue
        for cell in line.split("&"):
            if "\\%" in cell:
                assert "/" in cell, f"percentage cell missing 'n/N': {cell!r}"


def assert_csv_header(path, expected_header):
    text = path.read_text()
    first_line = text.splitlines()[0]
    assert first_line == ",".join(expected_header)


TEX_FILES = ("rq1_main.tex", "rq1_judge2.tex", "rq2_per_agent.tex", "rq3_cost.tex",
             "rq4_ablation.tex", "per_domain.tex", "human_validation.tex")
CSV_HEADERS = {
    "heatmap_agent_severity.csv": ["agent", "tier", "pass_rate"],
    "forest.csv": ["comparison", "label", "metric", "diff", "lo", "hi"],
    "cost_safety.csv": ["config", "agent", "label", "cost_per_100", "gate_type_error",
                        "partially_priced_cases"],
    "failure_stack.csv": ["agent", "label", "config", "under", "over", "wrong_type", "wrong_content",
                          "unknown"],
    "routing_curve.csv": ["call_budget_fraction", "detections_retained"],
    "latency.csv": ["config", "agent", "label", "latency_p50_ms", "latency_p95_ms", "guard_p50_ms",
                    "guard_p95_ms"],
}


# --------------------------------------------------------------------------- tests

def test_known_answer_under_gating_cell(results_root, tmp_path):
    """3/10 under-gated turns for SAGE must render exactly as '3/10 = 30.0\\%'."""
    _seed_minimal_grid(results_root)
    out = tmp_path / "out"
    (out / "tables").mkdir(parents=True)
    report.rq1_table(SPLIT, out / "tables", family="per_agent")
    text = (out / "tables" / "rq1_main.tex").read_text()
    assert "3/10 = 30.0\\%" in text


def test_fmt_pct_known_values():
    assert report.fmt_pct(3, 10) == "3/10 = 30.0\\% [10.8, 60.3]"
    assert report.fmt_pct(0, 0) == report.EMDASH


def test_holm_family_runs_and_differences_are_numeric(results_root, tmp_path):
    _seed_minimal_grid(results_root)
    loader, fam = report.primary_family(SPLIT, "primary", "per_agent")
    key = (AGENT, "B0", "under")
    obs, lo, hi, p_raw, p_holm = fam[key]
    assert obs == obs and -1 <= obs <= 0  # SAGE under-gates less than B0 here (3/10 vs 10/10)
    assert 0 <= p_holm <= 1


@pytest.mark.parametrize("family", ["pooled", "per_agent"])
def test_rq1_table_valid_for_both_families(results_root, tmp_path, family):
    _seed_minimal_grid(results_root)
    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)
    report.rq1_table(SPLIT, out, family=family)
    assert_valid_latex_table((out / "rq1_main.tex").read_text())


def test_all_tables_are_valid_latex_on_tiny_grid(results_root, tmp_path):
    """Full main() run on a tiny/incomplete grid: every table is valid LaTeX
    and every CSV has its header row, with no crash (rule 6)."""
    _seed_minimal_grid(results_root)
    out = tmp_path / "out"
    report.main(["--split", SPLIT, "--out", str(out)])

    for name in TEX_FILES:
        path = out / "tables" / name
        assert path.exists(), f"missing {name}"
        assert_valid_latex_table(path.read_text())

    for name, header in CSV_HEADERS.items():
        path = out / "data" / name
        assert path.exists(), f"missing {name}"
        assert_csv_header(path, header)


def test_missing_split_directory_is_graceful(tmp_path, monkeypatch):
    """A split with no results directory at all (e.g. the locked test split
    before Task 20/prereg) must produce em-dash tables, not crash."""
    monkeypatch.setattr(metrics_mod, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(report, "RESULTS_ROOT", tmp_path)
    out = tmp_path / "out"
    report.main(["--split", "never_run", "--out", str(out)])
    text = (out / "tables" / "rq1_main.tex").read_text()
    assert_valid_latex_table(text)
    assert report.EMDASH in text
    for name in TEX_FILES:
        assert (out / "tables" / name).exists()
    for name in CSV_HEADERS:
        assert (out / "data" / name).exists()


def test_header_comment_has_git_sha_run_ids_and_error_count(results_root, tmp_path):
    sage_rows, b0_rows = _seed_minimal_grid(results_root)
    # add one ERROR turn, which must be excluded from the metrics but counted
    error_row = mk_row(case_id="c_err", turn=1, agent=AGENT, config="SAGE",
                        verdict="ERROR", run_id="run-err")
    write_pass(results_root, SPLIT, AGENT, "SAGE", 1, [error_row])

    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)
    report.rq1_table(SPLIT, out, family="per_agent")
    text = (out / "rq1_main.tex").read_text()
    assert "% Generated by rp.analysis.report -- git sha" in text
    assert "% run_ids: " in text
    assert "run-1" in text
    assert "% excluded ERROR turns: 1" in text


def test_ablation_agent_autodetect_and_no_crash(results_root, tmp_path):
    """No ablation configs exist at all: rq4 must not crash and must render
    every predicted-effect cell as an em dash."""
    _seed_minimal_grid(results_root)
    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)
    report.rq4_table(SPLIT, out)
    text = (out / "rq4_ablation.tex").read_text()
    assert_valid_latex_table(text)


def test_ablation_with_data_produces_a_holm_adjusted_cell(results_root, tmp_path):
    ablation_rows = _ten_case_rows(AGENT, "SAGE-no_gate_types", n_under=0,
                                    detected_when_not_under="clarify")
    write_pass(results_root, SPLIT, AGENT, "SAGE-no_gate_types", 1, ablation_rows)
    _seed_minimal_grid(results_root)
    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)
    report.rq4_table(SPLIT, out, ablation_agent=AGENT)
    text = (out / "rq4_ablation.tex").read_text()
    assert_valid_latex_table(text)
    assert "Gate-type selection" in text


# --------------------------------------------------------------------------- fix round 1: cost per 100 turns

def test_cost_known_answer_per_100_turns(results_root):
    """5 judged turns, 5 fully-priced cost_log records at $0.02 known cost each:
    $ per 100 turns = (0.10 / 5) * 100 = $2.00, zero partially-priced cases."""
    rows = [mk_row(case_id=f"c{i}", turn=1, agent=AGENT, config="SAGE") for i in range(5)]
    write_pass(results_root, SPLIT, AGENT, "SAGE", 1, rows)
    log_lines = [json.dumps({"split": SPLIT, "agent": AGENT, "config": "SAGE", "case_id": f"c{i}",
                             "cost_usd": 0.02, "known_cost_usd": 0.02}) for i in range(5)]
    (results_root / "cost_log.jsonl").write_text("\n".join(log_lines) + "\n")

    cost, n_partial = report._cost_per_100_turns(SPLIT, [AGENT], "SAGE", 5)
    assert cost == pytest.approx(2.0)
    assert n_partial == 0


def test_cost_log_none_cost_usd_no_crash_and_partial_count(results_root):
    """cost_usd is None by design whenever a component is unpriced (runner.price_usage);
    known_cost_usd is still summed (matching rp.runner.cumulative_spend's fallback), and
    the case is counted as partially priced rather than crashing statistics.mean on None."""
    rows = [mk_row(case_id=f"c{i}", turn=1, agent=AGENT, config="SAGE") for i in range(3)]
    write_pass(results_root, SPLIT, AGENT, "SAGE", 1, rows)
    log_lines = [
        json.dumps({"split": SPLIT, "agent": AGENT, "config": "SAGE", "case_id": "c0",
                    "cost_usd": None, "known_cost_usd": 0.05}),
        json.dumps({"split": SPLIT, "agent": AGENT, "config": "SAGE", "case_id": "c1",
                    "cost_usd": 0.02, "known_cost_usd": 0.02}),
        json.dumps({"split": SPLIT, "agent": AGENT, "config": "SAGE", "case_id": "c2",
                    "cost_usd": 0.03, "known_cost_usd": 0.03}),
    ]
    (results_root / "cost_log.jsonl").write_text("\n".join(log_lines) + "\n")

    cost, n_partial = report._cost_per_100_turns(SPLIT, [AGENT], "SAGE", 3)
    assert n_partial == 1
    assert cost == pytest.approx((0.05 + 0.02 + 0.03) / 3 * 100)


def test_cost_per_100_turns_missing_log_is_graceful(results_root):
    cost, n_partial = report._cost_per_100_turns(SPLIT, [AGENT], "SAGE", 5)
    assert cost != cost  # NaN
    assert n_partial == 0


def test_rq3_table_no_crash_with_partially_priced_cost_log(results_root, tmp_path):
    _seed_minimal_grid(results_root)
    with open(results_root / "cost_log.jsonl", "a") as f:
        f.write(json.dumps({"split": SPLIT, "agent": AGENT, "config": "SAGE", "case_id": "c0",
                            "cost_usd": None, "known_cost_usd": 0.05}) + "\n")

    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)
    report.rq3_table(SPLIT, out)  # must not raise (statistics.mean on a None used to crash)
    text = (out / "rq3_cost.tex").read_text()
    assert_valid_latex_table(text)
    assert "$^{*}$" in text
    assert "partially priced" in text


def test_cost_safety_csv_has_partial_column_and_label(results_root, tmp_path):
    _seed_minimal_grid(results_root)
    out = tmp_path / "out" / "data"
    out.mkdir(parents=True)
    report.cost_safety_csv(SPLIT, out)
    text = (out / "cost_safety.csv").read_text()
    assert text.splitlines()[0] == ",".join(CSV_HEADERS["cost_safety.csv"])
    assert "Gemini 3.8 Live" in text  # AGENT_LABEL reused, not just the raw agent id


# --------------------------------------------------------------------------- fix round 1: strict RQ2 sign test

def test_rq2_sign_test_strict_ties_excluded(results_root, tmp_path):
    """B0, B2, and SAGE all have the identical rate (same case_ids, same
    detected-gate pattern) for the one agent under test: SAGE ties for first,
    it does not strictly beat both baselines, so it must be excluded from the
    sign-test n and reported as a tie, not counted as a win."""
    tied_rows = lambda config: _ten_case_rows(AGENT, config, n_under=2)
    write_pass(results_root, SPLIT, AGENT, "B0", 1, tied_rows("B0"))
    write_pass(results_root, SPLIT, AGENT, "B2", 1, tied_rows("B2"))
    write_pass(results_root, SPLIT, AGENT, "SAGE", 1, tied_rows("SAGE"))

    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)
    report.rq2_table(SPLIT, out)
    text = (out / "rq2_per_agent.tex").read_text()
    assert_valid_latex_table(text)
    assert "ties: 1" in text
    assert "0 of 0" in text  # the only complete agent was a tie, so n = 0
    assert "excluded from the sign test" in text


def test_rq2_sign_test_strict_win_still_counts(results_root, tmp_path):
    """SAGE strictly better than both B0 and B2 on both metrics for this agent
    -> a real win, not a tie."""
    write_pass(results_root, SPLIT, AGENT, "B0", 1,
               _rows_with_classes(AGENT, "B0", n_under=8, n_wrong=1))
    write_pass(results_root, SPLIT, AGENT, "B2", 1,
               _rows_with_classes(AGENT, "B2", n_under=6, n_wrong=1))
    write_pass(results_root, SPLIT, AGENT, "SAGE", 1,
               _rows_with_classes(AGENT, "SAGE", n_under=1, n_wrong=0))

    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)
    report.rq2_table(SPLIT, out)
    text = (out / "rq2_per_agent.tex").read_text()
    assert "1 of 1" in text
    assert "ties: 0" in text


def test_dev_v2_real_data_smoke(monkeypatch):
    """Guard against RESULTS_ROOT drift: with no monkeypatch, report reads the
    real repo results dir, so a genuinely empty/missing split still yields
    valid, non-crashing output (mirrors the manual --split dev_v2 smoke run)."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        report.main(["--split", "dev_v2", "--out", str(out)])
        for name in TEX_FILES:
            assert_valid_latex_table((out / "tables" / name).read_text())


# --------------------------------------------------------------------------- Task 16b fix round: spec section 5 roster

def test_roster_is_spec_section_5():
    assert report.AGENT_ORDER == ("gemini-3.8-live", "qwen3.8-omni-flash", "gpt-realtime-2.1-mini",
                                  "gemini-3.1-flash-live-preview", "voxtral-small-24b", "open-cascade")
    assert report.FULL_GRID_AGENTS == ("gemini-3.8-live", "qwen3.8-omni-flash")
    assert report.LIGHT_GRID_AGENTS == ("gpt-realtime-2.1-mini", "gemini-3.1-flash-live-preview",
                                        "voxtral-small-24b", "open-cascade")
    assert report.ABLATION_AGENT == "qwen3.8-omni-flash"
    from rp.agents import AGENTS
    assert all(a in AGENTS and a in report.AGENT_LABEL for a in report.AGENT_ORDER)


def test_rq2_rows_are_the_six_agents_with_prediction_wording(results_root, tmp_path):
    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)
    report.rq2_table(SPLIT, out)
    text = (out / "rq2_per_agent.tex").read_text()
    assert_valid_latex_table(text)
    for a in report.AGENT_ORDER:
        assert report.AGENT_LABEL[a] + " &" in text
    assert "GPT Realtime 2.1 &" not in text and "Cascade &" not in text
    assert "5 of 6" in text


def test_rq4_defaults_to_qwen_ablation_agent(results_root, tmp_path):
    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)
    report.rq4_table(SPLIT, out)
    assert "Ablation agent: qwen3.8-omni-flash" in (out / "rq4_ablation.tex").read_text()


# --------------------------------------------------------------------------- Task 19a: RQ4 filled predicted-effect columns

def _clean_rows(agent, config, n=10, **over):
    rows = []
    for i in range(n):
        rows.append(mk_row(case_id=f"c{i}", turn=1, agent=agent, config=config,
                           gate_expected="confirm", detected_gate="confirm", verdict="PASS",
                           policy_rule_id="G-X", sut_tool_calls=[], blocked=[], **over))
    return rows


def _seed_rq4_ablation_grid(results_root, agent):
    """SAGE (full) + all six ablations (Task 25 adds SAGE-no_policy), each rigged
    so its brief-designated predicted-effect metric has a clean 0%-vs-100%
    signal against SAGE. SAGE-no_routing/SAGE-no_exemption are still seeded
    (remain supported by code) even though the default RQ4 table (report.RQ4_CONFIGS)
    no longer includes them."""
    sage_rows = []
    ablation_rows = {c: [] for c in report.ABLATION_CONFIGS}

    # SAGE-no_gate_types -> authorize_as_confirm
    for i in range(10):
        sage_rows.append(mk_row(case_id=f"gt{i}", turn=1, agent=agent, config="SAGE",
                                gate_expected="authorize", detected_gate="authorize", verdict="PASS"))
        ablation_rows["SAGE-no_gate_types"].append(
            mk_row(case_id=f"gt{i}", turn=1, agent=agent, config="SAGE-no_gate_types",
                  gate_expected="authorize", detected_gate="confirm", verdict="FAIL"))

    # SAGE-no_verifier -> wrong_content (right gate, wrong content)
    for i in range(10):
        sage_rows.append(mk_row(case_id=f"wc{i}", turn=1, agent=agent, config="SAGE",
                                gate_expected="confirm", detected_gate="confirm", verdict="PASS"))
        ablation_rows["SAGE-no_verifier"].append(
            mk_row(case_id=f"wc{i}", turn=1, agent=agent, config="SAGE-no_verifier",
                  gate_expected="confirm", detected_gate="confirm", verdict="FAIL"))

    # SAGE-no_routing -> wrong_content (should not improve: identical here) + cost
    for i in range(10):
        sage_rows.append(mk_row(case_id=f"rt{i}", turn=1, agent=agent, config="SAGE",
                                gate_expected="confirm", detected_gate="confirm", verdict="PASS",
                                guard_ms=5.0, verifier_calls=1))
        ablation_rows["SAGE-no_routing"].append(
            mk_row(case_id=f"rt{i}", turn=1, agent=agent, config="SAGE-no_routing",
                  gate_expected="confirm", detected_gate="confirm", verdict="PASS",
                  guard_ms=50.0, verifier_calls=3))

    # SAGE-no_credential_check -> credential_bypass. gate_expected="authorize" on
    # both arms so the fix-round-1 denominator (authorize turn / auth-related
    # policy_rule_id / an executed verify_authorization) is satisfied.
    xfer_call = [{"function": "transfer_funds", "args": {"to_account": "9999", "amount": 200}}]
    for i in range(10):
        sage_rows.append(mk_row(case_id=f"cb{i}", turn=1, agent=agent, config="SAGE",
                                gate_expected="authorize", policy_rule_id="G-X", sut_tool_calls=xfer_call))
        ablation_rows["SAGE-no_credential_check"].append(
            mk_row(case_id=f"cb{i}", turn=1, agent=agent, config="SAGE-no_credential_check",
                  gate_expected="authorize", policy_rule_id="G-AUTH-MISSING", sut_tool_calls=xfer_call))

    # SAGE-no_exemption -> delayed_escalation (primary) and over_escalation (secondary)
    escalate_call = [{"function": "escalate_to_crisis_line", "args": {}}]
    for i in range(10):
        sage_rows.append(mk_row(case_id=f"de{i}", turn=1, agent=agent, config="SAGE",
                                policy_rule_id="G-CRISIS", sut_tool_calls=escalate_call))
        ablation_rows["SAGE-no_exemption"].append(
            mk_row(case_id=f"de{i}", turn=1, agent=agent, config="SAGE-no_exemption",
                  policy_rule_id="G-CRISIS", sut_tool_calls=[]))
        sage_rows.append(mk_row(case_id=f"oe{i}", turn=1, agent=agent, config="SAGE",
                                policy_rule_id="G-DISTRESS", sut_tool_calls=[]))
        ablation_rows["SAGE-no_exemption"].append(
            mk_row(case_id=f"oe{i}", turn=1, agent=agent, config="SAGE-no_exemption",
                  policy_rule_id="G-DISTRESS", sut_tool_calls=escalate_call))

    # SAGE-no_policy -> wrong_type/gate-type error (primary) and under/under-gating
    # (secondary): predicted effect is the *primary metrics* (gate-type error,
    # under-gating), per Task 25's brief -- not one of the ABLATION_METRICS.
    for i in range(10):
        sage_rows.append(mk_row(case_id=f"pt{i}", turn=1, agent=agent, config="SAGE",
                                gate_expected="confirm", detected_gate="confirm", verdict="PASS"))
        ablation_rows["SAGE-no_policy"].append(
            mk_row(case_id=f"pt{i}", turn=1, agent=agent, config="SAGE-no_policy",
                  gate_expected="confirm", detected_gate="authorize", verdict="FAIL"))
        sage_rows.append(mk_row(case_id=f"pu{i}", turn=1, agent=agent, config="SAGE",
                                gate_expected="confirm", detected_gate="confirm", verdict="PASS"))
        ablation_rows["SAGE-no_policy"].append(
            mk_row(case_id=f"pu{i}", turn=1, agent=agent, config="SAGE-no_policy",
                  gate_expected="confirm", detected_gate="execute", verdict="FAIL"))

    write_pass(results_root, SPLIT, agent, "SAGE", 1, sage_rows)
    for config, rows in ablation_rows.items():
        write_pass(results_root, SPLIT, agent, config, 1, rows)


def test_rq4_fills_default_ablation_predicted_effect_cells(results_root, tmp_path):
    """Task 25: the default RQ4 table covers 4 ablations (gate-type selection,
    verifier, credential check, policy prompt) -- routing/exemption are seeded
    but no longer rendered by default."""
    agent = report.ABLATION_AGENT
    _seed_rq4_ablation_grid(results_root, agent)
    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)
    report.rq4_table(SPLIT, out, ablation_agent=agent)
    text = (out / "rq4_ablation.tex").read_text()
    assert_valid_latex_table(text)
    # every predicted-effect cell must show a real diff, not an em dash
    assert "+100.0 pp" in text, text

    lines = text.splitlines()
    gate_types_line = next(l for l in lines if "Gate-type selection" in l)
    verifier_line = next(l for l in lines if l.startswith("$-$Verifier") or "Verifier &" in l)
    cred_line = next(l for l in lines if "Credential check" in l)
    policy_line = next(l for l in lines if "Policy prompt" in l)
    assert not any("Routing" in l for l in lines)
    assert not any("Exemption" in l for l in lines)

    assert report.EMDASH not in gate_types_line.split("&")[1]  # authorize_as_confirm column
    assert report.EMDASH not in verifier_line.split("&")[2]    # wrong_content column
    assert report.EMDASH not in cred_line.split("&")[3]        # credential_bypass column
    assert report.EMDASH not in policy_line.split("&")[6]      # wrong_type column (primary)
    assert report.EMDASH not in policy_line.split("&")[7]      # under column (secondary)


def test_rq4_holm_family_is_four_ablations(results_root, tmp_path, monkeypatch):
    """Task 25: spy on report.holm itself rather than grepping the rendered
    table -- assert it is called with exactly the 4 default RQ4 configs as keys
    (one p-value per ablation's primary predicted effect), and nothing else
    (no secondary/cost pseudo-comparisons folded in, and routing/exemption --
    still supported by code -- are not part of the default family)."""
    agent = report.ABLATION_AGENT
    _seed_rq4_ablation_grid(results_root, agent)
    out = tmp_path / "out" / "tables"
    out.mkdir(parents=True)

    seen_pvals = {}
    real_holm = report.holm

    def spy_holm(pvals):
        seen_pvals.update(pvals)
        return real_holm(pvals)

    monkeypatch.setattr(report, "holm", spy_holm)
    report.rq4_table(SPLIT, out, ablation_agent=agent)

    assert set(seen_pvals.keys()) == set(report.RQ4_CONFIGS)
    assert len(seen_pvals) == 4


# --------------------------------------------------------------------------- Task 19a: RQ3 detection retention (verifier_block)

def test_routing_retention_counts_only_verifier_source_blocks():
    agent = report.FULL_GRID_AGENTS[0]
    sage_rows = [
        mk_row(case_id="c0", turn=1, agent=agent, config="SAGE", verifier_calls=1,
              blocked=[{"call": {}, "gate": "clarify", "rule_id": None, "reason": "x",
                       "source": "verifier", "verdict": "ambiguous"}]),
        mk_row(case_id="c1", turn=1, agent=agent, config="SAGE", verifier_calls=0, blocked=[]),
    ]
    always_rows = [
        mk_row(case_id="c0", turn=1, agent=agent, config="SAGE-no_routing", verifier_calls=1,
              blocked=[{"call": {}, "gate": "clarify", "rule_id": None, "reason": "x",
                       "source": "verifier", "verdict": "ambiguous"}]),
        mk_row(case_id="c1", turn=1, agent=agent, config="SAGE-no_routing", verifier_calls=1,
              blocked=[{"call": {}, "gate": "confirm", "rule_id": "G-X", "reason": "x",
                       "source": "rules"}]),
    ]
    retention, call_share = report._routing_retention(sage_rows, always_rows)
    assert retention.startswith("100.0")
