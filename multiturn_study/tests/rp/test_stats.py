import json
import random

import pytest

from rp.analysis import metrics as metrics_mod
from rp.analysis.metrics import METRICS, case_table, load_runs, turn_flags
from rp.analysis.stats import cluster_bootstrap_diff, glmm_or, holm, perm_test, ratio, wilson


def rows(config, p_under, n_cases=60, turns=3, passes=3, seed=0):
    r = random.Random(seed); out = []
    for c in range(n_cases):
        for k in range(passes):
            for t in range(turns):
                u = r.random() < p_under
                out.append({"case_id": f"c{c}", "pass": k, "turn": t, "config": config, "gate_expected": "confirm",
                            "detected_gate": "execute" if u else "confirm", "verdict": "FAIL" if u else "PASS",
                            "severity": "medium"})
    return out


def test_bootstrap_detects_large_effect():
    d, lo, hi = cluster_bootstrap_diff(rows("B0", .30, seed=1), rows("SAGE", .05, seed=2), "under", n=2000, seed=3)
    assert hi < 0 and -0.35 < d < -0.15


def test_perm_null_is_not_significant():
    assert perm_test(rows("B0", .2, seed=4), rows("B1", .2, seed=5), "under", n=2000, seed=6) > 0.05


def test_perm_detects_large_effect():
    p = perm_test(rows("B0", .30, seed=1), rows("SAGE", .05, seed=2), "under", n=2000, seed=6)
    assert p < 0.01


def test_holm():
    adj = holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert adj == {"a": 0.03, "b": 0.04, "c": 0.04}


def test_holm_monotone_and_capped():
    # a raw ordering that would be non-monotone without the running max, and
    # a value that must be capped at 1.0
    adj = holm({"a": 0.5, "b": 0.3, "c": 0.9})
    vals_in_sorted_order = [adj["b"], adj["a"], adj["c"]]
    assert vals_in_sorted_order == sorted(vals_in_sorted_order)
    assert all(0 <= v <= 1 for v in adj.values())
    assert adj["c"] == 1.0


def test_wilson_bounds():
    lo, hi = wilson(5, 33)
    assert 0.06 < lo < 0.07 and 0.30 < hi < 0.32


def test_wilson_zero_n_is_nan():
    lo, hi = wilson(0, 0)
    assert lo != lo and hi != hi  # NaN != NaN


# ---- metrics.py --------------------------------------------------------

def test_turn_flags_categories():
    correct = {"gate_expected": "confirm", "detected_gate": "confirm", "verdict": "PASS", "severity": "low"}
    under = {"gate_expected": "confirm", "detected_gate": "execute", "verdict": "FAIL", "severity": "low"}
    over = {"gate_expected": "execute", "detected_gate": "confirm", "verdict": "FAIL", "severity": "low"}
    wrong_type = {"gate_expected": "confirm", "detected_gate": "authorize", "verdict": "FAIL", "severity": "low"}
    unscoreable = {"gate_expected": "confirm", "detected_gate": "unknown", "verdict": "FAIL", "severity": "low"}

    assert turn_flags(correct)["pass"] is True
    assert turn_flags(correct)["scoreable"] is True

    assert turn_flags(under)["under"] is True
    assert turn_flags(under)["over"] is False

    assert turn_flags(over)["over"] is True
    assert turn_flags(over)["under"] is False

    assert turn_flags(wrong_type)["wrong_type"] is True
    assert turn_flags(wrong_type)["under"] is False
    assert turn_flags(wrong_type)["over"] is False

    assert turn_flags(unscoreable)["scoreable"] is False


def test_metrics_tuple():
    assert METRICS == ("task_success", "severe", "under", "over", "burden", "wrong_type", "swl")


def test_case_table_shape_and_ratio():
    rs = rows("B0", 0.5, n_cases=3, turns=2, passes=2, seed=7)
    ct = case_table(rs)
    # one entry per (case_id, pass)
    assert set(ct.keys()) == {(f"c{c}", k) for c in range(3) for k in range(2)}
    for metrics in ct.values():
        for m in METRICS:
            assert m in metrics
    # ratio over all cases/passes should equal the plain pooled fraction
    all_cases = [c for c, _p in ct]
    num = sum(v["task_success"][0] for v in ct.values())
    den = sum(v["task_success"][1] for v in ct.values())
    assert ratio(ct, all_cases, "task_success") == pytest.approx(num / den)


def test_ratio_counts_repeated_keys_once_per_occurrence():
    rs = rows("B0", 0.5, n_cases=2, turns=2, passes=1, seed=9)
    ct = case_table(rs)
    single = ratio(ct, ["c0"], "task_success")
    doubled = ratio(ct, ["c0", "c0"], "task_success")
    assert single == pytest.approx(doubled)


# ---- glmm_or must not crash on degenerate data -------------------------

def test_glmm_or_all_zero_returns_nan_with_warning():
    rs = []
    for cfg in ("B0", "SAGE"):
        for c in range(5):
            rs.append({"case_id": f"c{c}", "pass": 0, "turn": 0, "config": cfg,
                        "gate_expected": "confirm", "detected_gate": "confirm",
                        "verdict": "PASS", "severity": "low"})
    with pytest.warns(UserWarning):
        out = glmm_or(rs, "under", ref_config="B0")
    assert set(out) == {"B0", "SAGE"}
    for val in out.values():
        assert all(v != v for v in val)  # every element is NaN


# ---- load_runs: judge selects FILES, never fields ----------------------

def _write_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _make_run_tree(tmp_path):
    base = tmp_path / "x" / "agentX" / "B0"
    primary_row = {"case_id": "c0", "turn": 0, "pass": 1, "config": "B0",
                   "gate_expected": "confirm", "detected_gate": "confirm",
                   "verdict": "FAIL", "verdict_judge": "SEVERE", "severity": "low"}
    judge2_row = {"case_id": "c0", "turn": 0, "pass": 1, "config": "B0",
                  "gate_expected": "confirm", "detected_gate": "confirm",
                  "verdict": "SEVERE", "verdict_judge": "SEVERE", "severity": "low"}
    _write_row(base / "pass1.jsonl", primary_row)
    _write_row(base / "pass1.judge2.jsonl", judge2_row)
    return base


def test_load_runs_primary_ignores_judge2_files(tmp_path, monkeypatch):
    _make_run_tree(tmp_path)
    monkeypatch.setattr(metrics_mod, "RESULTS_ROOT", tmp_path)
    rows_out = load_runs("x", agent=None, config="B0", judge="primary")
    assert len(rows_out) == 1
    assert rows_out[0]["verdict"] == "FAIL"


def test_load_runs_judge2_reads_only_judge2_files(tmp_path, monkeypatch):
    _make_run_tree(tmp_path)
    monkeypatch.setattr(metrics_mod, "RESULTS_ROOT", tmp_path)
    rows_out = load_runs("x", agent=None, config="B0", judge="judge2")
    assert len(rows_out) == 1
    assert rows_out[0]["verdict"] == "SEVERE"


def test_load_runs_never_uses_verdict_judge_as_verdict(tmp_path, monkeypatch):
    _make_run_tree(tmp_path)
    monkeypatch.setattr(metrics_mod, "RESULTS_ROOT", tmp_path)
    rows_out = load_runs("x", agent=None, config="B0", judge="primary")
    row = rows_out[0]
    assert row["verdict"] == "FAIL" and row["verdict_judge"] == "SEVERE"
    assert turn_flags(row)["severe"] is False


# ---- glmm_or recovers the sign of a real effect -------------------------

def test_glmm_recovers_effect_sign():
    a = rows("B0", 0.30, n_cases=60, turns=3, passes=3, seed=1)
    b = rows("SAGE", 0.05, n_cases=60, turns=3, passes=3, seed=2)
    out = glmm_or(a + b, "under", ref_config="B0")
    log_or, lo, hi = out["SAGE"]
    assert log_or < 0
    assert hi < 0


# ---- controller fix-round-1: GLMM per-run random effect, separation, convergence ----

def _row(c, k, t, cfg, exp, det, v="PASS"):
    return {"case_id": c, "pass": k, "turn": t, "config": cfg, "gate_expected": exp,
            "detected_gate": det, "verdict": v, "severity": "medium"}


def test_ratio_is_pooled_sum_not_mean_of_case_ratios():
    # c0: 1 scoreable turn, 1 under; c1: 9 scoreable turns, 0 under.
    # Pooled: 1/10 = 0.1. A (wrong) mean-of-case-ratios would give (1/1 + 0/9)/2 = 0.5.
    a = [_row("c0", 0, 0, "A", "confirm", "execute", "FAIL"),
         _row("c0", 0, 1, "A", "confirm", "unknown", "FAIL")] + \
        [_row("c1", 0, t, "A", "confirm", "confirm") for t in range(9)]
    ct = case_table(a)
    assert ratio(ct, ["c0", "c1"], "under") == pytest.approx(0.1)


def test_bootstrap_resample_multiset_weighting():
    a = [_row("c0", 0, 0, "A", "confirm", "execute", "FAIL"),
         _row("c0", 0, 1, "A", "confirm", "unknown", "FAIL")] + \
        [_row("c1", 0, t, "A", "confirm", "confirm") for t in range(9)]
    ct = case_table(a)
    # c0 drawn twice, c1 once: (1+1)/(1+1+9) = 2/11.
    assert ratio(ct, ["c0", "c0", "c1"], "under") == pytest.approx(2 / 11)


def test_glmm_or_separation_in_one_config_returns_nan():
    # B0 has real variation; SAGE is perfectly separated (always correct).
    a = rows("B0", 0.30, n_cases=20, turns=3, passes=1, seed=1)
    b = rows("SAGE", 0.0, n_cases=20, turns=3, passes=1, seed=2)
    with pytest.warns(UserWarning):
        out = glmm_or(a + b, "under", ref_config="B0")
    assert set(out) == {"B0", "SAGE"}
    for val in out.values():
        assert all(v != v for v in val)


def test_glmm_or_pass_key_includes_config():
    """pass_key must be keyed by (case, config, pass): a case's SAGE-pass-1 and
    B0-pass-1 are independent runs, not the same random-effect draw. With a
    per-run random effect (sd 1.5) layered on a per-case effect (sd 1.0) and a
    true config log-odds effect of -0.5, the fixed effect should still recover
    close to -0.5."""
    import math

    def sim(seed, run_sd=1.5, case_sd=1.0, n=40):
        r = random.Random(seed)
        out = []
        for c in range(n):
            cu = r.gauss(0, case_sd)
            for cfg, beta in (("B0", 0.0), ("SAGE", -0.5)):
                for k in range(3):
                    ru = r.gauss(0, run_sd)
                    for t in range(4):
                        p = 1 / (1 + math.exp(-(-1 + cu + beta + ru)))
                        u = r.random() < p
                        out.append(_row(f"c{c}", k, t, cfg, "confirm",
                                         "execute" if u else "confirm", "FAIL" if u else "PASS"))
        return out

    out = glmm_or(sim(1), "under", ref_config="B0")
    log_or, lo, hi = out["SAGE"]
    assert abs(log_or - (-0.5)) < 0.15


# ---- load_runs input validation -----------------------------------------

def test_load_runs_missing_split_raises_filenotfounderror(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics_mod, "RESULTS_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_runs("does_not_exist")


def test_load_runs_bad_judge_raises_valueerror(tmp_path, monkeypatch):
    _make_run_tree(tmp_path)
    monkeypatch.setattr(metrics_mod, "RESULTS_ROOT", tmp_path)
    with pytest.raises(ValueError):
        load_runs("x", config="B0", judge="bogus")


# ---- NaN propagation in bootstrap / perm test ---------------------------

def test_cluster_bootstrap_diff_nan_metric_returns_nan():
    a = [_row("c0", 0, 0, "A", "confirm", "unknown", "FAIL")]
    b = [_row("c0", 0, 0, "B", "confirm", "unknown", "FAIL")]
    d, lo, hi = cluster_bootstrap_diff(a, b, "under", n=50, seed=1)
    assert d != d and lo != lo and hi != hi


def test_perm_test_nan_metric_returns_nan():
    a = [_row("c0", 0, 0, "A", "confirm", "unknown", "FAIL")]
    b = [_row("c0", 0, 0, "B", "confirm", "unknown", "FAIL")]
    p = perm_test(a, b, "under", n=50, seed=1)
    assert p != p
