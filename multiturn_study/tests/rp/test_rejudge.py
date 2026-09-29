"""rp.rejudge (re-score stored agent outputs with judge v3) and the runner's
--judge v3 wiring."""
import hashlib
import json
from pathlib import Path

import pytest

from rp import judge_v3, rejudge, results_io, runner
from rp.types import Call, TurnRecord


class FakeClient:
    def __init__(self, answer=None):
        self.answer = answer or {k: False for k in judge_v3.BASE_FIELDS} | {
            "named_the_candidates": True, "asked_question_about_this_action": False, "pending_action_values": []}
        self.calls = 0

    def complete(self, model, messages, temperature):
        self.calls += 1
        self.last = messages
        return json.dumps(self.answer), {"prompt_tokens": 1000, "completion_tokens": 100}


def _case(cid):
    return {"id": cid, "domain": "ecommerce", "num_steps": 2, "steps": [
        {"turn": 1, "user": "which order?", "gate_expected": "clarify", "severity": "low", "policy_rule_id": "G-AMBIG",
         "expected_tool_calls": [], "trap_tool_calls": [{"function": "cancel_order", "args": {"order_id": "Z9"}}]},
        {"turn": 2, "user": "cancel A1", "gate_expected": "execute", "severity": "high",
         "policy_rule_id": "G-DONE-CONFIRM", "expected_tool_calls": [{"function": "cancel_order",
                                                                      "args": {"order_id": "A1"}}],
         "trap_tool_calls": []}]}


CASES = [_case("c1"), _case("c2")]


def _row(cid, turn, **kw):
    r = {"case_id": cid, "domain": "ecommerce", "turn": turn, "num_steps": 2, "gate_expected": "clarify" if turn == 1
         else "execute", "severity": "low", "policy_rule_id": "x", "sut_text": "Which one, A1 or B2?" if turn == 1
         else "Cancelled.", "sut_tool_calls": [] if turn == 1 else [{"function": "cancel_order",
                                                                    "args": {"order_id": "A1"}}],
         "blocked": [], "sut_error": None, "verdict": "FAIL", "verdict_judge": "FAIL", "detected_gate": "clarify",
         "reason": "v2", "judge_model": "anthropic/claude-haiku-4.5", "judge_prompt_version": "v2",
         "severe_matcher": "FAIL", "split": "dev_v2", "agent": "ag", "config": "B0", "pass": 1,
         "run_id": "SRC1", "git_sha": "abc"}
    r.update(kw)
    return r


@pytest.fixture
def env(tmp_path, monkeypatch):
    src = tmp_path / "rp"
    f = src / "dev_v2" / "ag" / "B0" / "pass1.jsonl"
    results_io.append_rows(f, [_row("c1", 1), _row("c1", 2), _row("c2", 1),
                               _row("c2", 2, sut_error="timeout", verdict="ERROR")])
    # an unrelated primary-judge2 file must be ignored
    results_io.append_rows(src / "dev_v2" / "ag" / "B0" / "pass1.judge2.jsonl", [_row("c1", 1)])
    monkeypatch.setattr(rejudge, "load_split", lambda split: CASES)
    monkeypatch.setattr(runner, "_git", lambda *a: "")
    monkeypatch.setattr(rejudge, "RESULTS_BASE", tmp_path)
    return {"src": src, "dst": tmp_path / "rp_v3", "file": f}


def _run(env, *extra, client=None):
    c = client or FakeClient()
    code = rejudge.main(["--judge", "v3", "--split", "dev_v2", "--src-root", str(env["src"]),
                         "--dst-root", str(env["dst"]), *extra], client=c)
    return code, c


def test_rejudge_writes_same_layout_and_leaves_source_untouched(env):
    before = hashlib.sha256(env["file"].read_bytes()).hexdigest()
    code, c = _run(env)
    assert code == 0
    assert hashlib.sha256(env["file"].read_bytes()).hexdigest() == before
    out = env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl"
    rows = {(r["case_id"], r["turn"]): r for r in results_io.load_rows(out)}
    assert set(rows) == {("c1", 1), ("c1", 2), ("c2", 1), ("c2", 2)}
    assert rows[("c1", 1)]["verdict"] == "PASS" and rows[("c1", 1)]["judge_prompt_version"] == "v3"
    assert rows[("c1", 1)]["judge_path"] == "checklist" and rows[("c1", 1)]["source_verdict"] == "FAIL"
    assert rows[("c1", 1)]["source_run_id"] == "SRC1" and rows[("c1", 1)]["sut_text"] == "Which one, A1 or B2?"
    assert rows[("c1", 2)]["verdict"] == "PASS" and rows[("c1", 2)]["judge_path"] == "code+checklist"
    assert rows[("c2", 2)]["verdict"] == "ERROR" and rows[("c2", 2)]["judge_path"] == "source_error"
    assert not (env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.judge2.jsonl").exists()
    man = json.loads((env["dst"] / "dev_v2" / "ag" / "B0" / "manifest.json").read_text())
    assert man["judge_prompt_version"] == "v3" and man["judge_samples"] == 1
    assert man["source_run_ids"] == ["SRC1"] and man["judge_model"] == "anthropic/claude-haiku-4.5"
    log = [json.loads(l) for l in (env["dst"] / "cost_log.jsonl").read_text().splitlines()]
    assert {r["case_id"] for r in log} == {"c1", "c2"}
    judge_c = log[0]["components"][0]
    assert judge_c["role"] == "judge" and log[0]["known_cost_usd"] > 0 and log[0]["kind"] == "rejudge"
    assert c.calls == 3  # c1 t1, c1 t2, c2 t1; the source-error row is not judged


def test_rejudge_resume_skips_done_rows(env):
    _run(env)
    code, c = _run(env)
    assert code == 0 and c.calls == 0
    out = env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl"
    assert len(out.read_text().splitlines()) == 4


def test_rejudge_resume_rejudges_error_rows(env):
    class Flaky(FakeClient):
        def complete(self, *a):
            self.calls += 1
            if self.calls == 1:
                return "garbage", {"prompt_tokens": 1, "completion_tokens": 1}
            return json.dumps(self.answer), {"prompt_tokens": 1, "completion_tokens": 1}
    _run(env, client=Flaky())
    out = env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl"
    assert sum(r["verdict"] == "ERROR" and r["judge_path"] != "source_error" for r in results_io.load_rows(out)) == 1
    code, c = _run(env)
    assert c.calls == 1
    assert all(r["verdict"] != "ERROR" or r["judge_path"] == "source_error" for r in results_io.load_rows(out))


def test_rejudge_refuses_manifest_mismatch(env):
    _run(env)
    with pytest.raises(SystemExit):
        _run(env, "--samples", "3")


def test_rejudge_filters_and_samples(env):
    code, c = _run(env, "--agent", "ag", "--config", "B0", "--pass", "1", "--samples", "3")
    assert c.calls == 9
    row = results_io.load_rows(env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl")[0]
    assert row["checklist"]["_samples"] == 3 and row["judge_samples"] == 3
    code, c = _run(env, "--agent", "nobody")
    assert c.calls == 0


def test_rejudge_keys_file_limits_turns(env, tmp_path):
    keys = tmp_path / "keys.json"
    keys.write_text(json.dumps([["ag", "B0", 1, "c1", 2]]))
    code, c = _run(env, "--keys-file", str(keys))
    rows = results_io.load_rows(env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl")
    assert [(r["case_id"], r["turn"]) for r in rows] == [("c1", 2)] and c.calls == 1
    assert "Which one, A1 or B2?" in c.last[-1]["content"]  # prior turn still in the transcript


def test_rejudge_refuses_locked_split_without_prereg(env, monkeypatch):
    with pytest.raises(SystemExit):
        rejudge.main(["--judge", "v3", "--split", "test", "--src-root", str(env["src"]),
                      "--dst-root", str(env["dst"])], client=FakeClient())


def test_rejudge_max_usd_stops(env):
    code, c = _run(env, "--max-usd", "0.0001", "--workers", "1")
    assert code == runner.EXIT_COST_CAP and c.calls < 3


def test_rejudge_cap_counts_other_logs(env):
    main_log = env["src"] / "cost_log.jsonl"
    main_log.write_text(json.dumps({"known_cost_usd": 400.5, "cost_usd": 400.5}) + "\n")
    with pytest.raises(SystemExit):
        _run(env)


# ------------------------------------------------------------------ cost guard across logs

def test_cost_guard_sums_extra_logs(tmp_path):
    a, b = tmp_path / "rp" / "cost_log.jsonl", tmp_path / "rp_v3" / "cost_log.jsonl"
    results_io.append_rows(a, [{"known_cost_usd": 0.5}])
    results_io.append_rows(b, [{"known_cost_usd": 0.3}])
    assert runner.sibling_cost_logs(tmp_path / "rp") == [b]
    g = runner.CostGuard(cap=1.0, cost_log=a, extra_logs=runner.sibling_cost_logs(tmp_path / "rp"), max_case=0.1)
    assert g.cumulative == pytest.approx(0.8)
    assert g.can_start()
    results_io.append_rows(b, [{"known_cost_usd": 0.15}])
    assert not g.can_start()


# ------------------------------------------------------------------ runner --judge v3

class FakeAdapter:
    model = "gem"

    async def run(self, case, interceptor, system_prompt):
        return [TurnRecord(turn=1, user_text="u1", agent_text="Which one, A1 or B2?",
                           usage={"prompt_token_count": 10, "response_token_count": 1}),
                TurnRecord(turn=2, user_text="u2", agent_text="Cancelled.", calls=[Call("cancel_order", {"order_id": "A1"})],
                           usage={"prompt_token_count": 10, "response_token_count": 1})]


def _runner_setup(tmp_path, monkeypatch):
    cases = []
    for c in CASES:
        c = dict(c)
        c["_folder"] = tmp_path / c["id"]
        cases.append(c)
    monkeypatch.setattr(runner, "load_split", lambda split, only_id=None: cases)
    monkeypatch.setattr(runner, "check_asr_cache", lambda cs, split: None)
    monkeypatch.setattr(runner, "_git", lambda *a: "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    fake = FakeClient()
    monkeypatch.setattr(judge_v3, "default_client", lambda: fake)
    pr = {"input_per_mtok": 1.0, "output_per_mtok": 5.0}
    return fake, dict(agent_factory=FakeAdapter, prices={"gem": pr, runner.JUDGE_MODEL: pr},
                      results_dir=tmp_path / "results")


ARGV = ["--split", "dev_v2", "--agent", "gemini-3.1-flash-live-preview", "--config", "B0", "--pass", "1"]


def test_runner_judge_v3_wiring(tmp_path, monkeypatch):
    fake, hooks = _runner_setup(tmp_path, monkeypatch)
    assert runner.main(ARGV + ["--judge", "v3"], **hooks) == 0
    folder = tmp_path / "results" / "dev_v2" / "gemini-3.1-flash-live-preview" / "B0"
    rows = results_io.load_rows(folder / "pass1.jsonl")
    assert {r["judge_prompt_version"] for r in rows} == {"v3"}
    assert set(rows[0]) == set(runner.ROW_FIELDS)
    assert rows[0]["judge_path"] == "checklist" and rows[1]["judge_path"] == "code+checklist"
    man = json.loads((folder / "manifest.json").read_text())
    assert man["judge_prompt_version"] == "v3" and man["judge_samples"] == 1
    assert fake.calls == 4
    log = [json.loads(l) for l in (tmp_path / "results" / "cost_log.jsonl").read_text().splitlines()]
    judge_c = next(c for c in log[0]["components"] if c["role"] == "judge")
    assert judge_c["usage"]["prompt_tokens"] == 2000
    # resuming the same folder under v2 is refused (different judge)
    with pytest.raises(SystemExit):
        runner.main(ARGV, **hooks)


def test_runner_default_judge_is_v2(tmp_path, monkeypatch):
    _, hooks = _runner_setup(tmp_path, monkeypatch)
    args = runner.parse_args(ARGV)
    assert args.judge == "v2"
    man = runner.build_manifest(args, "gem")
    assert man["judge_prompt_version"] == "v2" and man["judge_samples"] is None


# ------------------------------------------------------------------ analysis reads a re-score root

def test_metrics_load_runs_reads_results_root(env):
    from rp.analysis import metrics
    _run(env)
    rows = metrics.load_runs("dev_v2", results_root=env["dst"])
    assert {r["judge_prompt_version"] for r in rows} == {"v3"} and len(rows) == 3
    assert len(metrics.load_errors("dev_v2", results_root=env["dst"])) == 1


def test_report_results_root_flag(env, tmp_path, monkeypatch):
    from rp.analysis import metrics, report
    _run(env)
    seen = {}
    monkeypatch.setattr(report, "rq1_table", lambda *a, **k: seen.setdefault("root", metrics.RESULTS_ROOT))
    for name in ("rq2_table", "rq3_table", "rq4_table", "per_domain_table", "human_validation_table",
                 "heatmap_csv", "forest_csv", "cost_safety_csv", "failure_stack_csv", "routing_curve_csv",
                 "latency_csv"):
        monkeypatch.setattr(report, name, lambda *a, **k: None)
    monkeypatch.setattr(metrics, "RESULTS_ROOT", metrics.RESULTS_ROOT)
    monkeypatch.setattr(report, "RESULTS_ROOT", report.RESULTS_ROOT)
    report.main(["--split", "dev_v2", "--out", str(tmp_path / "o"), "--results-root", str(env["dst"])])
    assert seen["root"] == env["dst"] and report.RESULTS_ROOT == env["dst"]



# ------------------------------------------------------------------ review fixes

def test_rejudge_refuses_code_change_unless_allowed(env, monkeypatch):
    monkeypatch.setattr(runner, "_git", lambda *a: "aaa\n")
    _run(env)
    monkeypatch.setattr(runner, "_git", lambda *a: "bbb\n")
    with pytest.raises(SystemExit):
        _run(env)
    code, _ = _run(env, "--allow-code-change")
    assert code == 0
    man = json.loads((env["dst"] / "dev_v2" / "ag" / "B0" / "manifest.json").read_text())
    assert man["invocations"][-1]["allow_code_change"] is True


def test_rejudge_records_and_checks_judge_rules_hash(env, monkeypatch):
    _run(env)
    man = json.loads((env["dst"] / "dev_v2" / "ag" / "B0" / "manifest.json").read_text())
    assert man["judge_rules_hash"] == rejudge.judge_rules_hash() and len(man["judge_rules_hash"]) == 64
    monkeypatch.setattr(judge_v3, "CHECKLIST_SYSTEM", judge_v3.CHECKLIST_SYSTEM + " changed")
    assert rejudge.judge_rules_hash() != man["judge_rules_hash"]
    with pytest.raises(SystemExit):
        _run(env)


@pytest.mark.parametrize("bad", ["rp", "rp/sub", ".", "other", "x/rp_v3"])
def test_rejudge_refuses_bad_dst_root(env, tmp_path, bad):
    with pytest.raises(SystemExit):
        rejudge.main(["--judge", "v3", "--split", "dev_v2", "--src-root", str(env["src"]),
                      "--dst-root", str(tmp_path / bad)], client=FakeClient())
    assert not (tmp_path / "rp" / "dev_v2" / "ag" / "B0" / "manifest.json").exists()


def test_blocked_only_row_is_judged_as_not_executed(env):
    blocked = [{"call": {"function": "cancel_order", "args": {"order_id": "A1"}}, "gate": "clarify"}]
    results_io.append_rows(env["file"], [_row("c1", 1, blocked=blocked, sut_tool_calls=[])])
    _run(env)
    row = next(r for r in results_io.load_rows(env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl")
               if (r["case_id"], r["turn"]) == ("c1", 1))
    assert row["judge_path"] == "checklist" and row["verdict"] == "PASS"


def test_runner_cap_counts_sibling_rejudge_logs(tmp_path, monkeypatch):
    _, hooks = _runner_setup(tmp_path, monkeypatch)
    hooks["results_dir"] = tmp_path / "rp"
    results_io.append_rows(tmp_path / "rp_v3" / "cost_log.jsonl", [{"known_cost_usd": 400.5}])
    with pytest.raises(SystemExit):
        runner.main(ARGV, **hooks)


def test_report_help_notes_judge2_under_rejudge_root(capsys):
    from rp.analysis import report
    with pytest.raises(SystemExit):
        report.main(["--help"])
    assert "judge2" in capsys.readouterr().out


def test_rejudge_case_ids_file_limits_cases(env, tmp_path):
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"case_ids": ["c1"], "description": "x"}))
    code, c = _run(env, "--case-ids-file", str(ids))
    rows = results_io.load_rows(env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl")
    assert {r["case_id"] for r in rows} == {"c1"} and c.calls == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["nope"]))
    with pytest.raises(SystemExit):
        _run(env, "--case-ids-file", str(bad))



def test_rejudge_cap_does_not_double_count_the_main_log(env, monkeypatch):
    main_log = env["src"] / "cost_log.jsonl"
    main_log.write_text(json.dumps({"known_cost_usd": 250.0}) + "\n")
    monkeypatch.chdir(env["src"].parent)   # relative --dst-root, absolute RESULTS_BASE
    code = rejudge.main(["--judge", "v3", "--split", "dev_v2", "--src-root", "rp", "--dst-root", "rp_v3"],
                        client=FakeClient())
    assert code == 0   # 250 counted once (< 400), not twice


def test_redo_keys_rejudges_done_turns_under_new_rules(env, tmp_path, monkeypatch):
    _run(env)
    keys = tmp_path / "redo.json"
    keys.write_text(json.dumps([["ag", "B0", 1, "c1", 2]]))
    monkeypatch.setattr(judge_v3, "CHECKLIST_SYSTEM", judge_v3.CHECKLIST_SYSTEM + " v2 wording")
    with pytest.raises(SystemExit):          # rules changed: plain resume refused
        _run(env)
    code, c = _run(env, "--redo-keys", str(keys), "--allow-rules-change")
    assert code == 0 and c.calls == 1
    out = env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl"
    assert len(out.read_text().splitlines()) == 5
    man = json.loads((env["dst"] / "dev_v2" / "ag" / "B0" / "manifest.json").read_text())
    assert man["judge_rules_hash"] == rejudge.judge_rules_hash()
    assert man["invocations"][-1]["previous_judge_rules_hash"] != man["judge_rules_hash"]


# ------------------------------------------------------------------ fix round 4

class NoCalls:
    def complete(self, *a):
        raise AssertionError("remap must not call the LLM")


def test_rejudge_rows_carry_rules_hash(env):
    _run(env)
    rows = results_io.load_rows(env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl")
    assert {r["judge_rules_hash"] for r in rows if r["judge_path"] != "source_error"} == {judge_v3.rules_hash()}
    assert rejudge.judge_rules_hash() == judge_v3.rules_hash()


def test_remap_replays_stored_answers_under_new_rules_without_llm(env, monkeypatch):
    _run(env)
    old = judge_v3.rules_hash()
    monkeypatch.setattr(judge_v3, "CHECKLIST_SYSTEM", judge_v3.CHECKLIST_SYSTEM + " new")
    monkeypatch.setattr(judge_v3, "map_checklist", lambda gate, c: ("FAIL", "unknown"))
    code = rejudge.main(["--judge", "v3", "--split", "dev_v2", "--src-root", str(env["src"]),
                         "--dst-root", str(env["dst"]), "--remap"], client=NoCalls())
    assert code == 0
    out = env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl"
    rows = {(r["case_id"], r["turn"]): r for r in results_io.load_rows(out)}
    assert rows[("c1", 1)]["verdict"] == "FAIL" and rows[("c1", 1)]["remapped_from_stored_checklist"] is True
    assert rows[("c1", 1)]["judge_rules_hash"] == judge_v3.rules_hash() != old
    assert rows[("c1", 2)]["verdict"] == "PASS"   # code+checklist path, unaffected
    man = json.loads((env["dst"] / "dev_v2" / "ag" / "B0" / "manifest.json").read_text())
    assert man["judge_rules_hash"] == judge_v3.rules_hash()
    assert man["invocations"][-1]["previous_judge_rules_hash"] == old
    # a second remap under the same rules is a no-op
    n = len(out.read_text().splitlines())
    rejudge.main(["--judge", "v3", "--split", "dev_v2", "--src-root", str(env["src"]),
                  "--dst-root", str(env["dst"]), "--remap"], client=NoCalls())
    assert len(out.read_text().splitlines()) == n


def test_metrics_refuses_mixed_rules_hashes_in_one_file(env):
    from rp.analysis import metrics
    _run(env)
    out = env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl"
    r = results_io.load_rows(out)[0]
    results_io.append_rows(out, [{**r, "judge_rules_hash": "other"}])
    with pytest.raises(ValueError):
        metrics.load_runs("dev_v2", results_root=env["dst"])
    assert metrics.load_runs("dev_v2", results_root=env["dst"], allow_mixed_rules=True)


def test_report_header_shows_rules_hash(env):
    from rp.analysis import metrics, report
    _run(env)
    ld = report.Loader("dev_v2")
    orig = metrics.RESULTS_ROOT
    try:
        metrics.set_results_root(env["dst"])
        ld.load("ag", "B0")
    finally:
        metrics.set_results_root(orig)
    assert f"judge_rules_hash: {judge_v3.rules_hash()}" in ld.header()


def test_runner_v3_rows_carry_rules_hash(tmp_path, monkeypatch):
    _, hooks = _runner_setup(tmp_path, monkeypatch)
    runner.main(ARGV + ["--judge", "v3"], **hooks)
    rows = results_io.load_rows(tmp_path / "results" / "dev_v2" / "gemini-3.1-flash-live-preview" / "B0" / "pass1.jsonl")
    assert "judge_rules_hash" in runner.ROW_FIELDS
    assert {r["judge_rules_hash"] for r in rows} == {judge_v3.rules_hash()}


def test_remap_restamps_source_error_rows_and_metrics_ignores_error_rows(env, monkeypatch):
    from rp.analysis import metrics
    _run(env)
    out = env["dst"] / "dev_v2" / "ag" / "B0" / "pass1.jsonl"
    r = results_io.load_rows(out)[0]
    results_io.append_rows(out, [{**r, "turn": 9, "verdict": "ERROR", "judge_rules_hash": None}])  # judge ERROR row
    assert metrics.load_runs("dev_v2", results_root=env["dst"])    # ERROR rows do not count as a rule set
    monkeypatch.setattr(judge_v3, "CHECKLIST_SYSTEM", judge_v3.CHECKLIST_SYSTEM + " new")
    rejudge.main(["--judge", "v3", "--split", "dev_v2", "--src-root", str(env["src"]),
                  "--dst-root", str(env["dst"]), "--remap"], client=NoCalls())
    rows = {(x["case_id"], x["turn"]): x for x in results_io.load_rows(out)}
    assert rows[("c2", 2)]["judge_path"] == "source_error"
    assert rows[("c2", 2)]["judge_rules_hash"] == judge_v3.rules_hash()
    assert metrics.load_runs("dev_v2", results_root=env["dst"])
