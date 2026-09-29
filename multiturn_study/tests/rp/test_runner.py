import asyncio
import json
from pathlib import Path

import pytest

from rp import judge as rp_judge
from rp import results_io, runner
from rp.types import Call, TurnRecord


# ------------------------------------------------------------------ brief tests

def test_refuses_test_split_without_prereg(monkeypatch):
    monkeypatch.setattr(runner, "_has_tag", lambda t: False)
    try:
        runner.check_allowed("test")
        assert False
    except SystemExit as e:
        assert "prereg" in str(e)


def test_allows_test_split_with_prereg(monkeypatch):
    monkeypatch.setattr(runner, "_has_tag", lambda t: t == "prereg")
    runner.check_allowed("test")
    runner.check_allowed("dev_v2")


def test_resume_reruns_errored_case(tmp_path):
    out = tmp_path / "pass1.jsonl"
    rows = [{"case_id": "a", "turn": 1, "sut_error": None}, {"case_id": "a", "turn": 2, "sut_error": None},
            {"case_id": "b", "turn": 1, "sut_error": "timeout"}]
    out.write_text("\n".join(json.dumps(r) for r in rows))
    assert runner.done_cases(out, {"a": 2, "b": 1}) == {"a"}


def test_cost_estimate():
    prices = {"m": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}
    assert runner.cost_usd({"input_tokens": 1_000_000, "output_tokens": 500_000}, "m", prices) == 2.0


# ------------------------------------------------------------------ resume / last-row-wins

def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_last_row_wins_rerun_case_becomes_done(tmp_path):
    out = tmp_path / "p.jsonl"
    _write(out, [{"case_id": "b", "turn": 1, "sut_error": "timeout"},
                 {"case_id": "b", "turn": 2, "sut_error": "skipped_after_prior_turn_error"},
                 {"case_id": "b", "turn": 1, "sut_error": None, "verdict": "PASS"},
                 {"case_id": "b", "turn": 2, "sut_error": None, "verdict": "FAIL"}])
    assert runner.done_cases(out, {"b": 2}) == {"b"}
    latest = results_io.load_rows(out)
    assert [(r["case_id"], r["turn"], r["verdict"]) for r in latest] == [("b", 1, "PASS"), ("b", 2, "FAIL")]


def test_not_done_when_turn_missing_or_judge_error(tmp_path):
    out = tmp_path / "p.jsonl"
    _write(out, [{"case_id": "a", "turn": 1, "sut_error": None},
                 {"case_id": "c", "turn": 1, "sut_error": None, "verdict": "ERROR"}])
    assert runner.done_cases(out, {"a": 2, "c": 1}) == set()
    assert runner.done_cases(tmp_path / "missing.jsonl", {"a": 1}) == set()


def test_torn_trailing_line_is_ignored_and_append_starts_new_line(tmp_path):
    out = tmp_path / "p.jsonl"
    out.write_text(json.dumps({"case_id": "a", "turn": 1, "sut_error": None}) + "\n" + '{"case_id": "a", "tu')
    assert runner.done_cases(out, {"a": 1}) == {"a"}
    results_io.append_rows(out, [{"case_id": "z", "turn": 1, "sut_error": None}])
    assert {r["case_id"] for r in results_io.load_rows(out)} == {"a", "z"}


# ------------------------------------------------------------------ pricing

PRICES = {
    "rt": {"input_per_mtok": 4.0, "output_per_mtok": 16.0, "audio_input_per_mtok": 32.0,
           "audio_output_per_mtok": 64.0, "cached_input_per_mtok": 0.4, "cached_audio_input_per_mtok": 0.4},
    "gem": {"input_per_mtok": 0.5, "output_per_mtok": 2.0, "audio_input_per_mtok": 3.0,
            "audio_output_per_mtok": 12.0},
    "or": {"input_per_mtok": 1.0, "output_per_mtok": 5.0},
    "nulls": {"input_per_mtok": None, "output_per_mtok": None},
}
M = 1_000_000


def test_realtime_usage_split_by_modality_and_cache():
    u = {"input_tokens": 3 * M, "output_tokens": 2 * M, "total_tokens": 5 * M,
         "input_token_details.text_tokens": 2 * M, "input_token_details.audio_tokens": M,
         "input_token_details.cached_tokens": M,
         "input_token_details.cached_tokens_details.text_tokens": M,
         "input_token_details.cached_tokens_details.audio_tokens": 0,
         "output_token_details.text_tokens": M, "output_token_details.audio_tokens": M}
    # text: 1M uncached*4 + 1M cached*0.4 ; audio in 1M*32 ; out 1M*16 + 1M*64
    assert runner.cost_usd(u, "rt", PRICES) == pytest.approx(4 + 0.4 + 32 + 16 + 64)


def test_gemini_usage_split_by_modality():
    u = {"prompt_token_count": 2 * M, "response_token_count": M, "total_token_count": 3 * M,
         "prompt_tokens_details.TEXT": M, "prompt_tokens_details.AUDIO": M,
         "response_tokens_details.AUDIO": M, "safety_valve": "tool_rounds"}
    assert runner.cost_usd(u, "gem", PRICES) == pytest.approx(0.5 + 3 + 12)


def test_gemini_missing_details_priced_at_upper_bound_with_note():
    p = runner.price_usage({"prompt_token_count": M, "response_token_count": M}, "gem", PRICES)
    assert p["cost_usd"] == pytest.approx(3 + 12)
    assert p["notes"]


def test_openrouter_usage_nested():
    u = {"prompt_tokens": M, "completion_tokens": M, "total_tokens": 2 * M, "cost": 6.0,
         "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0}}
    p = runner.price_usage(u, "or", PRICES)
    assert p["cost_usd"] == pytest.approx(6.0)
    assert p["reported_cost_usd"] == 6.0


def test_unknown_usage_field_gives_null_cost_not_zero():
    p = runner.price_usage({"prompt_tokens": 10, "completion_tokens": 10, "weird_tokens": 7}, "or", PRICES)
    assert p["cost_usd"] is None and "weird_tokens" in p["unknown_fields"]
    assert p["known_cost_usd"] > 0


def test_unpriced_model_gives_null_cost():
    assert runner.cost_usd({"input_tokens": 10, "output_tokens": 1}, "absent", PRICES) is None
    assert runner.cost_usd({"input_tokens": 10, "output_tokens": 1}, "nulls", PRICES) is None


def test_missing_usage_marker_gives_null_cost():
    p = runner.price_usage({"prompt_tokens": 10, "completion_tokens": 1, "missing_usage_responses": 1}, "or", PRICES)
    assert p["cost_usd"] is None and p["notes"]


def test_empty_usage_costs_zero():
    assert runner.cost_usd({}, "or", PRICES) == 0.0


def test_pricing_json_entries_have_source_and_retrieved():
    prices = json.loads(runner.PRICING_PATH.read_text())
    for model in ("google/gemini-2.5-flash-lite", "openai/gpt-4.1-mini", "anthropic/claude-haiku-4.5",
                  "meta-llama/llama-3.3-70b-instruct", "gpt-4o-mini-transcribe", "gemini-3.8-live",
                  "gemini-3.1-flash-live-preview", "gemini-2.5-flash-native-audio-latest", "gpt-realtime-2.1",
                  "gpt-realtime-2.1-mini", "gpt-realtime-1.5"):
        e = prices[model]
        assert e["source"] and e["retrieved"]
        for k in ("input_per_mtok", "output_per_mtok", "audio_input_per_mtok", "audio_output_per_mtok"):
            assert k in e


# ------------------------------------------------------------------ verifier requirement / preflight

def test_verifier_required_configs():
    assert runner.needs_verifier("SAGE") and runner.needs_verifier("B4")
    assert runner.needs_verifier("SAGE-no_routing")
    assert not runner.needs_verifier("SAGE-no_verifier") and not runner.needs_verifier("B0")
    with pytest.raises(SystemExit):
        runner.make_interceptor("SAGE", None)
    with pytest.raises(SystemExit):
        runner.make_interceptor("B4", None)
    assert runner.make_interceptor("SAGE-no_verifier", None).name == "SAGE-no_verifier"


def test_fresh_interceptor_per_call():
    assert runner.make_interceptor("B3", None) is not runner.make_interceptor("B3", None)


def test_refuses_when_asr_cache_missing(tmp_path):
    folder = tmp_path / "c1"
    folder.mkdir()
    (folder / "input_turn1.wav").write_bytes(b"")
    case = {"id": "c1", "steps": [{"turn": 1}], "_folder": folder}
    with pytest.raises(SystemExit) as e:
        runner.check_asr_cache([case], "dev_v2")
    assert "python3 -m rp.asr --split dev_v2" in str(e.value)
    (folder / "input_turn1.asr.json").write_text(json.dumps({"text": "hi"}))
    runner.check_asr_cache([case], "dev_v2")


# ------------------------------------------------------------------ judge wrapper

def _case(folder=None):
    return {"id": "c1", "domain": "ecommerce", "num_steps": 2, "_folder": folder,
            "steps": [{"turn": 1, "user": "u1", "gate_expected": "execute", "severity": "low",
                       "policy_rule_id": "R1", "trap_tool_calls": []},
                      {"turn": 2, "user": "u2", "gate_expected": "confirm", "severity": "high",
                       "policy_rule_id": "R2", "trap_tool_calls": [{"function": "cancel_order",
                                                                    "args": {"order_id": "A1"}}]}]}


def test_judge_turns_matcher_decides_severe_and_judge_sees_only_executed(monkeypatch):
    seen = []

    def fake(case, step, sut, model):
        seen.append([dict(s) for s in sut])
        return {"verdict": "FAIL", "detected_gate": "execute", "reason": "r"}

    monkeypatch.setattr(rp_judge, "run_judge_v2", fake)
    turns = [TurnRecord(turn=1, agent_text="ok"),
             TurnRecord(turn=2, agent_text="done", calls=[Call("cancel_order", {"order_id": "a-1"})],
                        blocked=[{"call": {"function": "refund", "args": {}}, "gate": "confirm"}])]
    rows = rp_judge.judge_turns(_case(), turns, "jm")
    assert [r["verdict"] for r in rows] == ["FAIL", "SEVERE"]
    assert rows[1]["verdict_judge"] == "FAIL" and rows[1]["judge_model"] == "jm"
    assert rows[1]["judge_prompt_version"] == rp_judge.JUDGE_PROMPT_VERSION
    assert seen[1][1]["tool_calls"] == [{"function": "cancel_order", "args": {"order_id": "a-1"}}]


def test_judge_turns_error_turn_not_judged(monkeypatch):
    calls = []
    monkeypatch.setattr(rp_judge, "run_judge_v2", lambda *a, **k: calls.append(1) or {"verdict": "PASS"})
    rows = rp_judge.judge_turns(_case(), [TurnRecord(turn=1, agent_text="x"),
                                          TurnRecord(turn=2, error="timeout")], "jm")
    assert rows[1]["verdict"] == "ERROR" and rows[1]["verdict_judge"] is None and len(calls) == 1


def test_judge_usage_is_captured_from_http_response(monkeypatch):
    class Resp:
        def json(self):
            return {"usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    class FakeReal:
        def post(self, *a, **k):
            return Resp()

    monkeypatch.setattr(rp_judge._RECORDER, "_real", FakeReal())

    def fake(case, step, sut, model):
        rp_judge._jv2.requests.post("u")
        return {"verdict": "PASS"}

    monkeypatch.setattr(rp_judge, "run_judge_v2", fake)
    usage = {}
    rp_judge.judge_turns(_case(), [TurnRecord(turn=1), TurnRecord(turn=2)], "jm", usage_out=usage)
    assert usage == {"prompt_tokens": 10, "completion_tokens": 4}


# ------------------------------------------------------------------ end to end with fakes

class FakeAdapter:
    model = "gem"

    def __init__(self, fail_case=None):
        self.fail_case = fail_case

    async def run(self, case, interceptor, system_prompt):
        if case["id"] == self.fail_case:
            return [TurnRecord(turn=1, user_text="u", error="timeout"),
                    TurnRecord(turn=2, error="skipped_after_prior_turn_error")]
        return [TurnRecord(turn=1, user_text="asr1", agent_text="a1", latency_ms=12.0, guard_ms=0.1,
                           usage={"prompt_token_count": 100, "response_token_count": 10,
                                  "prompt_tokens_details.TEXT": 100, "response_tokens_details.AUDIO": 10}),
                TurnRecord(turn=2, user_text="asr2", agent_text="a2", calls=[Call("cancel_order", {"order_id": "A1"})],
                           blocked=[{"call": {"function": "process_refund", "args": {"amount": 5}},
                                     "gate": "confirm", "rule_id": "R", "reason": "x"}],
                           verifier_calls=1, usage={"prompt_token_count": 50, "response_token_count": 5,
                                                    "prompt_tokens_details.TEXT": 50,
                                                    "response_tokens_details.AUDIO": 5,
                                                    "safety_valve": "tool_rounds"})]


def _fake_judge(case, turns, model, usage_out=None):
    if usage_out is not None:
        usage_out.update({"prompt_tokens": 1000, "completion_tokens": 100})
    return [{"verdict": "PASS", "verdict_judge": "PASS", "detected_gate": "execute", "reason": "",
             "judge_model": model, "judge_prompt_version": "v2"} for _ in turns]


def _setup(tmp_path, monkeypatch, fail_case=None):
    cases = []
    for cid in ("c1", "c2"):
        c = _case(tmp_path / cid)
        c["id"] = cid
        cases.append(c)
    monkeypatch.setattr(runner, "load_split", lambda split, only_id=None: [c for c in cases
                                                                          if only_id in (None, c["id"])])
    monkeypatch.setattr(runner, "check_asr_cache", lambda cs, split: None)
    monkeypatch.setattr(runner, "_git", lambda *a: "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    hooks = dict(agent_factory=lambda: FakeAdapter(fail_case), judge_fn=_fake_judge,
                 prices={"gem": PRICES["gem"], runner.JUDGE_MODEL: PRICES["or"],
                         runner.VERIFIER_MODEL: PRICES["or"]},
                 results_dir=tmp_path / "results")
    return hooks


def _argv(*extra):
    return ["--split", "dev_v2", "--agent", "gemini-3.1-flash-live-preview", "--config", "B0",
            "--pass", "1", *extra]


def test_end_to_end_rows_manifest_cost_log(tmp_path, monkeypatch, capsys):
    hooks = _setup(tmp_path, monkeypatch)
    assert runner.main(_argv(), **hooks) == 0
    folder = tmp_path / "results" / "dev_v2" / "gemini-3.1-flash-live-preview" / "B0"
    rows = results_io.load_rows(folder / "pass1.jsonl")
    assert len(rows) == 4
    r = rows[1]
    assert set(r) == set(runner.ROW_FIELDS)
    assert {"run_id", "git_sha", "severe_matcher"} <= set(runner.ROW_FIELDS)
    assert r["case_id"] == "c1" and r["turn"] == 2 and r["policy_rule_id"] == "R2"
    assert r["severity"] == "high" and r["gate_expected"] == "confirm" and r["num_steps"] == 2
    assert r["sut_tool_calls"] == [{"function": "cancel_order", "args": {"order_id": "A1"}}]
    assert r["blocked"][0]["call"]["function"] == "process_refund"
    assert r["user_asr"] == "asr2" and r["verifier_calls"] == 1 and r["safety_valve"] == "tool_rounds"
    assert r["response_status"] is None
    assert (r["split"], r["agent"], r["config"], r["pass"]) == ("dev_v2", "gemini-3.1-flash-live-preview", "B0", 1)
    man = json.loads((folder / "manifest.json").read_text())
    for k in ("git_sha", "dirty", "created", "split", "agent", "config", "pass", "tool_overlay_sha", "policy_sha",
              "judge_model", "judge_prompt_version", "verifier_model", "asr_model", "cascade_llm", "agent_model",
              "python", "packages"):
        assert k in man
    assert man["agent_model"] == "gem" and "google-genai" in man["packages"]
    log = [json.loads(l) for l in (tmp_path / "results" / "cost_log.jsonl").read_text().splitlines()]
    assert len(log) == 2
    agent_c = next(c for c in log[0]["components"] if c["role"] == "agent")
    assert agent_c["cost_usd"] == pytest.approx((150 * 0.5 + 15 * 12) / M)
    judge_c = next(c for c in log[0]["components"] if c["role"] == "judge")
    assert judge_c["cost_usd"] == pytest.approx((1000 * 1 + 100 * 5) / M)
    assert "cumulative" in capsys.readouterr().out


def test_end_to_end_resume_skips_done_and_reruns_errored(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch, fail_case="c2")
    assert runner.main(_argv(), **hooks) == 0
    out = tmp_path / "results" / "dev_v2" / "gemini-3.1-flash-live-preview" / "B0" / "pass1.jsonl"
    assert runner.done_cases(out, {"c1": 2, "c2": 2}) == {"c1"}
    ran = []
    hooks2 = _setup(tmp_path, monkeypatch)
    inner = hooks2["agent_factory"]

    def factory():
        a = inner()
        orig = a.run

        async def run(case, i, p):
            ran.append(case["id"])
            return await orig(case, i, p)
        a.run = run
        return a
    hooks2["agent_factory"] = factory
    assert runner.main(_argv(), **hooks2) == 0
    assert ran == ["c2"]
    assert runner.done_cases(out, {"c1": 2, "c2": 2}) == {"c1", "c2"}
    assert len(out.read_text().splitlines()) == 6  # appended, old errored rows kept


def test_cost_cap_refuses_to_start(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    log = tmp_path / "results" / "cost_log.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"known_cost_usd": 400.5, "cost_usd": 400.5}) + "\n")
    with pytest.raises(SystemExit) as e:
        runner.main(_argv(), **hooks)
    assert e.value.code not in (0, None)
    assert not (tmp_path / "results" / "dev_v2").exists() or not list((tmp_path / "results").rglob("pass1.jsonl"))


def test_cost_guard_stops_before_exceeding_cap():
    g = runner.CostGuard(cumulative=399.0, cap=400.0)
    assert g.can_start()
    g.record(0.6)
    assert not g.can_start()  # 399.6 + 1 more case at 0.6 > 400
    g2 = runner.CostGuard(cumulative=10.0, cap=400.0)
    g2.record(391.0)
    assert g2.over_cap()


def test_manifest_mismatch_on_resume_refuses(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    assert runner.main(_argv("--limit", "1"), **hooks) == 0
    man_path = tmp_path / "results" / "dev_v2" / "gemini-3.1-flash-live-preview" / "B0" / "manifest.json"
    m = json.loads(man_path.read_text())
    m["tool_overlay_sha"] = "different"
    man_path.write_text(json.dumps(m))
    with pytest.raises(SystemExit) as e:
        runner.main(_argv(), **hooks)
    assert "tool_overlay_sha" in str(e.value)


def test_refuses_unknown_agent_or_config(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        runner.main(["--split", "dev_v2", "--agent", "nope", "--config", "B0", "--pass", "1"], **hooks)
    with pytest.raises(SystemExit):
        runner.main(["--split", "dev_v2", "--agent", "cascade", "--config", "B9", "--pass", "1"], **hooks)


def test_openrouter_cached_tokens_discounted_not_unknown():
    prices = {"or": {"input_per_mtok": 1.0, "output_per_mtok": 5.0, "cached_input_per_mtok": 0.1}}
    u = {"prompt_tokens": 2 * M, "completion_tokens": 0, "prompt_tokens_details": {"cached_tokens": M},
         "completion_tokens_details": {"reasoning_tokens": 5}}
    p = runner.price_usage(u, "or", prices)
    assert p["unknown_fields"] == [] and p["cost_usd"] == pytest.approx(1.0 + 0.1)


# ------------------------------------------------------------------ fix round 1

def test_judge_usage_separate_per_thread(monkeypatch):
    import threading

    class Resp:
        def __init__(self, n):
            self.n = n

        def json(self):
            return {"usage": {"prompt_tokens": self.n, "completion_tokens": 1}}

    class FakeReal:
        def post(self, url, **k):
            return Resp(int(url))

    monkeypatch.setattr(rp_judge._RECORDER, "_real", FakeReal())
    barrier = threading.Barrier(2)

    def fake(case, step, sut, model):
        barrier.wait()          # both threads are inside judge_turns at the same time
        rp_judge._jv2.requests.post(model)
        barrier.wait()
        return {"verdict": "PASS"}

    monkeypatch.setattr(rp_judge, "run_judge_v2", fake)
    out = {"7": {}, "100": {}}

    def work(m):
        rp_judge.judge_turns(_case(), [TurnRecord(turn=1), TurnRecord(turn=2)], m, usage_out=out[m])

    ts = [threading.Thread(target=work, args=(m,)) for m in out]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out["7"] == {"prompt_tokens": 14, "completion_tokens": 2}
    assert out["100"] == {"prompt_tokens": 200, "completion_tokens": 2}


def test_judge_turns_uses_loop_position_not_turn_number(monkeypatch):
    monkeypatch.setattr(rp_judge, "run_judge_v2", lambda *a, **k: {"verdict": "PASS"})
    case = _case()
    for i, s in enumerate(case["steps"]):
        s["turn"] = i + 5                       # turn numbers not starting at 1
    turns = [TurnRecord(turn=5), TurnRecord(turn=6, calls=[Call("cancel_order", {"order_id": "A1"})])]
    assert [r["verdict"] for r in rp_judge.judge_turns(case, turns, "jm")] == ["PASS", "SEVERE"]


def test_error_turn_keeps_severe_matcher(monkeypatch):
    monkeypatch.setattr(rp_judge, "run_judge_v2", lambda *a, **k: {"verdict": "PASS"})
    turns = [TurnRecord(turn=1),
             TurnRecord(turn=2, error="timeout", calls=[Call("cancel_order", {"order_id": "A1"})])]
    rows = rp_judge.judge_turns(_case(), turns, "jm")
    assert rows[1]["verdict"] == "ERROR" and rows[1]["severe_matcher"] == "SEVERE"
    assert rows[0]["severe_matcher"] == "PASS"


def test_prereg_gate_uses_locked_splits(monkeypatch):
    monkeypatch.setattr(runner, "_has_tag", lambda t: False)
    monkeypatch.setattr(runner, "LOCKED", {"other_locked"})
    runner.check_allowed("test")
    with pytest.raises(SystemExit):
        runner.check_allowed("other_locked")


def test_cost_guard_rereads_shared_log_across_instances(tmp_path):
    log = tmp_path / "cost_log.jsonl"
    a = runner.CostGuard(cap=1.0, cost_log=log, max_case=0.4)
    b = runner.CostGuard(cap=1.0, cost_log=log, max_case=0.4)
    assert b.can_start()
    a.record(0.7, {"case_id": "x", "known_cost_usd": 0.7, "cost_usd": 0.7})
    assert not b.can_start()                        # 0.7 + 0.4 > 1.0, seen from the shared log
    rec = json.loads(log.read_text().splitlines()[0])
    assert rec["cumulative_known_usd"] == pytest.approx(0.7)
    assert (tmp_path / "cost_log.jsonl.lock").exists()


def test_max_case_seed_from_log_or_default(tmp_path):
    log = tmp_path / "cost_log.jsonl"
    assert runner.seed_max_case(log, "gpt-realtime-2.1") == 0.60
    assert runner.seed_max_case(log, "gemini-3.8-live") == 0.60
    assert runner.seed_max_case(log, "gpt-realtime-1.5") == 0.30
    assert runner.seed_max_case(log, "cascade") == 0.10
    _write(log, [{"agent": "cascade", "known_cost_usd": 0.02}, {"agent": "cascade", "known_cost_usd": 0.05},
                 {"agent": "gpt-realtime-1.5", "known_cost_usd": 3.0}])
    assert runner.seed_max_case(log, "cascade") == 0.05


def _seed_log(tmp_path, cost):
    log = tmp_path / "results" / "cost_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"agent": "someone-else", "known_cost_usd": cost, "cost_usd": cost}) + "\n")


def test_cap_stops_before_exceeding_with_seeded_max(tmp_path, monkeypatch, capsys):
    hooks = _setup(tmp_path, monkeypatch)
    _seed_log(tmp_path, 399.5)   # default seed for this agent is $0.30: one case fits, two in flight do not
    assert runner.main(_argv("--workers", "2"), **hooks) == runner.EXIT_COST_CAP
    err = capsys.readouterr().err
    assert "stopped before exceeding the cap" in err


def test_cap_exceeded_by_in_flight_message(tmp_path, monkeypatch, capsys):
    hooks = _setup(tmp_path, monkeypatch)
    _seed_log(tmp_path, 399.0)

    def pricey_judge(case, turns, model, usage_out=None):
        usage_out.update({"prompt_tokens": 2_000_000, "completion_tokens": 0})   # $2 at $1/M
        return _fake_judge(case, turns, model)

    hooks["judge_fn"] = pricey_judge
    assert runner.main(_argv("--workers", "1"), **hooks) == runner.EXIT_COST_CAP
    err = capsys.readouterr().err
    assert "cap exceeded by in-flight cases (spent $" in err


def test_rows_and_cost_records_carry_run_id_and_git_sha(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "_git", lambda *a: "abc123\n" if a[:1] == ("rev-parse",) else "")
    assert runner.main(_argv(), **hooks) == 0
    folder = tmp_path / "results" / "dev_v2" / "gemini-3.1-flash-live-preview" / "B0"
    rows = results_io.load_rows(folder / "pass1.jsonl")
    man = json.loads((folder / "manifest.json").read_text())
    run_id = man["invocations"][0]["run_id"]
    assert run_id and all(r["run_id"] == run_id and r["git_sha"] == "abc123" for r in rows)
    log = [json.loads(l) for l in (tmp_path / "results" / "cost_log.jsonl").read_text().splitlines()]
    assert all(r["run_id"] == run_id and r["git_sha"] == "abc123" for r in log)


def test_resume_refuses_code_change_unless_allowed(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch, fail_case="c2")
    monkeypatch.setattr(runner, "_git", lambda *a: "sha1\n" if a[:1] == ("rev-parse",) else "")
    assert runner.main(_argv(), **hooks) == 0
    monkeypatch.setattr(runner, "_git", lambda *a: "sha2\n" if a[:1] == ("rev-parse",) else "")
    with pytest.raises(SystemExit) as e:
        runner.main(_argv(), **hooks)
    assert "git_sha" in str(e.value) and "--allow-code-change" in str(e.value)
    assert runner.main(_argv("--allow-code-change"), **hooks) == 0
    folder = tmp_path / "results" / "dev_v2" / "gemini-3.1-flash-live-preview" / "B0"
    invs = json.loads((folder / "manifest.json").read_text())["invocations"]
    assert [i["git_sha"] for i in invs] == ["sha1", "sha2"]
    assert [i["allow_code_change"] for i in invs] == [False, True]
    assert len({i["run_id"] for i in invs}) == 2


# ------------------------------------------------------------------ Task 16b fix round: audio tokens, whisper

def test_openrouter_audio_tokens_priced_at_audio_rate_not_double_counted():
    prices = {"or": {"input_per_mtok": 1.0, "output_per_mtok": 2.0, "audio_input_per_mtok": 100.0,
                     "audio_output_per_mtok": 50.0, "cached_input_per_mtok": 0.1}}
    u = {"prompt_tokens": 3 * M, "completion_tokens": 2 * M,
         "prompt_tokens_details": {"cached_tokens": M, "audio_tokens": M, "video_tokens": 0},
         "completion_tokens_details": {"reasoning_tokens": 5, "audio_tokens": M, "image_tokens": 0}}
    p = runner.price_usage(u, "or", prices)
    assert p["unknown_fields"] == []
    # text_in = 3M - 1M cached - 1M audio = 1M at 1.0; cached 1M at 0.1; audio 1M at 100
    # text_out = 2M - 1M audio = 1M at 2.0; audio_out 1M at 50
    assert p["cost_usd"] == pytest.approx(1.0 + 0.1 + 100.0 + 2.0 + 50.0)


def test_openrouter_audio_tokens_without_audio_rate_are_unpriced():
    prices = {"or": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}
    u = {"prompt_tokens": 10, "completion_tokens": 1, "prompt_tokens_details": {"audio_tokens": 4}}
    p = runner.price_usage(u, "or", prices)
    assert p["cost_usd"] is None and p["unpriced"] == ["audio_in"]


def test_voxtral_smoke_usage_matches_measured_cost():
    """Task 16b live smoke (dev_v2 ecommerce_mt_04, B0): ~$0.2175 per 100 turns with
    real pricing.json (OpenRouter's audio unit 1e-4 USD)."""
    prices = runner.load_prices()
    turns = [
        {"prompt_tokens": 2901, "completion_tokens": 45, "total_tokens": 2946,
         "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "audio_tokens": 14, "video_tokens": 0},
         "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 0, "audio_tokens": 0}},
        {"prompt_tokens": 3089, "completion_tokens": 36, "total_tokens": 3125,
         "prompt_tokens_details": {"cached_tokens": 768, "cache_write_tokens": 0, "audio_tokens": 24, "video_tokens": 0},
         "completion_tokens_details": {"reasoning_tokens": 0, "image_tokens": 0, "audio_tokens": 0}},
    ]
    ps = [runner.price_usage(u, "mistralai/voxtral-small-24b-2507", prices) for u in turns]
    assert all(p["cost_usd"] is not None and p["unknown_fields"] == [] for p in ps)
    per_100 = 100 * sum(p["known_cost_usd"] for p in ps) / len(ps)
    assert per_100 == pytest.approx(0.2175, abs=0.001)


def test_refuses_when_whisper_cache_missing(tmp_path):
    folder = tmp_path / "c1"
    folder.mkdir()
    (folder / "input_turn1.wav").write_bytes(b"")
    case = {"id": "c1", "steps": [{"turn": 1}], "_folder": folder}
    with pytest.raises(SystemExit) as e:
        runner.check_whisper_cache([case], "dev_v2")
    assert "python3 -m rp.asr --whisper --split dev_v2" in str(e.value)
    (folder / "input_turn1.whisper.json").write_text(json.dumps({"text": "hi"}))
    runner.check_whisper_cache([case], "dev_v2")


def test_main_refuses_open_cascade_without_whisper_cache(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as e:
        runner.main(["--split", "dev_v2", "--agent", "open-cascade", "--config", "B0", "--pass", "1"], **hooks)
    assert "--whisper --split dev_v2" in str(e.value)


# --- --case-ids-file: run a fixed subset of a split (Task 19b tuning subset) ---


def _out(tmp_path):
    return tmp_path / "results" / "dev_v2" / "gemini-3.1-flash-live-preview" / "B0" / "pass1.jsonl"


def test_case_ids_file_json_list_runs_only_listed_cases(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    f = tmp_path / "ids.json"
    f.write_text(json.dumps(["c2"]))
    assert runner.main(_argv("--case-ids-file", str(f)), **hooks) == 0
    assert {r["case_id"] for r in results_io.load_rows(_out(tmp_path))} == {"c2"}


def test_case_ids_file_accepts_object_with_case_ids(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    f = tmp_path / "ids.json"
    f.write_text(json.dumps({"seed": 1, "case_ids": ["c1"]}))
    assert runner.main(_argv("--case-ids-file", str(f)), **hooks) == 0
    assert {r["case_id"] for r in results_io.load_rows(_out(tmp_path))} == {"c1"}


def test_case_ids_file_unknown_id_refuses(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    f = tmp_path / "ids.json"
    f.write_text(json.dumps(["c1", "nope"]))
    with pytest.raises(SystemExit, match="nope"):
        runner.main(_argv("--case-ids-file", str(f)), **hooks)


def test_case_ids_file_and_case_id_are_exclusive(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    f = tmp_path / "ids.json"
    f.write_text(json.dumps(["c1"]))
    with pytest.raises(SystemExit):
        runner.main(_argv("--case-ids-file", str(f), "--case-id", "c1"), **hooks)


# --- a second pass into the same agent/config folder (Task 19b) ---------------

def _argv_pass(k, *extra):
    return ["--split", "dev_v2", "--agent", "gemini-3.1-flash-live-preview", "--config", "B0",
            "--pass", str(k), *extra]


def test_second_pass_in_same_folder_gets_its_own_manifest(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    folder = tmp_path / "results" / "dev_v2" / "gemini-3.1-flash-live-preview" / "B0"
    assert runner.main(_argv_pass(1), **hooks) == 0
    first = (folder / "manifest.json").read_text()
    assert runner.main(_argv_pass(2), **_setup(tmp_path, monkeypatch)) == 0
    assert (folder / "manifest.json").read_text() == first          # pass 1's manifest untouched
    man2 = json.loads((folder / "manifest.pass2.json").read_text())
    assert man2["pass"] == 2 and len(man2["invocations"]) == 1
    assert len(results_io.load_rows(folder / "pass2.jsonl")) == 4


def test_resume_of_second_pass_still_checks_its_manifest(tmp_path, monkeypatch):
    hooks = _setup(tmp_path, monkeypatch)
    folder = tmp_path / "results" / "dev_v2" / "gemini-3.1-flash-live-preview" / "B0"
    assert runner.main(_argv_pass(2), **hooks) == 0
    p = folder / "manifest.pass2.json"
    man = json.loads(p.read_text()); man["judge_model"] = "other"; p.write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="judge_model"):
        runner.main(_argv_pass(2), **_setup(tmp_path, monkeypatch))
