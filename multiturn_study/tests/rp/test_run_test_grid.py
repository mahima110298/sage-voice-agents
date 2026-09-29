import sys
from pathlib import Path

import run_test_grid as rtg


# ------------------------------------------------------------- run list

def test_run_list_has_52_rows_no_dupes():
    assert len(rtg.RUN_LIST) == 52
    assert len(set(rtg.RUN_LIST)) == 52


def test_run_list_pass1_before_pass2():
    passes = [p for _, _, p in rtg.RUN_LIST]
    assert passes == [1] * 26 + [2] * 26


def test_run_list_full_grid_then_light_then_ablations_within_each_pass():
    pass1 = [(a, c) for a, c, p in rtg.RUN_LIST if p == 1]
    assert pass1 == rtg.FULL_GRID + rtg.LIGHT_GRID + rtg.ABLATIONS
    pass2 = [(a, c) for a, c, p in rtg.RUN_LIST if p == 2]
    assert pass2 == rtg.FULL_GRID + rtg.LIGHT_GRID + rtg.ABLATIONS


def test_run_list_matches_prereg_agents_and_configs():
    # docs/prereg/2026-analysis-plan.md §8: 20 full-grid + 24 light-grid + 8 ablation = 52
    assert len(rtg.FULL_GRID) == 10 and len(rtg.LIGHT_GRID) == 12 and len(rtg.ABLATIONS) == 4
    assert set(a for a, _ in rtg.FULL_GRID) == {"gemini-3.8-live", "qwen3.8-omni-flash"}
    assert set(a for a, _ in rtg.LIGHT_GRID) == {
        "gpt-realtime-2.1-mini", "gemini-3.1-flash-live-preview", "voxtral-small-24b", "open-cascade"}
    assert set(c for _, c in rtg.ABLATIONS) == {
        "SAGE-no_gate_types", "SAGE-no_verifier", "SAGE-no_credential_check", "SAGE-no_policy"}


def test_family_assignment():
    assert rtg.family("gpt-realtime-2.1-mini") == "openai"
    assert rtg.family("gemini-3.8-live") == "gemini_live"
    assert rtg.family("gemini-3.1-flash-live-preview") == "gemini_live"
    assert rtg.family("qwen3.8-omni-flash") == "qwen"
    assert rtg.family("voxtral-small-24b") is None
    assert rtg.family("open-cascade") is None


# ------------------------------------------------------------- fake launcher

class FakeProc:
    """Controllable stand-in for subprocess.Popen: stays "running" (poll() ->
    None) until the test calls `.finish(code)`."""

    def __init__(self):
        self._code = None

    def finish(self, code=0):
        self._code = code

    def poll(self):
        return self._code


class FakeClock:
    """Records every sleep call (real and backoff) instead of blocking, and
    lets a test advance fake processes between "ticks"."""

    def __init__(self):
        self.sleeps = []
        self.on_tick = None  # optional callback(clock) invoked on each sleep

    def __call__(self, seconds):
        self.sleeps.append(seconds)
        if self.on_tick is not None:
            self.on_tick(self)


def make_launcher(procs_by_key):
    def launch_fn(spec, log_path):
        proc = FakeProc()
        procs_by_key[spec.key] = proc
        return proc
    return launch_fn


# ------------------------------------------------------------- scheduler

def test_scheduler_respects_max_procs_and_family_caps():
    procs = {}
    clock = FakeClock()
    sched = rtg.Scheduler(rtg.RUN_LIST, make_launcher(procs), max_procs=14, sleep_fn=clock)

    # advance until everything has launched at least once, finishing each
    # batch of running procs so the queue drains without ever exceeding a cap.
    max_running_seen = 0
    max_gemini_live_seen = 0
    max_openai_seen = 0

    def tick(clock):
        nonlocal max_running_seen, max_gemini_live_seen, max_openai_seen
        max_running_seen = max(max_running_seen, len(sched.running))
        max_gemini_live_seen = max(max_gemini_live_seen, sched._family_running_count("gemini_live"))
        max_openai_seen = max(max_openai_seen, sched._family_running_count("openai"))
        for spec, proc in list(sched.running.values()):
            proc.finish(0)

    clock.on_tick = tick
    sched.run(poll_interval=1)

    assert max_running_seen <= 14
    assert max_gemini_live_seen <= 3
    assert max_openai_seen <= 2
    assert len(sched.completed) == 52
    assert len(sched.launch_order) == 52
    assert set(sched.launch_order) == {f"{a}__{c}__pass{p}" for a, c, p in rtg.RUN_LIST}


def test_scheduler_fills_spare_slots_out_of_order_when_family_capped():
    # Only gemini-3.8-live (cap 3) and voxtral (uncapped) rows, more gemini
    # rows than the cap allows: voxtral should launch before later gemini rows
    # once the gemini_live family is full, instead of the scheduler stalling.
    run_list = [("gemini-3.8-live", "B0", 1), ("gemini-3.8-live", "B2", 1),
                ("gemini-3.8-live", "B3", 1), ("gemini-3.8-live", "B4", 1),
                ("voxtral-small-24b", "B0", 1)]
    procs = {}
    sched = rtg.Scheduler(run_list, make_launcher(procs), max_procs=14, sleep_fn=lambda s: None)
    sched._reap()
    sched._fill_slots()
    assert sched._family_running_count("gemini_live") == 3
    assert "voxtral-small-24b__B0__pass1" in sched.launch_order
    assert "gemini-3.8-live__B4__pass1" not in sched.launch_order


def test_scheduler_preserves_priority_order_among_unconstrained_rows():
    run_list = [("voxtral-small-24b", "B0", 1), ("open-cascade", "B0", 1), ("voxtral-small-24b", "B2", 1)]
    procs = {}
    sched = rtg.Scheduler(run_list, make_launcher(procs), max_procs=1, sleep_fn=lambda s: None)
    sched._reap()
    sched._fill_slots()
    assert sched.launch_order == ["voxtral-small-24b__B0__pass1"]


def test_scheduler_backoff_on_nonzero_exit_reduces_qwen_cap_and_sleeps():
    procs = {}
    clock = FakeClock()
    run_list = [("qwen3.8-omni-flash", "B0", 1)]
    sched = rtg.Scheduler(run_list, make_launcher(procs), max_procs=14, sleep_fn=clock, backoff_seconds=60)
    sched._reap()
    sched._fill_slots()
    procs["qwen3.8-omni-flash__B0__pass1"].finish(1)  # nonzero exit
    sched._reap()
    assert clock.sleeps == [60]
    assert sched.qwen_cap == rtg.QWEN_CAP_START - rtg.QWEN_CAP_STEP


def test_scheduler_qwen_cap_floors_at_minimum():
    procs = {}
    sched = rtg.Scheduler([], make_launcher(procs), sleep_fn=lambda s: None)
    for _ in range(10):
        sched._on_failure(rtg.RunSpec("qwen3.8-omni-flash", "B0", 1), 1)
    assert sched.qwen_cap == rtg.QWEN_CAP_MIN


def test_scheduler_backoff_on_non_qwen_failure_backs_off_but_leaves_qwen_cap_alone():
    procs = {}
    clock = FakeClock()
    run_list = [("gemini-3.8-live", "B0", 1)]
    sched = rtg.Scheduler(run_list, make_launcher(procs), max_procs=14, sleep_fn=clock, backoff_seconds=60)
    sched._reap()
    sched._fill_slots()
    start_cap = sched.qwen_cap
    procs["gemini-3.8-live__B0__pass1"].finish(1)  # nonzero exit, non-qwen agent
    sched._reap()
    assert clock.sleeps == [60]  # still backs off
    assert sched.qwen_cap == start_cap  # qwen cap untouched


def test_note_429_burst_on_non_qwen_run_backs_off_but_leaves_qwen_cap_alone():
    clock = FakeClock()
    sched = rtg.Scheduler([], lambda spec, log_path: None, sleep_fn=clock)
    start_cap = sched.qwen_cap
    sched.note_429_burst(rtg.RunSpec("voxtral-small-24b", "B0", 1))
    assert clock.sleeps == [rtg.DEFAULT_BACKOFF_SECONDS]
    assert sched.qwen_cap == start_cap


def test_note_429_burst_triggers_same_backoff_without_reaping():
    clock = FakeClock()
    sched = rtg.Scheduler([], lambda spec, log_path: None, sleep_fn=clock)
    start_cap = sched.qwen_cap
    sched.note_429_burst(rtg.RunSpec("qwen3.8-omni-flash", "B0", 1))
    assert clock.sleeps == [rtg.DEFAULT_BACKOFF_SECONDS]
    assert sched.qwen_cap == start_cap - rtg.QWEN_CAP_STEP


def test_scheduler_deadlock_guard_raises_instead_of_hanging():
    # A cap of 0 for every family would starve the queue forever; assert the
    # scheduler fails loudly instead of looping silently.
    sched = rtg.Scheduler([("gemini-3.8-live", "B0", 1)], lambda spec, log_path: None,
                           max_procs=0, sleep_fn=lambda s: None)
    try:
        sched.run(poll_interval=0)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


# ------------------------------------------------------------- dry run / waves

def test_simulate_waves_covers_every_run_respecting_caps():
    waves = rtg.simulate_waves(rtg.RUN_LIST, max_procs=14)
    flat = [key for wave in waves for key in wave]
    assert len(flat) == 52
    assert set(flat) == {f"{a}__{c}__pass{p}" for a, c, p in rtg.RUN_LIST}
    for wave in waves:
        assert len(wave) <= 14


# ------------------------------------------------------------- budget guard

def test_parse_credits():
    assert rtg.parse_credits({"data": {"total_credits": 30, "total_usage": 4.05}}) == 30 - 4.05
    assert rtg.parse_credits({"total_credits": 10, "total_usage": 1}) == 9


def test_budget_state_thresholds():
    assert rtg.budget_state(10.0) == (False, False)
    assert rtg.budget_state(2.99) == (True, False)
    assert rtg.budget_state(0.99) == (True, True)


def test_budget_guard_never_succeeded_when_api_key_missing():
    guard = rtg.BudgetGuard(api_key=None)
    guard.maybe_check(force=True)
    assert guard.ever_succeeded is False
    assert guard.consecutive_failures == 1
    # one failure alone doesn't force stop_new yet (that needs two in a row)
    assert guard.stop_new is False


def test_budget_guard_never_succeeded_when_first_check_errors(monkeypatch):
    def boom(api_key, timeout=10.0):
        raise TimeoutError("no network")
    monkeypatch.setattr(rtg, "openrouter_remaining_credits", boom)

    guard = rtg.BudgetGuard(api_key="k")
    guard.maybe_check(force=True)

    assert guard.ever_succeeded is False
    assert guard.consecutive_failures == 1


def test_budget_guard_two_consecutive_failures_force_stop_new_then_clears_on_success(monkeypatch):
    responses = iter([10.0, TimeoutError("down"), TimeoutError("down"), 10.0])

    def flaky(api_key, timeout=10.0):
        r = next(responses)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(rtg, "openrouter_remaining_credits", flaky)

    guard = rtg.BudgetGuard(api_key="k", check_interval=0)

    guard.maybe_check(force=True)  # 1: succeeds, remaining $10 -> no stop
    assert guard.ever_succeeded is True
    assert guard.stop_new is False

    guard.maybe_check(force=True)  # 2: fails (1st consecutive) -- not enough yet
    assert guard.consecutive_failures == 1
    assert guard.stop_new is False

    guard.maybe_check(force=True)  # 3: fails (2nd consecutive) -- forces stop_new
    assert guard.consecutive_failures == 2
    assert guard.stop_new is True

    guard.maybe_check(force=True)  # 4: succeeds again, remaining $10 -> clears
    assert guard.consecutive_failures == 0
    assert guard.stop_new is False


def test_budget_guard_recovers_but_balance_still_says_stop(monkeypatch):
    # a check can succeed again after failures, but if the balance itself is
    # still under the threshold stop_new must stay on (not blindly cleared).
    responses = iter([10.0, TimeoutError("down"), TimeoutError("down"), 2.5])

    def flaky(api_key, timeout=10.0):
        r = next(responses)
        if isinstance(r, Exception):
            raise r
        return r
    monkeypatch.setattr(rtg, "openrouter_remaining_credits", flaky)

    guard = rtg.BudgetGuard(api_key="k", check_interval=0)
    guard.maybe_check(force=True)
    guard.maybe_check(force=True)
    guard.maybe_check(force=True)
    assert guard.stop_new is True  # forced by consecutive failures

    guard.maybe_check(force=True)  # succeeds, but $2.50 < $3 stop_new threshold
    assert guard.consecutive_failures == 0
    assert guard.stop_new is True  # still True, now because of the balance


def test_budget_guard_apply_noop_when_not_stop_all():
    guard = rtg.BudgetGuard(api_key="x")
    assert guard.stop_all is False
    # must not raise and must not touch the fake proc at all
    guard.apply({"k": (rtg.RunSpec("qwen3.8-omni-flash", "B0", 1), object())})


def test_budget_guard_apply_sends_sigterm_and_raises_when_stop_all(monkeypatch):
    guard = rtg.BudgetGuard(api_key="x")
    guard.stop_all = True
    guard.remaining_usd = 0.42

    killed = []
    monkeypatch.setattr(rtg.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    proc = type("FakePid", (), {"pid": 4242})()
    running = {"k": (rtg.RunSpec("qwen3.8-omni-flash", "B0", 1), proc)}

    try:
        guard.apply(running)
        assert False, "expected BudgetExhausted"
    except rtg.BudgetExhausted as e:
        assert "0.42" in str(e)

    assert killed == [(4242, rtg.signal.SIGTERM)]


def test_budget_guard_apply_skips_procs_without_a_pid():
    guard = rtg.BudgetGuard(api_key="x")
    guard.stop_all = True
    guard.remaining_usd = 0.0
    running = {"k": (rtg.RunSpec("qwen3.8-omni-flash", "B0", 1), object())}  # no .pid
    try:
        guard.apply(running)
        assert False, "expected BudgetExhausted"
    except rtg.BudgetExhausted:
        pass  # no crash from the missing pid


def test_main_refuses_to_start_without_a_successful_credit_check(monkeypatch):
    monkeypatch.setattr(rtg, "load_env_file", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(rtg, "start_caffeinate", lambda pid: None)
    assert rtg.main([]) == 2


# ------------------------------------------------------- active-scheduler pointer

def test_poll_step_targets_the_active_scheduler_not_a_stale_one(tmp_path):
    """Regression for the round-1 bug: poll_step must always check whichever
    scheduler main() currently considers active (main pass, then the error
    sweep), not whichever scheduler existed when poll_step was created."""
    budget = rtg.BudgetGuard(api_key=None)  # stop_new/stop_all both False -> apply() no-ops
    watcher = rtg.Tail429Watcher(log_dir=tmp_path)
    seen_running_dicts = []

    def spy_check(running):
        seen_running_dicts.append(running)
        return []
    watcher.check = spy_check

    active = {"sched": None}
    poll_step = rtg.make_poll_step(active, budget, watcher, base_sleep=lambda s: None)

    procs_main, procs_sweep = {}, {}
    main_sched = rtg.Scheduler([("qwen3.8-omni-flash", "B0", 1)], make_launcher(procs_main),
                                sleep_fn=poll_step)
    active["sched"] = main_sched
    main_sched._reap()
    main_sched._fill_slots()
    procs_main["qwen3.8-omni-flash__B0__pass1"].finish(0)
    main_sched.run(poll_interval=0)  # drains; main_sched.running is now {}

    sweep_sched = rtg.Scheduler([("qwen3.8-omni-flash", "B2", 1)], make_launcher(procs_sweep),
                                 sleep_fn=poll_step)
    active["sched"] = sweep_sched
    sweep_sched._reap()
    sweep_sched._fill_slots()
    assert sweep_sched.running  # something is in flight for the sweep

    poll_step(0)  # simulate one poll cycle exactly like Scheduler.run() would call it

    assert seen_running_dicts, "watcher.check was never called"
    assert seen_running_dicts[-1] is sweep_sched.running
    assert seen_running_dicts[-1] is not main_sched.running
    assert seen_running_dicts[-1] != {}


def test_provider_of_component():
    assert rtg.provider_of_component("qwen3.8-omni-flash", "agent") == "openrouter"
    assert rtg.provider_of_component("open-cascade", "agent") == "groq"
    assert rtg.provider_of_component("gemini-3.8-live", "agent") == "google"
    assert rtg.provider_of_component("gpt-realtime-2.1-mini", "agent") == "openai"
    # judge/verifier are OpenRouter no matter the agent under study
    assert rtg.provider_of_component("gemini-3.8-live", "judge") == "openrouter"
    assert rtg.provider_of_component("open-cascade", "verifier") == "openrouter"


def test_spend_by_provider(tmp_path):
    cost_log = tmp_path / "cost_log.jsonl"
    rows = [
        {"agent": "qwen3.8-omni-flash", "components": [
            {"role": "agent", "known_cost_usd": 0.10},
            {"role": "judge", "known_cost_usd": 0.02}]},
        {"agent": "open-cascade", "components": [
            {"role": "agent", "known_cost_usd": 0.05},
            {"role": "judge", "known_cost_usd": 0.02}]},
    ]
    import json
    cost_log.write_text("\n".join(json.dumps(r) for r in rows))
    totals = rtg.spend_by_provider([cost_log])
    assert totals["groq"] == 0.05
    assert round(totals["openrouter"], 4) == round(0.10 + 0.02 + 0.02, 4)


def test_sibling_cost_logs_excludes_its_own_root(tmp_path):
    root = tmp_path / "rp"
    root.mkdir()
    own = root / "cost_log.jsonl"
    own.write_text("{}\n")
    other = tmp_path / "rp_v3"
    other.mkdir()
    other_log = other / "cost_log.jsonl"
    other_log.write_text("{}\n")

    logs = rtg.sibling_cost_logs(root)

    assert own.resolve() not in [p.resolve() for p in logs]
    assert other_log.resolve() in [p.resolve() for p in logs]


def test_spend_by_provider_not_double_counted_via_sibling_cost_logs(tmp_path):
    # Regression: root.parent.glob("rp*/cost_log.jsonl") also matches root's
    # own cost_log.jsonl ("rp*" matches "rp"); main() passes root's cost_log
    # explicitly *and* sibling_cost_logs()'s result, so sibling_cost_logs must
    # exclude root's own log or every dollar in it gets summed twice.
    import json
    root = tmp_path / "rp"
    root.mkdir()
    own = root / "cost_log.jsonl"
    own.write_text(json.dumps({"agent": "qwen3.8-omni-flash",
                                "components": [{"role": "agent", "known_cost_usd": 1.0}]}) + "\n")

    paths = [own, *rtg.sibling_cost_logs(root)]
    assert paths.count(own) == 1

    totals = rtg.spend_by_provider(paths)
    assert totals["openrouter"] == 1.0  # not 2.0


def test_count_429():
    assert rtg.count_429("ok\nok\n") == 0
    assert rtg.count_429("429 Too Many Requests\nretry\n429 again") == 2


def test_tail_429_watcher_detects_burst_only_in_new_content(tmp_path):
    key = "qwen3.8-omni-flash__B0__pass1"
    log_path = tmp_path / f"{key}.log"
    log_path.write_text("starting up\nok\n")
    spec = rtg.RunSpec("qwen3.8-omni-flash", "B0", 1)
    running = {key: (spec, object())}
    watcher = rtg.Tail429Watcher(log_dir=tmp_path, threshold=3)

    assert watcher.check(running) == []  # nothing new, no 429s yet

    with open(log_path, "a") as f:
        f.write("429\n429\n429\n429\n")
    assert watcher.check(running) == [spec]

    # already-read content isn't re-counted on the next check
    assert watcher.check(running) == []

    with open(log_path, "a") as f:
        f.write("429\n")  # below threshold on its own
    assert watcher.check(running) == []


def test_tail_429_watcher_ignores_runs_with_no_log_yet(tmp_path):
    watcher = rtg.Tail429Watcher(log_dir=tmp_path, threshold=1)
    spec = rtg.RunSpec("qwen3.8-omni-flash", "B0", 1)
    running = {spec.key: (spec, object())}
    assert watcher.check(running) == []


# ------------------------------------------------------------- error report

def test_write_errors_md_reports_turns_and_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(rtg, "TEST_DIR", tmp_path)
    folder = tmp_path / "qwen3.8-omni-flash" / "B0"
    folder.mkdir(parents=True)
    import json
    rows = [
        {"case_id": "a", "turn": 1, "sut_error": None, "verdict": "PASS", "judge_rules_hash": rtg.EXPECTED_JUDGE_RULES_HASH},
        {"case_id": "a", "turn": 2, "sut_error": "timeout", "verdict": "ERROR", "judge_rules_hash": rtg.EXPECTED_JUDGE_RULES_HASH},
    ]
    (folder / "pass1.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    text = rtg.write_errors_md(run_list=[("qwen3.8-omni-flash", "B0", 1)], path=tmp_path / "ERRORS.md")
    assert "qwen3.8-omni-flash__B0__pass1" in text
    assert "| 2 | 1 | yes |" in text
    assert (tmp_path / "ERRORS.md").exists()


def test_check_judge_rules_hash_flags_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(rtg, "TEST_DIR", tmp_path)
    folder = tmp_path / "qwen3.8-omni-flash" / "B0"
    folder.mkdir(parents=True)
    import json
    rows = [{"case_id": "a", "turn": 1, "judge_rules_hash": "not-the-prereg-hash"}]
    (folder / "pass1.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert rtg.check_judge_rules_hash("qwen3.8-omni-flash", "B0", 1) is False


def test_filter_run_list_keeps_only_named_agents_in_order():
    rows = rtg.filter_run_list(rtg.RUN_LIST, ["qwen3.8-omni-flash"])
    assert len(rows) == 18 and {a for a, _, _ in rows} == {"qwen3.8-omni-flash"}
    assert rows == [r for r in rtg.RUN_LIST if r[0] == "qwen3.8-omni-flash"]
    assert rtg.filter_run_list(rtg.RUN_LIST, None) == rtg.RUN_LIST


def test_filter_run_list_rejects_unknown_agent():
    import pytest
    with pytest.raises(SystemExit):
        rtg.filter_run_list(rtg.RUN_LIST, ["qwen-typo"])
