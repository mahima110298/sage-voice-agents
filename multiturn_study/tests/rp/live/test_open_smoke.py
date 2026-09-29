"""Live smokes for the open-weight agents (Task 16b; costs cents; skipped unless RP_LIVE=1).

  cd multiturn_study && RP_LIVE=1 python3 -m pytest tests/rp/live/test_open_smoke.py -s
  RP_LIVE_AGENTS=voxtral-small-24b RP_LIVE=1 python3 -m pytest tests/rp/live/test_open_smoke.py -s

ONE dev_v2 case (ecommerce_mt_04) per agent, configs B0 and SAGE (SAGE with the
real Verifier(VERIFIER_MODEL)). dev_v2 only -- never the locked test split.
Each turn prints latency, usage, calls, blocks, text, and the known cost from
rp.runner.price_usage; the summary line gives cost per 100 turns. A rate-limit
(HTTP 429, or the free tier's 502 ResourceExhausted) on a case is retried politely (twice, 20 s then 40 s).
"""
import asyncio
import os
import time

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RP_LIVE") != "1", reason="live API")

_RATE_LIMITED = ("429", "ResourceExhausted", "rate limit", "rate-limit")
OPEN_AGENTS = ("qwen3.8-omni-flash", "voxtral-small-24b", "open-cascade")  # nemotron dropped (report)


def _run_once(agent, cfg_name, case):
    from rp.agents import AGENTS
    from rp.interceptors import build_config
    from rp.prompts import BASE_PROMPT
    from rp.verifier import VERIFIER_MODEL, Verifier
    verifier = Verifier(VERIFIER_MODEL) if cfg_name.startswith("SAGE") else None
    ad = AGENTS[agent]()
    return ad, asyncio.run(ad.run(case, build_config(cfg_name, verifier), BASE_PROMPT))


def _run(agent, cfg_name, case):
    from rp.runner import load_prices, price_usage
    for attempt, wait in enumerate((20, 40, None)):
        ad, turns = _run_once(agent, cfg_name, case)
        if wait is None or not any(m in (t.error or "") for t in turns for m in _RATE_LIMITED):
            break
        print(f"\n[{agent}/{cfg_name}] rate-limited (attempt {attempt + 1}); sleeping {wait}s")
        time.sleep(wait)
    prices = load_prices()
    print(f"\n== {agent} ({ad.model}) / {cfg_name} / {case['id']}")
    total = 0.0
    for t in turns:
        p = price_usage({k: v for k, v in t.usage.items() if k != "safety_valve"}, ad.model, prices)
        total += p["known_cost_usd"]
        print(f"turn {t.turn}: latency_ms={t.latency_ms} guard_ms={t.guard_ms:.2f} "
              f"verifier_calls={t.verifier_calls} error={t.error}")
        print(f"  user:    {t.user_text!r}")
        print(f"  calls:   {[c.as_dict() for c in t.calls]}")
        print(f"  blocked: {t.blocked}")
        print(f"  agent:   {t.agent_text!r}")
        print(f"  usage:   {t.usage}")
        print(f"  cost:    known={p['known_cost_usd']:.6f} cost_usd={p['cost_usd']} "
              f"unknown={p['unknown_fields']} notes={p['notes']}")
    print(f"SUMMARY {agent}/{cfg_name}: known agent cost ${total:.6f} over {len(turns)} turns "
          f"= ${100 * total / len(turns):.4f} per 100 turns; "
          f"latency_ms={[round(t.latency_ms or -1) for t in turns]}")
    return turns


@pytest.mark.parametrize("agent", [a for a in OPEN_AGENTS
                                   if a in (os.environ.get("RP_LIVE_AGENTS") or ",".join(OPEN_AGENTS)).split(",")])
def test_open_agent_one_dev_case(agent):
    from rp.cases import load_split
    case = load_split("dev_v2", only_id=os.environ.get("RP_LIVE_CASE", "ecommerce_mt_04"))[0]
    for cfg_name in (os.environ.get("RP_LIVE_CONFIG") or "B0,SAGE").split(","):
        turns = _run(agent, cfg_name, case)
        assert len(turns) == len(case["steps"])
        assert all(t.error is None for t in turns), [t.error for t in turns]
        assert all(t.user_text for t in turns) and any(t.agent_text or t.calls or t.blocked for t in turns)
        assert all(t.latency_ms is not None and t.latency_ms >= 0 for t in turns)
