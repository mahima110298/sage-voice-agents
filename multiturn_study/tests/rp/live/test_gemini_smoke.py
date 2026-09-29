"""Live smoke test for the Gemini Live adapter (costs cents; skipped unless RP_LIVE=1).

  cd multiturn_study && RP_LIVE=1 python3 -m pytest tests/rp/live/test_gemini_smoke.py -s
Overrides: RP_LIVE_MODEL (default gemini-3.1-flash-live-preview), RP_LIVE_CONFIG
(default SAGE), RP_LIVE_CASE (a dev_v2 case id; default ecommerce_mt_04, a
lookup-then-cancel case, so SAGE's block path is exercised). dev_v2 only --
never the locked test split.
"""
import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RP_LIVE") != "1", reason="live API")


def test_one_dev_case_runs():
    from rp.cases import load_split
    from rp.agents.gemini_live import GeminiLiveAdapter
    from rp.interceptors import build_config
    from rp.prompts import BASE_PROMPT
    model = os.environ.get("RP_LIVE_MODEL", "gemini-3.1-flash-live-preview")
    cfg = os.environ.get("RP_LIVE_CONFIG", "SAGE")
    case = load_split("dev_v2", only_id=os.environ.get("RP_LIVE_CASE", "ecommerce_mt_04"))[0]
    turns = asyncio.run(GeminiLiveAdapter(model).run(case, build_config(cfg, None), BASE_PROMPT))
    print(f"\n== {model} / {cfg} / {case['id']}")
    for t in turns:
        print(f"turn {t.turn}: latency_ms={t.latency_ms} guard_ms={t.guard_ms:.2f} error={t.error}")
        print(f"  user:    {t.user_text!r}")
        print(f"  calls:   {[c.as_dict() for c in t.calls]}")
        print(f"  blocked: {t.blocked}")
        print(f"  agent:   {t.agent_text!r}")
        print(f"  usage:   {t.usage}")
    assert len(turns) == len(case["steps"]) and all(t.error is None for t in turns)
    assert all(t.user_text for t in turns) and any(t.agent_text for t in turns)
