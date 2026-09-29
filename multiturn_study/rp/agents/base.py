"""Shared agent-adapter interface.

Every agent under study (Gemini Live, OpenAI Realtime, the cascade) is driven by
the same per-case loop, `run_case`, so the only thing that differs between
agents is how audio goes in and text/tool calls come out. An adapter provides:

  connect(case, tools, system_instruction)
      -> async context manager yielding one persistent session for the whole
         case (conversation memory carries across turns inside it).
  async run_turn(session, wav, on_call, state, out) -> None
      Stream one user turn's audio, and for every function call the model makes
      call `on_call(name, args) -> dict` and send that dict back as the tool
      response. Append the agent's spoken/printed text to `out.text` as it
      arrives (so a turn that errors keeps what was captured). Set
      `state.current.latency_ms` (end of user audio -> first model output) and
      add token counts into `state.current.usage`. Raise on failure; raise
      asyncio.TimeoutError for a turn timeout. A tripped safety valve is not a
      failure: set `state.current.usage["safety_valve"]` and return normally.

Every tool call goes through `handle_call`, i.e. through the Interceptor
(SAGE or a baseline) before it can reach the mock backend.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from rp import asr
from rp.backend import MockBackend
from rp.state import SessionState
from rp.tools import schemas_for
from rp.types import Call, TurnRecord

SKIPPED = "skipped_after_prior_turn_error"


def handle_call(name, args, state, interceptor, backend):
    """Run one tool call through the interceptor. Returns (tool response, executed?)."""
    call = Call(name, dict(args or {}))
    t0 = time.perf_counter()
    d = interceptor.on_call(call, state)
    state.current.guard_ms += (time.perf_counter() - t0) * 1000
    state.current.verifier_calls += int(d.verifier_called)
    if d.allow:
        res = backend.respond(call)
        state.record_call(call, res)
        return res, True
    state.record_block(call, d)
    return d.message, False


@dataclass
class TurnOutput:
    text: str = ""


def system_instruction(system_prompt: str, interceptor, backend) -> str:
    return system_prompt + interceptor.prompt_suffix() + backend.session_context()


def _err(e: BaseException) -> str:
    if isinstance(e, asyncio.TimeoutError):
        return "timeout"
    return str(e)[:300] or type(e).__name__


async def run_case(adapter, case: dict, interceptor, system_prompt: str,
                   case_timeout_s: float | None = None) -> list[TurnRecord]:
    """Run every turn of `case` through one persistent adapter session.

    Returns one TurnRecord per step. A turn that errors keeps the text and calls
    captured so far with `error` set; every later turn is returned with
    error=SKIPPED (the session is not trusted after an error). A failure outside
    any turn (connect, or the whole-case timeout) is recorded on the turn in
    progress (or the first unfinished turn) as "session_error: ...".
    """
    backend = MockBackend(case)
    state = SessionState(backend)
    tools = schemas_for(case["domain"])
    instr = system_instruction(system_prompt, interceptor, backend)
    steps = case["steps"]
    out = TurnOutput()

    def on_call(name, args):
        return handle_call(name, args, state, interceptor, backend)[0]

    async def _run():
        nonlocal out
        async with adapter.connect(case, tools, instr) as session:
            for step in steps:
                turn = step["turn"]
                wav = case["_folder"] / f"input_turn{turn}.wav"
                state.begin_turn(turn, asr.transcribe(wav))
                out = TurnOutput()
                try:
                    await adapter.run_turn(session, wav, on_call, state, out)
                except Exception as e:  # noqa: BLE001 -- recorded, never silently passed
                    state.current.error = _err(e)
                    state.end_turn(out.text)
                    return
                state.end_turn(out.text)

    try:
        if case_timeout_s is None:
            await _run()
        else:
            await asyncio.wait_for(_run(), timeout=case_timeout_s)
    except Exception as e:  # noqa: BLE001
        msg = f"session_error: {_err(e)[:200]}"
        if state.current is not None:        # failed mid-turn (e.g. whole-case timeout)
            state.current.error = msg
            state.end_turn(out.text)
        elif len(state.turns) < len(steps):  # failed before a turn began (e.g. connect)
            state.turns.append(TurnRecord(turn=steps[len(state.turns)]["turn"], error=msg))

    records = list(state.turns)
    for step in steps[len(records):]:
        records.append(TurnRecord(turn=step["turn"], error=SKIPPED))
    return records
