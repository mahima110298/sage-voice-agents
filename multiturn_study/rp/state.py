from __future__ import annotations
from rp.numwords import numbers_in_text, norm
from rp.types import Call, TurnRecord, Decision


class SessionState:
    def __init__(self, backend):
        self.backend = backend
        self.turns: list[TurnRecord] = []
        self.current: TurnRecord | None = None

    def begin_turn(self, n: int, user_text: str) -> None:
        self.current = TurnRecord(turn=n, user_text=user_text or "")

    def record_call(self, call: Call, result: dict) -> None:
        self.current.calls.append(call)
        self.current.results.append((call, result))

    def record_block(self, call: Call, d: Decision) -> None:
        entry = {"call": call.as_dict(), "gate": d.gate, "rule_id": d.rule_id,
                 "reason": d.reason, "source": d.source}
        if d.source == "verifier":
            entry["verdict"] = d.verdict
        self.current.blocked.append(entry)

    def end_turn(self, agent_text: str) -> TurnRecord:
        self.current.agent_text = agent_text or ""
        self.turns.append(self.current)
        done, self.current = self.current, None
        return done

    def prior_agent_text(self) -> str:
        return self.turns[-1].agent_text if self.turns else ""

    def all_user_text(self) -> str:
        parts = [t.user_text for t in self.turns] + ([self.current.user_text] if self.current else [])
        return " ".join(parts)

    def lookup_results(self, tool: str) -> list:
        recs = [t.results for t in self.turns] + ([self.current.results] if self.current else [])
        return [res.get("result") for rs in recs for (c, res) in rs if c.function == tool]

    def mentioned_values(self) -> set[str]:
        """Every number and normalized token said or returned so far, plus session-context
        scalars (the agent sees these in its system prompt, so it can 'mention' them too)."""
        texts = [t.user_text + " " + t.agent_text for t in self.turns]
        if self.current:
            texts.append(self.current.user_text)
        vals = set()
        for tx in texts:
            vals |= numbers_in_text(tx)
            vals |= {norm(w) for w in tx.split()}
        for rs in [t.results for t in self.turns] + ([self.current.results] if self.current else []):
            for _, res in rs:
                vals |= {norm(x) for x in _flatten(res)}
        vals |= {norm(v) for v in self.backend.context_values().values()}
        return {v for v in vals if v}


def _flatten(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _flatten(v)
    elif isinstance(o, list):
        for v in o:
            yield from _flatten(v)
    else:
        yield str(o)
