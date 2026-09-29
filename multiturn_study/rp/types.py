from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Call:
    function: str
    args: dict = field(default_factory=dict)

    def key(self) -> tuple:
        return (self.function, tuple(sorted((k, str(v)) for k, v in self.args.items())))

    def as_dict(self) -> dict:
        return {"function": self.function, "args": dict(self.args)}


@dataclass
class TurnRecord:
    turn: int
    user_text: str = ""
    agent_text: str = ""
    calls: list = field(default_factory=list)        # executed Calls
    blocked: list = field(default_factory=list)      # dicts: call, gate, rule_id, reason
    results: list = field(default_factory=list)      # (Call, result dict)
    latency_ms: float | None = None
    guard_ms: float = 0.0
    verifier_calls: int = 0
    usage: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class Decision:
    allow: bool
    gate: str | None = None
    rule_id: str | None = None
    reason: str = ""
    message: dict | None = None
    verifier_called: bool = False
    # Block provenance (Task 19a): "rules" when the gate checker (or a rule-based
    # baseline like B3) produced the block, "verifier" when a non-ok verifier
    # verdict did. `verdict` is only meaningful for source == "verifier": the
    # verifier's own "ambiguous" | "wrong_target" | "missing_value" label.
    source: str = ""
    verdict: str | None = None
