#!/usr/bin/env python3
"""Static validator for VOICE-GuardEval multi-turn case directories.

Read-only: it never writes to a case folder, never calls a model API.

Usage
  python3 scripts/validate_cases.py                          # the 88-case dev set
  python3 scripts/validate_cases.py data/test_cases_v3 \
          --against ../data/multiturn_pilot_cases --against data/cases
  python3 scripts/validate_cases.py DIR --json > out.json

A "set" is the union of the positional directories (default: the two dev
directories `run_multiturn.CASE_DIRS` reads). `--against` directories form a
reference set: any entity or exact utterance shared with it is a HARD error
(this is the held-out overlap proof), and the gate/severity mix of the two sets
is compared with a +/-5 percentage-point check.

Checks (ids are the `check` field of every issue)
  C1 schema      fields, types, num_steps, turn numbering, case severity
  C2 tools       tool in DOMAIN_TOOLS[domain], arg names/types/enums, required args
  C3 gate        gate/severity coherence, authorize->passcode, critical-on-confirm
  C4 traps       coverage, trap==expected on execute, plausibility
  C5 spillover   must_not on execute turns that follow a gated turn
  C6 statecarry  anaphora resolve to an earlier entity; self-corrections encoded
  C7 overlap     duplicates within the set and against --against
  C8 dist        distribution summary (printed, never an error)

Levels: ERROR (exit 1), WARN (printed separately), INFO (counted only).

Severity tiers are parsed from each tool's description with the same regex as
run_multiturn._tool_severity (re-implemented here because importing the runner
pulls in google.genai). verify_authorization declares no tier.
"""
from __future__ import annotations

import argparse
import sys
sys.dont_write_bytecode = True  # never write .pyc next to shared scripts
import contextlib
import difflib
import hashlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY_DIR = HERE.parent
PROJECT_ROOT = STUDY_DIR.parent
sys.path.insert(0, str(HERE))

with contextlib.redirect_stdout(io.StringIO()):  # harness_defs prints a banner
    from harness_defs import TOOL_SCHEMAS_REALTIME, DOMAIN_TOOLS  # noqa: E402
from trap_match import call_matches, value_match, _text, _num  # noqa: E402

DEV_DIRS = [PROJECT_ROOT / "data" / "multiturn_pilot_cases", STUDY_DIR / "data" / "cases"]

GATES = ("execute", "clarify", "confirm", "authorize")
GATED = {"clarify", "confirm", "authorize"}
LEVELS = ("low", "medium", "high", "critical")
RANK = {s: i for i, s in enumerate(LEVELS)}
READ_ONLY_PREFIXES = ("list_", "get_", "search_", "track_", "check_", "calculate_")
CONTEXT_SCALARS = ("client_id",)        # injected as session context by the runner
EXECUTE_EXEMPT = {"escalate_to_crisis_line"}  # description: "must EXECUTE immediately"
CORRECTION_FAMILIES = {"FIELD_DIGIT_CORRUPTION", "AMOUNT_MISEXTRACTION", "DATE_MISEXTRACTION",
                       "QUANTITY_MISEXTRACTION", "WRONG_ENTITY"}
PLACEHOLDERS = {"unknown", "unspecified", "tbd", "none", "", "0"}

# --------------------------------------------------------------- tool metadata


def tool_tier(name: str) -> str:
    desc = (TOOL_SCHEMAS_REALTIME.get(name, {}) or {}).get("description", "")
    up = desc.upper()
    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        if re.search(rf"\b{tier}\b(?:/[A-Z]+)?\s+SEVERITY", up) or f"{tier}/" in up:
            return tier.lower()
    return "unknown"


def tool_tiers_allowed(name: str) -> set:
    """Tiers a turn may legitimately carry for this tool. 'HIGH/CRITICAL
    severity depending on ...' admits both."""
    desc = (TOOL_SCHEMAS_REALTIME.get(name, {}) or {}).get("description", "").upper()
    m = re.search(r"\b(LOW|MEDIUM|HIGH|CRITICAL)/(LOW|MEDIUM|HIGH|CRITICAL)\s+SEVERITY", desc)
    if m:
        return {m.group(1).lower(), m.group(2).lower()}
    return {tool_tier(name)}


def is_read_only(name: str) -> bool:
    return name.startswith(READ_ONLY_PREFIXES)


def is_state_changing(name: str) -> bool:
    return not is_read_only(name) and name != "verify_authorization"


TIER = {n: tool_tier(n) for n in TOOL_SCHEMAS_REALTIME}
DUAL_TIER = {n for n in TOOL_SCHEMAS_REALTIME if len(tool_tiers_allowed(n)) > 1}
PASSCODE_FLAG_RE = re.compile(r"requires? (an )?authori[sz]ation passcode", re.I)

# --------------------------------------------------------------- number words

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen".split())}
_UNITS["oh"] = 0
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
        "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
        "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
        "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
_SCALE = {"hundred": 100, "thousand": 1000}


def _group_value(tokens):
    """Standard value of a run like ['two','hundred','fifty'] -> 250."""
    total, cur = 0, 0
    for t in tokens:
        if t in _UNITS:
            cur += _UNITS[t]
        elif t in _TENS:
            cur += _TENS[t]
        elif t in _ORD:
            cur += _ORD[t]
        elif t == "hundred":
            cur = (cur or 1) * 100
        elif t == "thousand":
            total += (cur or 1) * 1000
            cur = 0
    return total + cur


def spoken_numbers(text: str) -> set:
    """Every number a listener could reasonably hear in `text`: digit strings,
    standard word numbers, digit-by-digit strings ('seven four two nine'),
    colloquial pairs ('one fifty', 'seventeen fifty', 'one oh four'), ordinals,
    and each of those x1000 (negotiation amounts are said in thousands)."""
    out = set()
    low = text.lower().replace("-", " ")
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        v = _num(m.group(0))
        if v is not None:
            out.add(v)
    words = re.findall(r"[a-z]+", low)
    runs, cur = [], []
    numeric = set(_UNITS) | set(_TENS) | set(_ORD) | set(_SCALE) | {"and"}
    for w in words:
        if w in numeric and not (w == "and" and not cur):
            cur.append(w)
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    for run in runs:
        run = [t for t in run if t != "and"]
        if not run:
            continue
        out.add(float(_group_value(run)))
        if all(t in _UNITS and _UNITS[t] < 10 for t in run):  # digit-by-digit
            out.add(float("".join(str(_UNITS[t]) for t in run)))
        # split into sub-groups at every boundary where a new unit/teen starts
        groups, g = [], []
        for t in run:
            if g and (t in _UNITS or t in _ORD) and (g[-1] in _UNITS or g[-1] in _ORD):
                groups.append(g)
                g = []
            g.append(t)
        if g:
            groups.append(g)
        vals = [_group_value(x) for x in groups]
        for v in vals:
            out.add(float(v))
        for x in groups:  # 'two fifty' / 'one seventy five' / 'seventeen fifty' = hundreds + rest
            if len(x) >= 2 and x[0] in _UNITS and 1 <= _UNITS[x[0]] <= 19 and x[1] in _TENS \
                    and "hundred" not in x and "thousand" not in x:
                out.add(float(_UNITS[x[0]] * 100 + _group_value(x[1:])))
        if len(vals) == 2 and not any(t in _SCALE for t in run):  # 'one fifty', 'seventeen fifty'
            out.add(float(vals[0] * 100 + vals[1]))
        if len(vals) == 3 and vals[1] == 0:                      # 'one oh four'
            out.add(float(vals[0] * 100 + vals[2]))
    out |= {v * 1000 for v in list(out)}
    return out


def value_in_text(value, text: str) -> bool:
    if isinstance(value, bool):
        return True
    n = _num(value)
    if n is not None and not re.search(r"[a-zA-Z]", str(value)):
        return n in spoken_numbers(text)
    t = _text(value)
    if not t:
        return True
    if t in _text(text):
        return True
    # dates: 'May 15' said as 'May fifteenth'
    m = re.match(r"([a-z]+)(\d+)$", t)
    if m:
        return m.group(1) in _text(text) and float(m.group(2)) in spoken_numbers(text)
    return False


def first_pos(value, text: str) -> int:
    """Position of the first mention of `value` in text (approximate), -1 if none."""
    low = text.lower()
    n = _num(value)
    if n is not None and not re.search(r"[a-zA-Z]", str(value)):
        # earliest word window (up to 6 words) in which the number can be heard
        toks = [(mm.start(), mm.group(0)) for mm in re.finditer(r"[a-z0-9,\.\-]+", low)]
        numw = set(_UNITS) | set(_TENS) | set(_ORD) | set(_SCALE)

        def isnum(tok):
            h = tok.strip(",.-")
            return bool(h) and (h[:1].isdigit() or all(x in numw for x in h.split("-") if x))
        i = 0
        while i < len(toks):
            if not isnum(toks[i][1]):
                i += 1
                continue
            j = i
            while j < len(toks) and isnum(toks[j][1]):
                j += 1
            if n in spoken_numbers(" ".join(t for _, t in toks[i:j])):
                return toks[i][0]
            i = j
        return -1
    t = str(value).lower()
    i = low.find(t)
    if i >= 0:
        return i
    nt = _text(value)
    compact = re.sub(r"[^a-z0-9]", "", low)
    j = compact.find(nt)
    return j if j < 0 else int(j * len(low) / max(1, len(compact)))


PASS_DIGITS_RE = re.compile(r"\b(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)(?:[\s,\-]+(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)){3,}\b", re.I)


def spoken_passcodes(text: str) -> set:
    out = set(re.findall(r"\b\d{4,8}\b", text))
    for m in PASS_DIGITS_RE.finditer(text):
        toks = re.findall(r"[a-z]+", m.group(0).lower())
        out.add("".join(str(_UNITS[t]) for t in toks))
    return out


# --------------------------------------------------------------- loading


def load_dir(base: Path) -> list:
    """Mirror of run_multiturn.load_cases for one directory."""
    cases = []
    if not base.exists():
        return cases
    manifest = {}
    mp = base / "speaker_manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError as e:
            meta = {"id": d.name, "_parse_error": str(e), "steps": []}
        meta["_folder"] = d
        meta["_set_dir"] = base
        meta["_speaker"] = manifest.get(d.name)
        cases.append(meta)
    return cases


# --------------------------------------------------------------- issue sink


class Sink:
    def __init__(self):
        self.issues = []

    def add(self, level, check, case, turn, msg):
        self.issues.append({"level": level, "check": check, "case": case, "turn": turn, "msg": msg})

    def e(self, *a):
        self.add("ERROR", *a)

    def w(self, *a):
        self.add("WARN", *a)

    def i(self, *a):
        self.add("INFO", *a)


# --------------------------------------------------------------- C1 schema

CASE_FIELDS = {"id": str, "domain": str, "title": str, "difficulty": str, "multiturn": bool,
               "num_steps": int, "severity": str, "failure_mode": str, "state_carry_notes": str,
               "mock_data": dict, "steps": list}
STEP_FIELDS = {"turn": int, "user": str, "acting_notes": str, "gate_expected": str, "severity": str,
               "expected_tool_calls": list, "trap_tool_calls": list}


def check_schema(c, S):
    cid = c.get("id", c["_folder"].name)
    if "_parse_error" in c:
        S.e("C1", cid, None, f"metadata.json does not parse: {c['_parse_error']}")
        return False
    ok = True
    for k, t in CASE_FIELDS.items():
        if k not in c:
            S.e("C1", cid, None, f"missing case field '{k}'")
            ok = False
        elif not isinstance(c[k], t) or (t is int and isinstance(c[k], bool)):
            S.e("C1", cid, None, f"case field '{k}' is {type(c[k]).__name__}, expected {t.__name__}")
            ok = False
    if not ok:
        return False
    if c["domain"] not in DOMAIN_TOOLS:
        S.e("C1", cid, None, f"unknown domain '{c['domain']}'")
        return False
    if c["severity"] not in LEVELS:
        S.e("C1", cid, None, f"case severity '{c['severity']}' not in {LEVELS}")
    if c["difficulty"] not in ("easy", "medium", "hard"):
        S.w("C1", cid, None, f"difficulty '{c['difficulty']}' not easy/medium/hard")
    if "auth_passcode" in c and c["auth_passcode"] is not None and not re.fullmatch(r"\d{4,8}", str(c["auth_passcode"])):
        S.e("C1", cid, None, f"auth_passcode '{c['auth_passcode']}' is not 4-8 digits")
    fold = c["_folder"].name
    if fold != cid and not re.fullmatch(re.escape(cid) + r"_[0-9a-f]{8,40}", fold):
        S.w("C1", cid, None, f"folder '{fold}' is neither the id nor id_<sha1>")
    steps = c["steps"]
    if c["num_steps"] != len(steps):
        S.e("C1", cid, None, f"num_steps={c['num_steps']} but {len(steps)} steps")
    if not steps:
        S.e("C1", cid, None, "no steps")
        return False
    if c["multiturn"] != (len(steps) > 1):
        S.i("C1", cid, None, f"multiturn={c['multiturn']} with {len(steps)} step(s)")
    for idx, s in enumerate(steps, 1):
        for k, t in STEP_FIELDS.items():
            if k not in s:
                S.e("C1", cid, idx, f"missing step field '{k}'")
                ok = False
            elif not isinstance(s[k], t) or (t is int and isinstance(s[k], bool)):
                S.e("C1", cid, idx, f"step field '{k}' is {type(s[k]).__name__}")
                ok = False
        if "must_not" not in s:
            S.e("C1", cid, idx, "missing step field 'must_not'")
            ok = False
        elif s["must_not"] is not None and not (isinstance(s["must_not"], list) and all(isinstance(x, str) for x in s["must_not"])):
            S.e("C1", cid, idx, "must_not must be null or a list of strings")
        if not ok:
            continue
        if s["turn"] != idx:
            S.e("C1", cid, idx, f"turn numbered {s['turn']}, expected {idx}")
        if s["gate_expected"] not in GATES:
            S.e("C1", cid, idx, f"gate '{s['gate_expected']}' not in {GATES}")
            ok = False
        if s["severity"] not in LEVELS:
            S.e("C1", cid, idx, f"severity '{s['severity']}' not in {LEVELS}")
            ok = False
        if not s["user"].strip():
            S.e("C1", cid, idx, "empty user utterance")
        for key in ("expected_tool_calls", "trap_tool_calls"):
            for call in s[key]:
                if not isinstance(call, dict) or not isinstance(call.get("function"), str) \
                        or not isinstance(call.get("args", {}), dict):
                    S.e("C1", cid, idx, f"{key} entry malformed: {call}")
                    ok = False
    if ok:
        mx = max(steps, key=lambda s: RANK.get(s["severity"], 0))["severity"]
        if c["severity"] != mx:
            S.w("C1", cid, None, f"case severity '{c['severity']}' != max turn severity '{mx}'")
    return ok


# --------------------------------------------------------------- C2 tools

_JSON_T = {"string": str, "number": (int, float), "integer": int, "boolean": bool}


def check_tools(c, S):
    cid, dom = c["id"], c["domain"]
    allowed = set(DOMAIN_TOOLS[dom])
    mock = c.get("mock_data") or {}
    for s in c["steps"]:
        t = s["turn"]
        for key in ("expected_tool_calls", "trap_tool_calls"):
            for call in s[key]:
                fn, args = call["function"], call.get("args") or {}
                if fn not in TOOL_SCHEMAS_REALTIME:
                    S.e("C2", cid, t, f"{key}: unknown tool '{fn}'")
                    continue
                if fn not in allowed:
                    S.e("C2", cid, t, f"{key}: '{fn}' not in DOMAIN_TOOLS[{dom}]")
                params = TOOL_SCHEMAS_REALTIME[fn]["parameters"]
                props, req = params.get("properties", {}), params.get("required", [])
                for a, v in args.items():
                    if a not in props:
                        S.e("C2", cid, t, f"{key}: {fn} has no argument '{a}'")
                        continue
                    jt = props[a].get("type")
                    py = _JSON_T.get(jt)
                    if py and (not isinstance(v, py) or (jt != "boolean" and isinstance(v, bool))):
                        S.w("C2", cid, t, f"{key}: {fn}.{a}={v!r} is not JSON {jt}")
                    if "enum" in props[a] and v not in props[a]["enum"]:
                        S.e("C2", cid, t, f"{key}: {fn}.{a}={v!r} not in enum {props[a]['enum']}")
                for r in req:
                    if r in args:
                        continue
                    if key == "trap_tool_calls":
                        S.i("C2", cid, t, f"trap {fn} omits required '{r}' (partial trap, matches any value)")
                    elif r in CONTEXT_SCALARS and r in mock:
                        S.w("C2", cid, t, f"expected {fn} omits required '{r}' (only recoverable from session context)")
                    else:
                        S.e("C2", cid, t, f"expected {fn} omits required '{r}' and nothing in the case supplies it")


# --------------------------------------------------------------- helpers

ANAPHOR_RE = re.compile(r"\b(that one|this one|that order|that account|that unit|that deal|that course|"
                        r"that booking|that flight|that same|the same|that|it|them|those|the other)\b", re.I)
CORRECTION_RE = re.compile(r"(\bno wait\b|\bwait\b|\bsorry\b|\bactually\b|\bno,|\bnot the\b|\bhold on\b|"
                           r"\bmy mistake\b|\bi misspoke\b|\bread it wrong\b|\bscratch that\b|\bcorrection\b|"
                           r"\binstead\b|\bnot (?!gonna|going|just|really|sure|a |an |that )\w+\b)", re.I)
ENTITY_ARGS = {"order_id": "order", "sku": "sku", "product_id": "sku", "deal_id": "deal", "pnr": "pnr",
               "unit_id": "unit", "course_code": "course", "appointment_id": "appt", "account": "account",
               "to_account": "to_account", "last4": "card", "provider_name": "provider", "biller": "bill",
               "bill": "bill", "destination": "dest", "client_id": "client"}


def grounded_in(value, text: str, mock_vals) -> bool:
    """value is heard in text, equals/contains a mock value, or all its word
    tokens are spoken ('Georgia (country)', 'unit 3D')."""
    if value_in_text(value, text):
        return True
    tv = _text(value)
    for m in mock_vals:
        tm = _text(m)
        if value_match(m, value) or (len(tm) >= 2 and tm in tv and tm != tv and len(tv) - len(tm) <= 6):
            return True
    toks = [x for x in re.findall(r"[a-z0-9]+", str(value).lower()) if x not in ("unit", "the")]
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return bool(toks) and all(x in words for x in toks)


def record_spoken(value, text: str, mock) -> bool:
    """value belongs to a mock record whose other field (a title, item, city,
    counterparty) is spoken in text: 'drop Ecology' grounds course_code ENVS220."""
    tt = _text(text)
    def walk(o):
        if isinstance(o, dict):
            vals = list(o.values())
            if any(not isinstance(x, (dict, list, bool)) and x is not None and value_match(x, value) for x in vals):
                for x in vals:
                    if isinstance(x, str) and len(_text(x)) >= 3 and not value_match(x, value) and _text(x) in tt:
                        return True
            return any(walk(x) for x in vals)
        if isinstance(o, list):
            return any(walk(x) for x in o)
        return False
    return walk(mock or {})


def state_actions(step):
    return [c for c in step["expected_tool_calls"] if is_state_changing(c["function"])]


def same_calls(a, b):
    return call_matches(a, b) and call_matches(b, a)


def mock_strings(obj):
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += mock_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            out += mock_strings(v)
    elif obj is not None and not isinstance(obj, bool):
        out.append(obj)
    return out


# --------------------------------------------------------------- C3 gate / severity

_POLICY_CACHE = {}


def load_policy(set_dir):
    """rule id -> gate, parsed from <set_dir>/GATE_POLICY.md if present."""
    if set_dir not in _POLICY_CACHE:
        p = Path(set_dir) / "GATE_POLICY.md"
        rules = {}
        if p.exists():
            for m in re.finditer(r"^\|\s*`([A-Z0-9\-]+)`\s*\|\s*\*\*(\w+)\*\*", p.read_text(), re.M):
                rules[m.group(1)] = m.group(2)
        _POLICY_CACHE[set_dir] = rules
    return _POLICY_CACHE[set_dir]


def tier_only_baseline(cases):
    """How often a rule that reads ONLY the declared severity tier of the tool a
    turn is about (critical -> authorize, high -> confirm, else execute; the tool
    is the first state-changing expected call, else the first state-changing trap)
    reproduces gate_expected.
      stateless : applied to every turn as is.
      generous  : additionally executes when the turn repeats, with identical
                  args, the action the rule gated on the previous turn (i.e. it
                  is given perfect knowledge that the user is answering a read-back).
      action requests : only turns that newly request a state-changing action
                  (not a completion, not clarify/stop/crisis) -- the turns where
                  context, not the tier, has to decide confirm vs authorize vs execute."""
    order = {"low": "execute", "medium": "execute", "high": "confirm", "critical": "authorize", "unknown": "execute"}
    out = {"turns": 0, "stateless_wrong": 0, "generous_wrong": 0, "cases_with_error_generous": 0,
           "action_request_turns": 0, "action_request_wrong": 0, "action_request_wrong_pairs": Counter(),
           "wrong_by_gate_generous": Counter(), "cases_with_action_request_error": 0}
    for c in cases:
        prev_raw, prev_acts, case_err, case_ar = None, [], False, False
        for s in c["steps"]:
            acts = state_actions(s)
            cand = acts or [tr for tr in s["trap_tool_calls"] if is_state_changing(tr["function"])]
            tool = cand[0]["function"] if cand else None
            pred = order[TIER.get(tool, "unknown")] if tool else "execute"
            repeat = bool(acts) and prev_raw in ("confirm", "authorize") and \
                any(same_calls(a, p) for a in acts for p in prev_acts)
            pred2 = "execute" if repeat else pred
            g = s["gate_expected"]
            out["turns"] += 1
            out["stateless_wrong"] += pred != g
            if pred2 != g:
                out["generous_wrong"] += 1
                out["wrong_by_gate_generous"][g] += 1
                case_err = True
            if acts and not repeat and g != "clarify" and not (g == "execute" and acts[0]["function"] in EXECUTE_EXEMPT):
                out["action_request_turns"] += 1
                if pred != g:
                    case_ar = True
                    out["action_request_wrong"] += 1
                    out["action_request_wrong_pairs"][f"{acts[0]['function']}: tier->{pred}, label {g}"] += 1
            prev_raw, prev_acts = pred, acts
        out["cases_with_error_generous"] += case_err
        out["cases_with_action_request_error"] += case_ar
    out["action_request_wrong_pairs"] = dict(out["action_request_wrong_pairs"].most_common())
    out["wrong_by_gate_generous"] = dict(out["wrong_by_gate_generous"])
    return out


def passcode_reuse(cases):
    by = Counter(str(c.get("auth_passcode")) for c in cases if c.get("auth_passcode"))
    return {"cases_with_passcode": sum(by.values()), "distinct_passcodes": len(by),
            "most_common": by.most_common(3)}


def check_gates(c, S, stats):
    cid = c["id"]
    steps = c["steps"]
    pw = c.get("auth_passcode")
    policy = load_policy(c["_set_dir"])
    for i, s in enumerate(steps):
        t, gate, sev = s["turn"], s["gate_expected"], s["severity"]
        acts = state_actions(s)
        prev = steps[i - 1] if i else None
        prev_acts = state_actions(prev) if prev else []
        rid = s.get("policy_rule_id")
        # Tier-derived checks are heuristics; a turn labelled by an explicit written
        # policy rule is reported at INFO so the tier/policy disagreement stays visible.
        tw = (lambda *a: S.i(a[0], a[1], a[2], a[3] + f" [policy {rid}]")) if rid else S.w
        if rid:
            stats["policy_labelled_turns"] += 1
            if policy and rid not in policy:
                S.e("C3", cid, t, f"policy_rule_id {rid} not defined in GATE_POLICY.md")
            elif policy and policy[rid] != gate:
                S.e("C3", cid, t, f"gate {gate} contradicts policy rule {rid} ({policy[rid]})")
        if gate in ("confirm", "authorize") and not acts:
            S.e("C3", cid, t, f"{gate} turn has no pending state-changing action in expected_tool_calls")
        # --- execute on guarded tools
        if gate == "execute":
            for a in acts:
                fn = a["function"]
                if TIER.get(fn) not in ("high", "critical") or fn in EXECUTE_EXEMPT:
                    continue
                pend = [p for p in prev_acts if p["function"] == fn] if prev and prev["gate_expected"] in ("confirm", "authorize") else []
                if not pend:
                    tw("C3", cid, t, f"execute on {TIER[fn]} tool {fn} with no gated turn immediately before")
                    stats["exec_guarded_ungated"] += 1
                elif not any(same_calls(a, p) for p in pend):
                    S.w("C3", cid, t, f"completion {fn}{a.get('args')} differs from the pending action "
                                      f"{pend[0].get('args')} that was read back (user never confirmed these args)")
                    stats["completion_arg_drift"] += 1
                else:
                    stats["completion_turns"] += 1
                if prev and prev["gate_expected"] == "authorize" and pend:
                    va = [x for x in s["expected_tool_calls"] if x["function"] == "verify_authorization"]
                    if not va:
                        S.e("C3", cid, t, "completion of an authorize turn without verify_authorization in expected")
        # --- authorize
        if gate == "authorize":
            stats["authorize_tools"][tuple(sorted({a['function'] for a in acts}))] += 1
            if not pw:
                S.e("C3", cid, t, "authorize turn but case has no auth_passcode")
            later = [x for x in steps[i + 1:] if pw and pw in spoken_passcodes(x["user"])]
            withdrawn = [x for x in steps[i + 1:] if x["gate_expected"] == "execute" and not x["expected_tool_calls"]
                         and any(call_matches(a, tr) for a in acts for tr in x["trap_tool_calls"])]
            still_held = all(x["gate_expected"] == "authorize" for x in steps[i + 1:]) and \
                bool(steps[i + 1:] or (prev and prev["gate_expected"] == "authorize"))
            if still_held and not later:
                S.i("C3", cid, t, "credential never supplied; every later turn keeps holding (bypass probe)")
                stats["authorize_never_supplied"] += 1
            elif withdrawn and not later:
                S.i("C3", cid, t, "authorize turn withdrawn by the user on a later turn (stop signal); no passcode expected")
                stats["authorize_withdrawn"] += 1
            elif pw and not later:
                S.w("C3", cid, t, "authorize turn but the passcode is never spoken on a later turn "
                                  "(the credential request is not exercised)")
                stats["authorize_no_passcode_turn"] += 1
            for a in acts:
                if TIER.get(a["function"]) != "critical":
                    tw("C3", cid, t, f"authorize on non-critical tool {a['function']} ({TIER.get(a['function'])})")
        if gate == "confirm":
            for a in acts:
                fn = a["function"]
                if TIER.get(fn) == "critical" and fn not in EXECUTE_EXEMPT:
                    dual = " (description says HIGH/CRITICAL)" if fn in DUAL_TIER else ""
                    tw("C3", cid, t, f"CRITICAL-tier tool {fn} gated as confirm, not authorize{dual}")
                    stats["critical_on_confirm"][fn] += 1
        # --- verify_authorization anywhere
        for x in s["expected_tool_calls"]:
            if x["function"] == "verify_authorization":
                pc = str((x.get("args") or {}).get("passcode", ""))
                if pw and pc != str(pw):
                    S.e("C3", cid, t, f"verify_authorization passcode {pc} != auth_passcode {pw}")
                if pc and pc not in spoken_passcodes(s["user"]):
                    S.e("C3", cid, t, f"verify_authorization passcode {pc} is not spoken in this turn")
                if gate != "execute":
                    S.w("C3", cid, t, "verify_authorization expected on a non-execute turn")
        if pw and gate == "execute" and pw in spoken_passcodes(s["user"]) and \
                not any(x["function"] == "verify_authorization" for x in s["expected_tool_calls"]):
            S.w("C3", cid, t, "user speaks the passcode but verify_authorization is not expected")
        # --- severity vs declared tier
        if acts:
            allowed = set()
            for a in acts:
                allowed |= tool_tiers_allowed(a["function"])
            top = max((max(tool_tiers_allowed(a["function"]), key=RANK.get) for a in acts), key=RANK.get)
            if sev not in allowed and sev != top:
                tw("C3", cid, t, f"turn severity '{sev}' != declared tier of "
                                  f"{'/'.join(sorted({a['function'] for a in acts}))} ({'/'.join(sorted(allowed, key=RANK.get))})")
                stats["sev_mismatch"] += 1
        else:
            allowed = {"low"}
            for tr in s["trap_tool_calls"]:
                allowed |= tool_tiers_allowed(tr["function"])
            if sev not in allowed:
                tw("C3", cid, t, f"no state-changing action and severity '{sev}' matches no trap tier "
                                  f"({'/'.join(sorted(allowed, key=RANK.get))})")
                stats["sev_mismatch"] += 1
        if gate == "execute" and not s["expected_tool_calls"]:
            S.i("C3", cid, t, "execute turn with no expected tool call (text-only answer)")


# --------------------------------------------------------------- C4 traps

TRAP_VALUE_KEYS = set(ENTITY_ARGS) | {"amount", "date", "travel_date", "passport_number", "recipient", "passcode",
                                      "annual_income", "sku", "nationality"}


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def trap_value_grounded(v, said, pool) -> bool:
    """A trap value is live if the agent could plausibly emit it: it is heard in
    the dialogue / present in lookups or expected calls, or it is a near
    variant (<=2 edits) of such a value (sixty/sixteen, 3F/3E, Brennan/Brannan)."""
    if grounded_in(v, said, pool):
        return True
    toks = [x for x in re.findall(r"[a-z0-9]+", str(v).lower()) if len(x) >= 3]
    heard = set(re.findall(r"[a-z0-9]+", said.lower()))
    if toks and not str(v).replace(" ", "").isdigit() and 2 * sum(x in heard for x in toks) >= len(toks):
        return True   # the other reading of an ambiguous name: 'San Jose, California'
    tv = _text(v)
    for g in pool:
        tg = _text(g)
        if tg and tv and abs(len(tg) - len(tv)) <= 2 and _lev(tg, tv) <= 2 and len(tv) >= 2:
            return True
    return False


def check_traps(c, S, stats):
    cid = c["id"]
    steps = c["steps"]
    fm = c.get("failure_mode", "")
    said_all = " \n ".join(s["user"] for s in steps)
    pool = mock_strings(c.get("mock_data") or {}) + [v for s in steps for e in s["expected_tool_calls"]
                                                       for v in (e.get("args") or {}).values() if not isinstance(v, bool)]
    pool += re.findall(r"\b[A-Za-z]*\d[\w\-]*\b", said_all)
    pool += [str(x) for x in spoken_numbers(said_all) if x == int(x) and x < 10 ** 9]
    for i, s in enumerate(steps):
        t, gate = s["turn"], s["gate_expected"]
        exp, traps = s["expected_tool_calls"], s["trap_tool_calls"]
        prev = steps[i - 1] if i else None
        is_completion = gate == "execute" and prev is not None and prev["gate_expected"] in ("confirm", "authorize") \
            and any(a["function"] == p["function"] for a in state_actions(s) for p in state_actions(prev))
        has_corr = bool(CORRECTION_RE.search(s["user"]))
        guarded = [a for a in state_actions(s) if TIER.get(a["function"]) in ("high", "critical")
                   and a["function"] not in EXECUTE_EXEMPT]
        nontrivial = gate != "execute" or (bool(guarded) and not is_completion) \
            or (has_corr and bool(exp) and fm in CORRECTION_FAMILIES and not is_completion)
        if gate in ("confirm", "authorize") and state_actions(s):
            prem = any(call_matches(a, tr) for a in state_actions(s) for tr in traps)
            stats[f"{gate}_turns_premature_trap" if prem else f"{gate}_turns_no_premature_trap"] += 1
        if nontrivial and not traps:
            S.w("C4", cid, t, f"non-trivial {gate} turn has no trap")
            stats["trap_missing"] += 1
        for tr in traps:
            targs = tr.get("args") or {}
            for k, v in targs.items():
                if k in TRAP_VALUE_KEYS and str(v).strip().lower() not in PLACEHOLDERS and not isinstance(v, bool) \
                        and not trap_value_grounded(v, said_all, pool):
                    S.w("C4", cid, t, f"trap {tr['function']}({k}={v!r}) is not heard, looked up, or a near variant "
                                      "of anything in the case; only a hallucinating agent could trigger it (dead trap)")
                    stats["trap_ungrounded"] += 1
                if str(v).strip().lower() in PLACEHOLDERS:
                    S.w("C4", cid, t, f"trap {tr['function']}({k}={v!r}) uses a placeholder value; it can only "
                                      f"fire if the agent emits that literal (dead trap)")
                    stats["trap_placeholder"] += 1
            eq = [e for e in exp if call_matches(e, tr) and is_state_changing(e["function"])]
            if gate == "execute":
                if eq:
                    S.e("C4", cid, t, f"trap {tr['function']}{targs} matches an expected call on an execute turn "
                                      f"(a correct agent would be scored SEVERE)")
            else:
                if eq:
                    stats["trap_premature"] += 1   # trap == pending action -> encodes acting early
                    continue
            same_fn = [e for e in exp if e["function"] == tr["function"]]
            if not same_fn:
                stats["trap_cross_tool"] += 1
                continue
            diffs = min(sum(1 for k, v in targs.items() if k not in (e.get("args") or {}) or not value_match(e["args"][k], v))
                        for e in same_fn)
            if diffs == 0:
                continue
            stats["trap_corrupted_slot"] += 1
            if diffs > 1 and (fm in CORRECTION_FAMILIES | {"AMBIGUOUS_TARGET"} or has_corr):
                S.w("C4", cid, t, f"trap {tr['function']}{targs} differs from expected in {diffs} slots "
                                  "(a probe should corrupt exactly one)")


# --------------------------------------------------------------- C5 spillover


def check_spillover(c, S, stats):
    cid = c["id"]
    steps = c["steps"]
    for i in range(1, len(steps)):
        s, prev = steps[i], steps[i - 1]
        if s["gate_expected"] == "execute" and prev["gate_expected"] in GATED:
            stats["post_gate_execute"] += 1
            mn = s.get("must_not") or []
            if not mn:
                S.w("C5", cid, s["turn"], f"execute turn after a {prev['gate_expected']} turn has no must_not")
                stats["spillover_missing"] += 1
            elif "ASK_CONFIRMATION" not in mn:
                S.w("C5", cid, s["turn"], f"execute turn after a {prev['gate_expected']} turn: must_not lacks ASK_CONFIRMATION")


# --------------------------------------------------------------- C6 state carry / corrections


def check_state(c, S, stats):
    cid = c["id"]
    steps = c["steps"]
    fm = c.get("failure_mode", "")
    mock_vals = mock_strings(c.get("mock_data") or {})
    resolved = {}           # entity group -> last value in an expected call
    said = ""               # all user text so far
    for s in steps:
        t, u = s["turn"], s["user"]
        prior_said = said
        said = said + " \n " + u
        anaph = bool(ANAPHOR_RE.search(u))
        for call in s["expected_tool_calls"]:
            for k, v in (call.get("args") or {}).items():
                grp = ENTITY_ARGS.get(k)
                if not grp or isinstance(v, bool):
                    continue
                in_now = grounded_in(v, u, []) or record_spoken(v, u, c.get("mock_data"))
                in_before = grounded_in(v, said, [])  # includes this turn: 'Georgia' (T1) + 'the country' (T2)
                in_mock = grounded_in(v, "", mock_vals)
                if grp in resolved and not in_now and anaph and not value_match(resolved[grp], v):
                    S.w("C6", cid, t, f"anaphoric turn: {call['function']}.{k}={v!r} but the entity resolved "
                                      f"earlier for '{grp}' was {resolved[grp]!r}")
                    stats["state_mismatch"] += 1
                if not (in_now or in_before or in_mock or grp in resolved and value_match(resolved[grp], v)):
                    if k in CONTEXT_SCALARS:
                        continue
                    S.w("C6", cid, t, f"{call['function']}.{k}={v!r} is not grounded in any utterance, "
                                      f"mock_data, or earlier expected call")
                    stats["state_ungrounded"] += 1
        for call in s["expected_tool_calls"]:
            for k, v in (call.get("args") or {}).items():
                if ENTITY_ARGS.get(k) and not isinstance(v, bool):
                    resolved[ENTITY_ARGS[k]] = v
        # ---- self-corrections
        has_corr = bool(CORRECTION_RE.search(u))
        prev = steps[t - 2] if t > 1 else None
        completion = prev is not None and prev["gate_expected"] in ("confirm", "authorize") and s["gate_expected"] == "execute"
        if not has_corr or not s["expected_tool_calls"] or completion:
            continue
        slot_diffs = []
        for tr in s["trap_tool_calls"]:
            for e in s["expected_tool_calls"]:
                if e["function"] != tr["function"]:
                    continue
                for k, v in (tr.get("args") or {}).items():
                    ev = (e.get("args") or {}).get(k)
                    if ev is not None and not value_match(ev, v):
                        slot_diffs.append((e["function"], k, ev, v))
        if not slot_diffs:
            if fm in CORRECTION_FAMILIES and s["gate_expected"] != "clarify":
                S.w("C6", cid, t, "correction marker in a correction-family case but no trap carries a "
                                  "pre-correction value")
                stats["corr_no_trap"] += 1
            continue
        stats["corr_turns"] += 1
        for fn, k, post, pre in slot_diffs:
            pre_seen = grounded_in(pre, said, mock_vals)
            post_seen = grounded_in(post, said, mock_vals)
            if not post_seen:
                S.w("C6", cid, t, f"corrected value {fn}.{k}={post!r} never appears in the dialogue")
                stats["corr_bad"] += 1
            if not pre_seen:
                S.i("C6", cid, t, f"trap value {fn}.{k}={pre!r} is not a spoken pre-correction value "
                                  "(wrong-entity trap rather than self-correction trap)")
            elif value_in_text(pre, u) and value_in_text(post, u):
                pp = first_pos(pre, u)
                negated = pp >= 0 and re.search(r"\bnot (the )?$", u[max(0, pp - 9):pp].lower())
                if not negated and 0 <= first_pos(post, u) < pp:
                    S.w("C6", cid, t, f"{fn}.{k}: expected {post!r} is spoken BEFORE trap {pre!r}; "
                                      "correction direction may be inverted")
                    stats["corr_bad"] += 1


# --------------------------------------------------------------- C7 overlap

ENTITY_SKIP_KEYS = {"quantity", "bedrooms", "enabled", "nonstop", "scope", "type", "term", "status",
                    "deadline_passed", "note", "reason", "passenger", "title", "item_type"}
GENERIC = {"checking", "savings", "full", "partial", "weekly_mood", "unknown", "unspecified", "individual",
           "group", "true", "false", "none", "offer_amount", "floor_price", "me"}
ID_TOKEN_RE = re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)[A-Za-z][A-Za-z0-9_\-]{2,}\b|\b\d{1,3}[A-Z]\b|\b\d{4,}\b")
NAME_RE = re.compile(r"\bDr\.?\s+[A-Z][a-z]+(?:[- ][A-Z][a-z]+)?")


def _norm_entity(v):
    if isinstance(v, bool) or v is None:
        return None
    n = _num(v)
    if n is not None and not re.search(r"[a-zA-Z]", str(v)):
        return "#" + (str(int(n)) if n == int(n) else str(n))
    t = _text(v)
    if not t or t in GENERIC or len(t) < 2:
        return None
    return t


def _walk_mock(obj, key=None, out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk_mock(v, k, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_mock(v, key, out)
    elif key not in ENTITY_SKIP_KEYS:
        out.append((key, obj))
    return out


def case_entities(c) -> dict:
    """normalized entity -> set of provenance labels, for one case."""
    ents = defaultdict(set)
    for k, v in _walk_mock(c.get("mock_data") or {}):
        n = _norm_entity(v)
        if n:
            ents[n].add(f"mock.{k}")
    for s in c["steps"]:
        for key in ("expected_tool_calls", "trap_tool_calls"):
            for call in s[key]:
                for k, v in (call.get("args") or {}).items():
                    if k in ENTITY_SKIP_KEYS or k == "passcode":
                        continue
                    n = _norm_entity(v)
                    if n:
                        ents[n].add(f"{call['function']}.{k}")
        for tok in ID_TOKEN_RE.findall(s["user"]):
            n = _norm_entity(tok)
            if n:
                ents[n].add("utterance.id")
        for nm in NAME_RE.findall(s["user"]):
            n = _norm_entity(nm.replace("Dr.", "Dr"))
            if n:
                ents[n].add("utterance.name")
    if c.get("auth_passcode"):
        ents["#" + str(int(c["auth_passcode"]))].add("auth_passcode")
    return ents


def _utt_norm(u):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", u.lower())).strip()


def near_dups(a_list, b_list, thresh, same_set):
    """a_list/b_list: [(case, turn, norm_text)]. Returns [(a, b, ratio)]."""
    out = []
    for i, (ca, ta, ua) in enumerate(a_list):
        wa = set(ua.split())
        if len(wa) < 4:
            continue
        start = i + 1 if same_set else 0
        for cb, tb, ub in b_list[start:]:
            if same_set and ca == cb:
                continue
            wb = set(ub.split())
            if len(wb) < 4 or len(wa & wb) / len(wa | wb) < 0.5:
                continue
            sm = difflib.SequenceMatcher(None, ua, ub)
            if sm.quick_ratio() < thresh:
                continue
            r = sm.ratio()
            if r >= thresh:
                out.append(((ca, ta, ua), (cb, tb, ub), r))
    return out


def check_overlap(cases, ref_cases, S, stats, near_thresh=0.85):
    utts = [(c["id"], s["turn"], _utt_norm(s["user"])) for c in cases for s in c["steps"]]
    # within-set exact duplicate utterances
    by_text = defaultdict(list)
    for cid, t, u in utts:
        by_text[u].append((cid, t))
    for u, occ in by_text.items():
        ids = {x[0] for x in occ}
        if len(ids) > 1:
            S.w("C7", occ[0][0], occ[0][1], f"utterance '{u}' repeated verbatim in {len(ids)} cases: "
                                            f"{', '.join(sorted(f'{a}/T{b}' for a, b in occ))}")
            stats["within_exact_utt"] += 1
    for (a, b, r) in near_dups(utts, utts, near_thresh, same_set=True):
        if a[2] == b[2]:
            continue
        S.i("C7", a[0], a[1], f"near-duplicate utterance ({r:.2f}) of {b[0]}/T{b[1]}: '{a[2]}' ~ '{b[2]}'")
        stats["within_near_utt"] += 1
    # within-set identical cases
    sig = defaultdict(list)
    for c in cases:
        sig[json.dumps([(s["user"], s["expected_tool_calls"]) for s in c["steps"]], sort_keys=True)].append(c["id"])
    for ids in sig.values():
        if len(ids) > 1:
            S.e("C7", ids[0], None, f"identical dialogue+expected in cases {ids}")
    # within-set shared entities
    ent_cases = defaultdict(set)
    for c in cases:
        for n in case_entities(c):
            ent_cases[n].add(c["id"])
    for n, ids in sorted(ent_cases.items()):
        if len(ids) > 1:
            idlike = bool(re.search(r"\d", n)) and not n.startswith("#")
            (S.w if idlike else S.i)("C7", sorted(ids)[0], None,
                                     f"entity '{n}' shared by {len(ids)} cases: {', '.join(sorted(ids))}")
            stats["within_shared_idlike" if idlike else "within_shared_other"] += 1
    stats["distinct_entities"] = len(ent_cases)
    # speaker profiles within set
    trip = defaultdict(list)
    for c in cases:
        sp = c.get("_speaker")
        if sp:
            trip[(sp["voice"], sp["age_bracket"], sp["accent"])].append(c["id"])
        else:
            stats["no_speaker_profile"] += 1
    for k, ids in trip.items():
        if len(ids) > 1:
            S.w("C7", ids[0], None, f"speaker triple {k} reused by {ids}")
    if not ref_cases:
        return
    # ---- across sets
    ref_ent = defaultdict(set)
    for c in ref_cases:
        for n, prov in case_entities(c).items():
            ref_ent[n].add(c["id"])
    shared = 0
    for c in cases:
        for n, prov in case_entities(c).items():
            if n in ref_ent:
                S.e("C7", c["id"], None, f"entity '{n}' ({', '.join(sorted(prov))}) also appears in reference "
                                         f"case(s) {', '.join(sorted(ref_ent[n])[:4])}")
                shared += 1
    stats["cross_shared_entities"] = shared
    ref_utts = [(c["id"], s["turn"], _utt_norm(s["user"])) for c in ref_cases for s in c["steps"]]
    ref_text = defaultdict(list)
    for cid, t, u in ref_utts:
        ref_text[u].append(f"{cid}/T{t}")
    for cid, t, u in utts:
        if u in ref_text:
            S.e("C7", cid, t, f"utterance '{u}' is verbatim in reference {ref_text[u][:3]}")
            stats["cross_exact_utt"] += 1
    for (a, b, r) in near_dups(utts, ref_utts, near_thresh, same_set=False):
        if a[2] == b[2]:
            continue
        S.w("C7", a[0], a[1], f"near-duplicate ({r:.2f}) of reference {b[0]}/T{b[1]}: '{a[2]}' ~ '{b[2]}'")
        stats["cross_near_utt"] += 1
    ref_trip = {(c["_speaker"]["voice"], c["_speaker"]["age_bracket"], c["_speaker"]["accent"])
                for c in ref_cases if c.get("_speaker")}
    for c in cases:
        sp = c.get("_speaker")
        if sp and (sp["voice"], sp["age_bracket"], sp["accent"]) in ref_trip:
            S.w("C7", c["id"], None, f"speaker triple reused from the reference set")
            stats["cross_speaker_triple"] += 1


# --------------------------------------------------------------- C8 distribution


def distribution(cases) -> dict:
    d = {"cases": len(cases), "turns": 0, "domain_cases": Counter(), "domain_turns": Counter(),
         "gate": Counter(), "severity": Counter(), "gate_x_severity": Counter(), "failure_mode": Counter(),
         "turns_per_case": Counter(), "case_severity": Counter(), "gated_tools": Counter(),
         "probe_families": Counter()}
    for c in cases:
        d["domain_cases"][c["domain"]] += 1
        d["failure_mode"][c["failure_mode"]] += 1
        d["turns_per_case"][len(c["steps"])] += 1
        d["case_severity"][c["severity"]] += 1
        for pf in c.get("probe_families") or []:
            d["probe_families"][pf] += 1
        for s in c["steps"]:
            d["turns"] += 1
            d["domain_turns"][c["domain"]] += 1
            d["gate"][s["gate_expected"]] += 1
            d["severity"][s["severity"]] += 1
            d["gate_x_severity"][f"{s['gate_expected']}/{s['severity']}"] += 1
            if s["gate_expected"] in ("confirm", "authorize"):
                for a in state_actions(s):
                    d["gated_tools"][f"{s['gate_expected']}:{a['function']}"] += 1
    return {k: (dict(sorted(v.items(), key=lambda kv: str(kv[0]))) if isinstance(v, Counter) else v) for k, v in d.items()}


def pct(counter, total):
    return {k: round(100.0 * v / total, 1) for k, v in counter.items()} if total else {}


def compare(dist, ref, keys=("gate", "severity")):
    rows = []
    for key in keys:
        cats = sorted(set(dist[key]) | set(ref[key]))
        for cat in cats:
            a = 100.0 * dist[key].get(cat, 0) / max(1, dist["turns"])
            b = 100.0 * ref[key].get(cat, 0) / max(1, ref["turns"])
            rows.append({"dim": key, "cat": cat, "set_pct": round(a, 1), "ref_pct": round(b, 1),
                         "delta_pp": round(a - b, 1), "within_5pp": abs(a - b) <= 5.0})
    return rows


def print_dist(d, title):
    print(f"\n== {title}: {d['cases']} cases, {d['turns']} turns")
    T = max(1, d["turns"])
    for key in ("domain_cases", "domain_turns", "gate", "severity", "gate_x_severity", "failure_mode",
                "turns_per_case", "case_severity", "gated_tools", "probe_families"):
        if not d[key]:
            continue
        denom = d["cases"] if key in ("domain_cases", "failure_mode", "turns_per_case", "case_severity", "probe_families") else T
        items = ", ".join(f"{k}={v} ({100.0 * v / max(1, denom):.1f}%)" for k, v in d[key].items())
        print(f"  {key:16s} {items}")


# --------------------------------------------------------------- main


def run(dirs, against, near_thresh=0.85):
    S = Sink()
    stats = defaultdict(int)
    stats["authorize_tools"] = Counter()
    stats["critical_on_confirm"] = Counter()
    cases = [c for d in dirs for c in load_dir(d)]
    ref_cases = [c for d in against for c in load_dir(d)]
    ids = Counter(c.get("id") for c in cases)
    for k, v in ids.items():
        if v > 1:
            S.e("C1", k, None, f"case id appears {v} times in the set")
    good = []
    for c in cases:
        if check_schema(c, S):
            good.append(c)
    for c in good:
        check_tools(c, S)
        check_gates(c, S, stats)
        check_traps(c, S, stats)
        check_spillover(c, S, stats)
        check_state(c, S, stats)
    check_overlap(good, [c for c in ref_cases if "steps" in c and c.get("steps")], S, stats, near_thresh)
    dist = distribution(good)
    ref_dist = distribution(ref_cases) if ref_cases else None
    stats["tier_only_baseline"] = tier_only_baseline(good)
    stats["passcodes"] = passcode_reuse(good)
    if ref_cases:
        stats["reference_tier_only_baseline"] = tier_only_baseline(ref_cases)
        stats["reference_passcodes"] = passcode_reuse(ref_cases)
    wav_missing = sum(1 for c in good for s in c["steps"] if not (c["_folder"] / f"input_turn{s['turn']}.wav").exists())
    stats["turns_without_audio"] = wav_missing
    stats["passcode_flag_in_descriptions"] = sum(
        1 for n in TOOL_SCHEMAS_REALTIME if PASSCODE_FLAG_RE.search(TOOL_SCHEMAS_REALTIME[n]["description"]))
    return S, stats, dist, ref_dist


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="*", type=Path, help="case directories (default: dev set)")
    ap.add_argument("--against", action="append", type=Path, default=[], help="reference directory (repeatable)")
    ap.add_argument("--json", action="store_true", help="emit one JSON object on stdout")
    ap.add_argument("--near", type=float, default=0.85, help="near-duplicate similarity threshold")
    ap.add_argument("--show-info", action="store_true", help="also print INFO issues")
    a = ap.parse_args(argv)
    dirs = [d.resolve() for d in (a.dirs or DEV_DIRS)]
    against = [d.resolve() for d in a.against]
    S, stats, dist, ref_dist = run(dirs, against, a.near)
    errs = [x for x in S.issues if x["level"] == "ERROR"]
    warns = [x for x in S.issues if x["level"] == "WARN"]
    infos = [x for x in S.issues if x["level"] == "INFO"]
    stats_out = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in stats.items()}
    if "authorize_tools" in stats_out:
        stats_out["authorize_tools"] = {"+".join(k) if k else "(none)": v for k, v in stats_out["authorize_tools"].items()}
    comp = compare(dist, ref_dist) if ref_dist else None
    if a.json:
        print(json.dumps({"dirs": [str(d) for d in dirs], "against": [str(d) for d in against],
                          "counts": {"errors": len(errs), "warnings": len(warns), "info": len(infos)},
                          "by_check": {lvl: dict(Counter(x["check"] for x in S.issues if x["level"] == lvl))
                                       for lvl in ("ERROR", "WARN", "INFO")},
                          "issues": S.issues, "stats": stats_out, "distribution": dist,
                          "reference_distribution": ref_dist, "comparison": comp}, indent=1, default=str))
        return 1 if errs else 0

    def show(lst, label):
        print(f"\n######## {label} ({len(lst)})")
        by = defaultdict(list)
        for x in lst:
            by[x["check"]].append(x)
        for chk in sorted(by):
            print(f"-- {chk} ({len(by[chk])})")
            for x in by[chk]:
                where = f"{x['case']}" + (f"/T{x['turn']}" if x["turn"] else "")
                print(f"   {where:28s} {x['msg']}")

    print("validate_cases:", ", ".join(str(d) for d in dirs))
    show(errs, "HARD ERRORS")
    show(warns, "WARNINGS")
    if a.show_info:
        show(infos, "INFO")
    print(f"\nINFO issues: {len(infos)} ({dict(Counter(x['check'] for x in infos))}); use --show-info to list")
    print("\nstats:", json.dumps(stats_out, default=str))
    print_dist(dist, "SET")
    if ref_dist:
        print_dist(ref_dist, "REFERENCE")
        print("\n== gate/severity mix vs reference (+/-5pp)")
        for r in comp:
            print(f"  {r['dim']:9s} {r['cat']:10s} set {r['set_pct']:5.1f}%  ref {r['ref_pct']:5.1f}%  "
                  f"delta {r['delta_pp']:+5.1f}pp  {'ok' if r['within_5pp'] else 'OUT OF BAND'}")
    print(f"\nRESULT: {len(errs)} hard error(s), {len(warns)} warning(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
