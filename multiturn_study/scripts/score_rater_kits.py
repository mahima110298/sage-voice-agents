#!/usr/bin/env python3
"""Score returned rater kits (see build_rater_kits.py).

Put the returned workbooks in rater_kits/returned/ with their original names,
then:  python3 score_rater_kits.py

Reports, for whichever kits are present:
  severity  Cohen's kappa (unweighted and linear-weighted) between raters and
            between each rater and the author; the must-execute-immediately
            question separately; every disagreement listed for adjudication
  judge     inter-rater kappa and a 3x3 PASS/FAIL/SEVERE confusion matrix;
            agreement and kappa of each rater, and of their adjudicated
            consensus, with judge v1 and judge v2
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook

KITS = Path(__file__).resolve().parent.parent / "rater_kits"
RET = KITS / "returned"
TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
VERDICTS = ["PASS", "FAIL", "SEVERE"]


def kappa(a, b, labels, weighted=False):
    pairs = [(x, y) for x, y in zip(a, b) if x in labels and y in labels]
    n = len(pairs)
    if n == 0:
        return float("nan"), 0
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    w = [[(abs(i - j) / (k - 1) if weighted else float(i != j)) for j in range(k)] for i in range(k)]
    obs = Counter(pairs)
    ma, mb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    do = sum(w[idx[x]][idx[y]] * c for (x, y), c in obs.items()) / n
    de = sum(w[idx[x]][idx[y]] * ma[x] * mb[y] for x in labels for y in labels) / (n * n)
    return (1 - do / de if de else 1.0), n


def matrix(a, b, labels, name_a, name_b):
    c = Counter(zip(a, b))
    out = [f"    {name_a} \\ {name_b}  " + " ".join(f"{l:>8}" for l in labels)]
    for x in labels:
        out.append(f"    {x:<{len(name_a) + len(name_b) + 4}}" + " ".join(f"{c.get((x, y), 0):>8}" for y in labels))
    return "\n".join(out)


def read_sheet(path, sheet, cols):
    ws = load_workbook(path, data_only=True)[sheet]
    head = [c.value for c in ws[1]]
    ix = [head.index(c) for c in cols]
    return {r[0]: [str(r[i] or "").strip() for i in ix] for r in ws.iter_rows(min_row=2, values_only=True) if r[0]}


def score_severity():
    files = [RET / f"severity_labeling_rater_{x}.xlsx" for x in "AB"]
    if not all(f.exists() for f in files):
        print("severity: waiting for both returned sheets")
        return
    cols = ["tier (LOW/MEDIUM/HIGH/CRITICAL)", "must execute immediately? (yes/no)"]
    A, B = (read_sheet(f, "Tools", cols) for f in files)
    key = {int(r["item"]): r for r in csv.DictReader(open(KITS / "_key_severity.csv"))}
    items = sorted(key)
    a = [A[i][0].upper() for i in items]
    b = [B[i][0].upper() for i in items]
    au = [key[i]["author_tier"] for i in items]
    print("=== SEVERITY ===")
    for name, x, y in (("A vs B", a, b), ("A vs author", a, au), ("B vs author", b, au)):
        k, n = kappa(x, y, TIERS)
        kw, _ = kappa(x, y, TIERS, weighted=True)
        agree = sum(p == q for p, q in zip(x, y)) / len(items)
        print(f"  {name:<12} n={n}  exact agreement {agree:.1%}  kappa {k:.3f}  weighted kappa {kw:.3f}")
    print(matrix(a, b, TIERS, "A", "B"))
    ia = [A[i][1].lower() for i in items]
    ib = [B[i][1].lower() for i in items]
    k, n = kappa(ia, ib, ["yes", "no"])
    print(f"  must-execute-immediately A vs B: kappa {k:.3f} (n={n})")
    print("  disagreements for adjudication:")
    for i, x, y, z in zip(items, a, b, au):
        if len({x, y, z}) > 1:
            print(f"    item {i:>2} {key[i]['tool']:<32} A={x:<9} B={y:<9} author={z}")


def score_judge():
    files = [RET / f"judge_validation_rater_{x}.xlsx" for x in "AB"]
    if not all(f.exists() for f in files):
        print("judge: waiting for both returned sheets")
        return
    cols = ["your verdict", "gate produced"]
    A, B = (read_sheet(f, "Turns", cols) for f in files)
    key = {int(r["item"]): r for r in csv.DictReader(open(KITS / "_key_judge.csv"))}
    items = sorted(key)
    a = [A[i][0].upper() for i in items]
    b = [B[i][0].upper() for i in items]
    v1 = [key[i]["verdict_v1"] for i in items]
    v2 = [key[i]["verdict_v2"] for i in items]
    print("=== JUDGE VALIDATION ===")
    k, n = kappa(a, b, VERDICTS)
    print(f"  inter-rater  n={n}  agreement {sum(x == y for x, y in zip(a, b)) / len(items):.1%}  kappa {k:.3f}")
    print(matrix(a, b, VERDICTS, "A", "B"))
    # Consensus: where raters agree. Disagreements are listed for adjudication and
    # excluded here until an adjudicated label is entered.
    cons = [x if x == y else "" for x, y in zip(a, b)]
    for jname, j in (("judge v1", v1), ("judge v2", v2)):
        for rname, r in (("rater A", a), ("rater B", b), ("consensus", cons)):
            k, n = kappa(r, j, VERDICTS)
            agree = sum(x == y for x, y in zip(r, j) if x) / max(1, sum(1 for x in r if x))
            print(f"  {rname:<10} vs {jname}: n={n}  agreement {agree:.1%}  kappa {k:.3f}")
        print(matrix([c for c in cons if c], [y for c, y in zip(cons, j) if c], VERDICTS, "human", jname))
    print("  rater disagreements for adjudication:", [i for i, x, y in zip(items, a, b) if x != y])


if __name__ == "__main__":
    score_severity()
    score_judge()
