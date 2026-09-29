"""Reading and appending runner output (`results/rp/.../pass{K}.jsonl`).

The runner only ever APPENDS. When a case is re-run (because an earlier
attempt had an agent error or a judge error), its new rows are appended after
the old ones and the old rows stay in the file as an audit trail.

Rule for every reader: the LAST row per (case_id, turn) wins. `load_rows`
applies that rule; analysis code must use it (or apply the same rule) and never
read the raw lines directly. A re-run always writes a row for every turn of the
case, so the latest rows of a case always come from one run.

A torn final line (the process was killed mid-write) is skipped with a warning;
`append_rows` first terminates such a line so new rows start on a fresh line.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def iter_raw(path: Path):
    """Every parseable row in file order (including superseded ones)."""
    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            print(f"warning: {path}:{i + 1}: unparseable row skipped", file=sys.stderr)


def load_rows(path: Path) -> list[dict]:
    """Latest row per (case_id, turn), in the order each key first appeared."""
    latest: dict[tuple, dict] = {}
    for row in iter_raw(path):
        latest[(row.get("case_id"), row.get("turn"))] = row
    return list(latest.values())


def append_rows(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(prefix + "".join(json.dumps(r, default=str) + "\n" for r in rows))
        f.flush()
