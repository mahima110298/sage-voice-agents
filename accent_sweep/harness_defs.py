"""Load TOOL_SCHEMAS, DOMAIN_TOOLS, and SUT_SYSTEM directly out of
voice_guard_eval.ipynb instead of retyping them here — these are the exact
tool schemas and Risk Router policy the existing text-based pipeline already
uses, and copy-pasting ~20 tool definitions by hand is exactly the kind of
place a transcription slip would silently change what's being evaluated.

Only the three cells that define these objects are executed (not the whole
notebook — no network calls, no Colab/Drive mounting).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "voice_guard_eval.ipynb"


def _load_harness_namespace() -> dict:
    nb = json.loads(NOTEBOOK_PATH.read_text())
    ns: dict = {}
    found = {"TOOL_SCHEMAS": False, "DOMAIN_TOOLS": False, "SUT_SYSTEM": False}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "TOOL_SCHEMAS = {" in src or "SUT_SYSTEM = " in src:
            exec(compile(src, "<voice_guard_eval.ipynb cell>", "exec"), ns)
            found["TOOL_SCHEMAS"] = found["TOOL_SCHEMAS"] or "TOOL_SCHEMAS" in ns
            found["DOMAIN_TOOLS"] = found["DOMAIN_TOOLS"] or "DOMAIN_TOOLS" in ns
            found["SUT_SYSTEM"] = found["SUT_SYSTEM"] or "SUT_SYSTEM" in ns
    missing = [k for k, v in found.items() if not v]
    if missing:
        raise RuntimeError(
            f"Could not find {missing} in {NOTEBOOK_PATH} — the notebook's cell "
            "structure may have changed; update the markers in _load_harness_namespace."
        )
    return ns


_ns = _load_harness_namespace()
TOOL_SCHEMAS: dict = _ns["TOOL_SCHEMAS"]
DOMAIN_TOOLS: dict = _ns["DOMAIN_TOOLS"]
SUT_SYSTEM: str = _ns["SUT_SYSTEM"]

if __name__ == "__main__":
    print(f"Loaded {len(TOOL_SCHEMAS)} tool schemas across {len(DOMAIN_TOOLS)} domains")
    print(f"SUT_SYSTEM: {len(SUT_SYSTEM)} chars")
