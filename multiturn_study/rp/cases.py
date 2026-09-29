"""Load the dev_v2 and locked test splits. Never loads the original dev set."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent.parent
SPLITS = {
    "dev_v2": [STUDY_DIR / "data" / "dev_v2"],
    "test": [STUDY_DIR / "data" / "test_cases_v3"],
}
LOCKED = {"test"}


class LockError(RuntimeError):
    pass


def verify_lock(folder: Path) -> None:
    try:
        lock = json.loads((folder / "LOCK.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise LockError(f"cannot read LOCK.json: {e}")

    # Find all case folders with metadata.json
    case_folders = {
        d.name for d in folder.iterdir() if d.is_dir()
        and (d / "metadata.json").exists()
    }

    # Check for added or removed cases
    locked_folders = set(lock["files"])
    if case_folders != locked_folders:
        added = case_folders - locked_folders
        removed = locked_folders - case_folders
        msg = "case set mismatch:"
        if added:
            msg += f" added {sorted(added)}"
        if removed:
            msg += f" removed {sorted(removed)}"
        raise LockError(msg)

    # Verify hashes
    for case_dir, digest in lock["files"].items():
        try:
            got = hashlib.sha256((folder / case_dir / "metadata.json").read_bytes()).hexdigest()
        except FileNotFoundError as e:
            raise LockError(f"{case_dir}: metadata.json missing: {e}")
        if got != digest:
            raise LockError(f"{case_dir}: metadata.json changed since lock")


def load_split(name: str, only_id: str | None = None) -> list[dict]:
    dirs = SPLITS[name]
    cases = []
    for base in dirs:
        if name in LOCKED:
            verify_lock(base)
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            meta = d / "metadata.json"
            if not meta.exists():
                continue
            case = json.loads(meta.read_text())
            case["_folder"] = d
            if only_id and case["id"] != only_id:
                continue
            cases.append(case)
    return cases
