import json
import pytest
from rp.cases import load_split, verify_lock, LockError, STUDY_DIR


def test_test_split_loads_119_cases():
    cases = load_split("test")
    assert len(cases) == 119
    assert sum(len(c["steps"]) for c in cases) == 373
    assert all(c["_folder"].is_dir() for c in cases)


def test_lock_detects_tampering(tmp_path):
    case = tmp_path / "x_case"
    case.mkdir()
    (case / "metadata.json").write_text('{"id": "x"}')
    import hashlib
    h = hashlib.sha256((case / "metadata.json").read_bytes()).hexdigest()
    (tmp_path / "LOCK.json").write_text(json.dumps({"files": {"x_case": h}}))
    verify_lock(tmp_path)  # passes
    (case / "metadata.json").write_text('{"id": "y"}')
    with pytest.raises(LockError):
        verify_lock(tmp_path)


def test_unknown_split_raises():
    with pytest.raises(KeyError):
        load_split("nope")


def test_lock_detects_added_case(tmp_path):
    """Lock must reject folder not listed in lock["files"]."""
    import hashlib
    # Create first case and lock
    case1 = tmp_path / "x_case"
    case1.mkdir()
    (case1 / "metadata.json").write_text('{"id": "x"}')
    h1 = hashlib.sha256((case1 / "metadata.json").read_bytes()).hexdigest()
    (tmp_path / "LOCK.json").write_text(json.dumps({"files": {"x_case": h1}}))
    verify_lock(tmp_path)  # passes

    # Add a second case not in the lock
    case2 = tmp_path / "y_case"
    case2.mkdir()
    (case2 / "metadata.json").write_text('{"id": "y"}')

    with pytest.raises(LockError):
        verify_lock(tmp_path)


def test_lock_detects_removed_case(tmp_path):
    """Lock must reject when a case folder is missing."""
    import hashlib
    # Create lock that references two cases
    case1 = tmp_path / "x_case"
    case1.mkdir()
    (case1 / "metadata.json").write_text('{"id": "x"}')
    h1 = hashlib.sha256((case1 / "metadata.json").read_bytes()).hexdigest()

    (tmp_path / "LOCK.json").write_text(json.dumps({
        "files": {"x_case": h1, "y_case": "dummy_hash"}
    }))

    # y_case folder doesn't exist
    with pytest.raises(LockError):
        verify_lock(tmp_path)


def test_missing_lock_file_raises_lockerror(tmp_path):
    """Missing LOCK.json must raise LockError, not FileNotFoundError."""
    with pytest.raises(LockError):
        verify_lock(tmp_path)
