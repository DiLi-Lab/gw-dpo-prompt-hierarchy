"""Tests for the shared resume / output-dir-discovery helpers."""

import json
from pathlib import Path

import pytest

from src.evaluation.external.resume import (
    ResumeArgsMismatch,
    is_complete,
    latest_run_dir,
    resolve_output_dir,
    save_run_args,
    validate_run_args,
)


def _make_run(parent: Path, name: str, *, complete: bool) -> Path:
    d = parent / name
    d.mkdir(parents=True)
    if complete:
        (d / "metrics.json").write_text("{}")
    return d


# --- latest_run_dir / is_complete ---------------------------------------


def test_latest_run_dir_returns_none_for_missing_dir(tmp_path: Path) -> None:
    assert latest_run_dir(tmp_path / "does_not_exist") is None


def test_latest_run_dir_returns_none_for_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert latest_run_dir(tmp_path / "empty") is None


def test_latest_run_dir_ignores_non_run_subdirs(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "not_a_run").mkdir()
    (base / "results").mkdir()
    assert latest_run_dir(base) is None


def test_latest_run_dir_picks_newest_by_lexicographic_sort(tmp_path: Path) -> None:
    """run_<UTC-ts> names sort lexicographically == chronologically."""
    base = tmp_path / "base"
    _make_run(base, "run_20260101_010000", complete=True)
    newer = _make_run(base, "run_20260102_010000", complete=False)
    _make_run(base, "run_20260101_120000", complete=True)
    assert latest_run_dir(base) == newer


def test_is_complete_detects_metrics_json(tmp_path: Path) -> None:
    partial = _make_run(tmp_path, "run_p", complete=False)
    complete = _make_run(tmp_path, "run_c", complete=True)
    assert is_complete(complete) is True
    assert is_complete(partial) is False


# --- resolve_output_dir -------------------------------------------------


def test_resolve_explicit_output_dir_short_circuits(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    out, mode = resolve_output_dir(
        output_dir_arg=str(explicit),
        resume=True,
        base_dir=tmp_path / "base",
        timestamp="20260505_000000",
    )
    assert out == explicit
    assert mode == "explicit"


def test_resolve_no_resume_creates_fresh_ts_dir(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _make_run(base, "run_20260101_010000", complete=True)
    out, mode = resolve_output_dir(
        output_dir_arg=None,
        resume=False,
        base_dir=base,
        timestamp="20260505_120000",
    )
    assert out == base / "run_20260505_120000"
    assert mode == "fresh"


def test_resolve_resume_with_no_prior_run_creates_fresh(tmp_path: Path) -> None:
    base = tmp_path / "base"
    out, mode = resolve_output_dir(
        output_dir_arg=None,
        resume=True,
        base_dir=base,
        timestamp="20260505_120000",
    )
    assert out == base / "run_20260505_120000"
    assert mode == "fresh"


def test_resolve_resume_skips_when_latest_run_is_complete(tmp_path: Path) -> None:
    base = tmp_path / "base"
    complete = _make_run(base, "run_20260101_010000", complete=True)
    out, mode = resolve_output_dir(
        output_dir_arg=None,
        resume=True,
        base_dir=base,
        timestamp="20260505_120000",
    )
    assert out == complete
    assert mode == "resume_complete"


def test_resolve_resume_reuses_newest_partial_when_no_complete_after(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    _make_run(base, "run_20260101_010000", complete=False)
    newer_partial = _make_run(base, "run_20260102_010000", complete=False)
    out, mode = resolve_output_dir(
        output_dir_arg=None,
        resume=True,
        base_dir=base,
        timestamp="20260505_120000",
    )
    assert out == newer_partial
    assert mode == "resume_partial"


def test_resolve_resume_complete_takes_priority_when_newest_is_complete(
    tmp_path: Path,
) -> None:
    """If the newest run is complete, skip — don't dig into older partials."""
    base = tmp_path / "base"
    _make_run(base, "run_20260101_010000", complete=False)
    newer_complete = _make_run(base, "run_20260103_010000", complete=True)
    out, mode = resolve_output_dir(
        output_dir_arg=None,
        resume=True,
        base_dir=base,
        timestamp="20260505_120000",
    )
    assert out == newer_complete
    assert mode == "resume_complete"


# --- run_args persistence + mismatch detection --------------------------


def test_save_run_args_writes_run_args_json(tmp_path: Path) -> None:
    save_run_args(tmp_path, {"model": "gw_dpo", "format": "delimited", "limit": None})
    on_disk = json.loads((tmp_path / "run_args.json").read_text())
    assert on_disk == {"model": "gw_dpo", "format": "delimited", "limit": None}


def test_validate_run_args_no_op_when_file_missing(tmp_path: Path) -> None:
    validate_run_args(tmp_path, {"model": "gw_dpo"})  # no exception


def test_validate_run_args_passes_on_match(tmp_path: Path) -> None:
    save_run_args(tmp_path, {"model": "gw_dpo", "limit": None})
    validate_run_args(tmp_path, {"model": "gw_dpo", "limit": None})


def test_validate_run_args_raises_on_mismatch(tmp_path: Path) -> None:
    save_run_args(tmp_path, {"model": "gw_dpo", "limit": None})
    with pytest.raises(ResumeArgsMismatch) as exc:
        validate_run_args(tmp_path, {"model": "gw_dpo", "limit": 50})
    assert "limit" in str(exc.value)


def test_validate_run_args_diffs_multiple_keys(tmp_path: Path) -> None:
    save_run_args(tmp_path, {"model": "gw_dpo", "format": "delimited"})
    with pytest.raises(ResumeArgsMismatch) as exc:
        validate_run_args(tmp_path, {"model": "sft_only", "format": "chat_template"})
    msg = str(exc.value)
    assert "model" in msg and "format" in msg
