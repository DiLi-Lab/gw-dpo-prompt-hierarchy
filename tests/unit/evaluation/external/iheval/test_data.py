"""IHEval data loader: walks benchmark/ tree and emits per-task records."""

import json
from pathlib import Path

from src.evaluation.external.iheval.data import (
    IHEvalRecord,
    iter_iheval_records,
)


def _write_min_tree(root: Path) -> Path:
    """Write a minimal benchmark/ tree with two settings of one task."""
    base = root / "benchmark" / "rule-following" / "single-turn"
    aligned = base / "aligned" / "default"
    conflict = base / "conflict" / "default"
    aligned.mkdir(parents=True)
    conflict.mkdir(parents=True)
    (aligned / "input_data.json").write_text(json.dumps([
        {"id": 1, "system": "S1", "instruction": "I1", "answer": {"x": 1}},
    ]))
    (conflict / "input_data.json").write_text(json.dumps([
        {"id": 2, "system": "S2", "instruction": "I2", "answer": {"x": 2}},
        {"id": 3, "system": "S3", "instruction": "I3", "answer": {"x": 3}},
    ]))
    return root / "benchmark"


def test_iter_records_emits_one_per_input(tmp_path: Path) -> None:
    bench_root = _write_min_tree(tmp_path)
    records = list(iter_iheval_records(
        bench_root,
        tasks=("single-turn",),
        settings=("aligned", "conflict"),
    ))
    assert len(records) == 3


def test_records_carry_task_setting_and_payload(tmp_path: Path) -> None:
    bench_root = _write_min_tree(tmp_path)
    records = list(iter_iheval_records(
        bench_root,
        tasks=("single-turn",),
        settings=("aligned",),
    ))
    r = records[0]
    assert isinstance(r, IHEvalRecord)
    assert r.task == "single-turn"
    assert r.setting == "aligned"
    assert r.sub == "default"
    assert r.id == 1
    assert r.system == "S1"
    assert r.instruction == "I1"
    assert r.answer == {"x": 1}


def test_record_uid_is_unique_across_subs(tmp_path: Path) -> None:
    bench_root = _write_min_tree(tmp_path)
    records = list(iter_iheval_records(
        bench_root,
        tasks=("single-turn",),
        settings=("aligned", "conflict"),
    ))
    uids = [r.uid for r in records]
    assert len(uids) == len(set(uids))


def test_unknown_task_produces_zero_records(tmp_path: Path) -> None:
    bench_root = _write_min_tree(tmp_path)
    records = list(iter_iheval_records(
        bench_root,
        tasks=("translation",),
        settings=("aligned",),
    ))
    assert records == []
