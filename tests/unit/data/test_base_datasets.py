"""Tests for base dataset download and splitting."""

from datasets import Dataset

from src.data.base_datasets import split_dataset, validate_splits


def _make_fake_dataset(n: int) -> Dataset:
    """Create a small fake dataset for testing."""
    return Dataset.from_dict({
        "instruction": [f"instruction_{i}" for i in range(n)],
        "input": [f"input_{i}" if i % 2 == 0 else "" for i in range(n)],
        "output": [f"output_{i}" for i in range(n)],
    })


def test_split_dataset_sizes():
    ds = _make_fake_dataset(100)
    train, eval_ = split_dataset(ds, train_fraction=0.85, seed=42)
    assert len(train) == 85
    assert len(eval_) == 15


def test_split_dataset_no_overlap():
    ds = _make_fake_dataset(100)
    train, eval_ = split_dataset(ds, train_fraction=0.85, seed=42)
    train_instructions = set(train["instruction"])
    eval_instructions = set(eval_["instruction"])
    assert train_instructions.isdisjoint(eval_instructions)


def test_split_dataset_deterministic():
    ds = _make_fake_dataset(100)
    train1, _ = split_dataset(ds, train_fraction=0.85, seed=42)
    train2, _ = split_dataset(ds, train_fraction=0.85, seed=42)
    assert train1["instruction"] == train2["instruction"]


def test_validate_splits_missing(tmp_path):
    from src.config.paths import PathsConfig
    paths = PathsConfig(project_root=tmp_path)
    report = validate_splits(paths)
    assert report["all_exist"] is False
    assert len(report["missing"]) == 4


def test_validate_splits_present(tmp_path):
    from src.config.paths import PathsConfig
    paths = PathsConfig(project_root=tmp_path)
    for name in ["alpaca_train", "alpaca_eval", "dolly_train", "dolly_eval"]:
        split_dir = paths.splits_dir / name
        split_dir.mkdir(parents=True)
        ds = _make_fake_dataset(10)
        ds.save_to_disk(str(split_dir))

    report = validate_splits(paths)
    assert report["all_exist"] is True
    assert len(report["missing"]) == 0
    assert report["counts"]["alpaca_train"] == 10
