"""Base dataset download, splitting, and validation.

Downloads Alpaca Cleaned and Dolly 15K from HuggingFace, applies an
85/15 train/eval split with a fixed seed, and saves splits to disk.
The split happens before any construction to prevent data contamination.
"""

import logging

from datasets import Dataset, load_dataset, load_from_disk

from src.config.paths import PathsConfig

logger = logging.getLogger(__name__)

ALPACA_DATASET: str = "yahma/alpaca-cleaned"
DOLLY_DATASET: str = "databricks/databricks-dolly-15k"
DEFAULT_TRAIN_FRACTION: float = 0.85
DEFAULT_SEED: int = 42


def split_dataset(
    dataset: Dataset,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into train and eval sets.

    Args:
        dataset: The dataset to split.
        train_fraction: Fraction of data for training (default 0.85).
        seed: Random seed for shuffling (default 42).

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    shuffled = dataset.shuffle(seed=seed)
    split_idx = int(len(shuffled) * train_fraction)
    train = shuffled.select(range(split_idx))
    eval_ = shuffled.select(range(split_idx, len(shuffled)))
    return train, eval_


def download_and_split(
    paths: PathsConfig, seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    """Download base datasets, split, and save to disk.

    Args:
        paths: Path configuration specifying where to save splits.
        seed: Random seed for reproducible shuffling.

    Returns:
        Dict mapping split names to their example counts.
    """
    paths.splits_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s ...", ALPACA_DATASET)
    alpaca = load_dataset(ALPACA_DATASET, split="train")
    alpaca_train, alpaca_eval = split_dataset(alpaca, seed=seed)

    logger.info("Downloading %s ...", DOLLY_DATASET)
    dolly = load_dataset(DOLLY_DATASET, split="train")
    dolly_train, dolly_eval = split_dataset(dolly, seed=seed)

    splits = {
        "alpaca_train": alpaca_train,
        "alpaca_eval": alpaca_eval,
        "dolly_train": dolly_train,
        "dolly_eval": dolly_eval,
    }

    split_paths = {
        "alpaca_train": paths.alpaca_train,
        "alpaca_eval": paths.alpaca_eval,
        "dolly_train": paths.dolly_train,
        "dolly_eval": paths.dolly_eval,
    }

    counts: dict[str, int] = {}
    for name, ds in splits.items():
        save_path = split_paths[name]
        logger.info("Saving %s (%d examples) to %s", name, len(ds), save_path)
        ds.save_to_disk(str(save_path))
        counts[name] = len(ds)

    return counts


def validate_splits(paths: PathsConfig) -> dict:
    """Check whether all expected splits exist and report their sizes.

    Args:
        paths: Path configuration specifying split locations.

    Returns:
        Dict with keys: all_exist (bool), missing (list), counts (dict).
    """
    split_paths = {
        "alpaca_train": paths.alpaca_train,
        "alpaca_eval": paths.alpaca_eval,
        "dolly_train": paths.dolly_train,
        "dolly_eval": paths.dolly_eval,
    }

    missing: list[str] = []
    counts: dict[str, int] = {}

    for name, path in split_paths.items():
        if not path.exists():
            missing.append(name)
        else:
            ds = load_from_disk(str(path))
            counts[name] = len(ds)

    return {
        "all_exist": len(missing) == 0,
        "missing": missing,
        "counts": counts,
    }
