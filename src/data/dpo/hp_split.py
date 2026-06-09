"""Stratified train/validation split for DPO hyperparameter search.

Partitions a list of DPO preference records into two disjoint index sets:

- ``hp_select``: a held-out set used only for post-training HP ranking.
- ``val_train``: the remainder, used as the training-time ``eval_dataset``.

Stratification is over ``(level_gap, is_calibration)`` with proportional
allocation, largest-remainder rounding to hit ``target_size`` exactly,
and clamping so no bucket is over-sampled beyond its source count.
"""

import random


def build_hp_split(
    records: list[dict],
    target_size: int,
    seed: int,
) -> tuple[list[int], list[int], dict[tuple[int, bool], int]]:
    """Split records into hp_select / val_train by stratified sampling.

    Args:
        records: Preference records with ``level_gap`` and
            ``is_calibration`` fields.
        target_size: Desired size of the hp_select partition. If larger
            than ``len(records)``, capped at the total record count.
        seed: RNG seed for reproducibility.

    Returns:
        A tuple ``(hp_select_indices, val_train_indices, bucket_counts)``:

        - ``hp_select_indices``: sorted list of indices into ``records``.
        - ``val_train_indices``: sorted list of remaining indices.
        - ``bucket_counts``: per-``(level_gap, is_calibration)`` count of
          records placed into hp_select.
    """
    rng = random.Random(seed)
    target_size = min(target_size, len(records))

    buckets: dict[tuple[int, bool], list[int]] = {}
    for idx, rec in enumerate(records):
        key = (int(rec["level_gap"]), bool(rec.get("is_calibration", False)))
        buckets.setdefault(key, []).append(idx)

    total = len(records)
    exact = {k: len(v) * target_size / total for k, v in buckets.items()}
    allocation = {k: int(v) for k, v in exact.items()}

    # Clamp allocation to bucket size (can't sample more than exist).
    for k in allocation:
        allocation[k] = min(allocation[k], len(buckets[k]))

    # Largest-remainder rounding to make sum == target_size.
    remainder = target_size - sum(allocation.values())
    if remainder > 0:
        # Candidates with room to grow (not yet clamped to bucket size).
        fractional_order = sorted(
            ((exact[k] - int(exact[k]), k) for k in allocation),
            reverse=True,
        )
        for _, k in fractional_order:
            if remainder == 0:
                break
            if allocation[k] < len(buckets[k]):
                allocation[k] += 1
                remainder -= 1

    hp_select_indices: list[int] = []
    for key, group_indices in buckets.items():
        n = allocation[key]
        if n > 0:
            hp_select_indices.extend(rng.sample(group_indices, n))

    hp_select_set = set(hp_select_indices)
    val_train_indices = [i for i in range(len(records)) if i not in hp_select_set]

    return sorted(hp_select_indices), sorted(val_train_indices), allocation
