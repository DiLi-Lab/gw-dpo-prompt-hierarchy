"""Weighted Hierarchy Score and gap-bucketed PPA aggregation.

WHS = Σ (j-i) · PPA_{i,j} / Σ (j-i)  over the 10 conflict pairs.
Also reports per-gap averages and unweighted macro average.

The gap weighting is novel in this exact form for hierarchy work; the
closest precedent is weighted Kendall's τ in IR (Kumar & Vassilvitskii
2010), where long-range rank inversions are penalised more than adjacent
ones. Reporting the unweighted macro average alongside WHS lets readers
compare against papers that aggregate uniformly.
"""

from collections import defaultdict

from src.evaluation.ppa import CONFLICT_PAIRS


def gap_for_pair(pair: str) -> int:
    """Hierarchy distance for a pair string like ``"L0_vs_L4"``."""
    a, _, b = pair.partition("_vs_")
    i = int(a[1:])
    j = int(b[1:])
    return j - i


def compute_whs(per_pair: dict[str, float], per_pair_count: dict[str, int]) -> dict:
    """Aggregate per-pair PPA into WHS, macro avg, and per-gap averages.

    Args:
        per_pair: PPA per conflict pair, e.g. ``{"L0_vs_L1": 0.5, ...}``.
        per_pair_count: Number of evaluated scenarios per pair. Pairs with
            zero count are excluded from all aggregates.

    Returns:
        Dict with keys:
        - ``whs``: gap-weighted PPA.
        - ``macro_avg``: unweighted mean PPA over populated pairs.
        - ``per_gap_avg``: ``{1: avg, 2: avg, 3: avg, 4: avg}``.
        - ``per_gap_count``: ``{gap: number of populated pairs in that gap}``.
    """
    populated = [p for p in CONFLICT_PAIRS if per_pair_count.get(p, 0) > 0]

    if not populated:
        return {
            "whs": 0.0,
            "macro_avg": 0.0,
            "per_gap_avg": {g: 0.0 for g in (1, 2, 3, 4)},
            "per_gap_count": {g: 0 for g in (1, 2, 3, 4)},
        }

    weighted_sum = sum(gap_for_pair(p) * per_pair[p] for p in populated)
    weight_total = sum(gap_for_pair(p) for p in populated)
    whs = weighted_sum / weight_total if weight_total > 0 else 0.0

    macro_avg = sum(per_pair[p] for p in populated) / len(populated)

    gap_buckets: dict[int, list[float]] = defaultdict(list)
    for p in populated:
        gap_buckets[gap_for_pair(p)].append(per_pair[p])
    per_gap_avg = {
        g: (sum(gap_buckets[g]) / len(gap_buckets[g])) if gap_buckets[g] else 0.0
        for g in (1, 2, 3, 4)
    }
    per_gap_count = {g: len(gap_buckets[g]) for g in (1, 2, 3, 4)}

    return {
        "whs": whs,
        "macro_avg": macro_avg,
        "per_gap_avg": per_gap_avg,
        "per_gap_count": per_gap_count,
    }
