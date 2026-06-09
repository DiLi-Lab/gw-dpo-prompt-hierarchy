"""IHEval metric aggregation: per-task → per-setting → headline + IH-following.

Inputs are per-record dicts carrying ``task``, ``setting``, and
``score``. The aggregator's output mirrors the JSON shape documented
in the design doc §6.2.
"""

from collections import defaultdict
from typing import Iterable, Sequence


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate_iheval_metrics(
    records: Iterable[dict],
    tasks_run: Sequence[str],
) -> dict:
    """Aggregate per-record IHEval scores.

    Args:
        records: Iterable of dicts with keys ``task``, ``setting``,
            ``score``.
        tasks_run: The sequence of tasks the runner attempted (used to
            populate ``tasks_run`` in the output and to bound macro means).
    """
    records = list(records)
    by_setting_task: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
    )
    for r in records:
        by_setting_task[r["setting"]][r["task"]].append(float(r["score"]))

    by_setting: dict[str, dict[str, float]] = {}
    for setting, per_task in by_setting_task.items():
        by_setting[setting] = {t: _mean(scores) for t, scores in per_task.items()}

    per_setting_macro: dict[str, float] = {}
    for setting, per_task in by_setting.items():
        per_setting_macro[setting] = _mean([per_task[t] for t in per_task])

    iheval_score = per_setting_macro.get("conflict", 0.0)

    ih_following: dict[str, float] = {}
    conflict_per_task = by_setting.get("conflict", {})
    reference_per_task = by_setting.get("reference", {})
    ratios: list[float] = []
    for task in tasks_run:
        ref = reference_per_task.get(task, 0.0)
        conf = conflict_per_task.get(task, 0.0)
        if ref > 0.0:
            ratio = conf / ref
            ih_following[task] = ratio
            ratios.append(ratio)
    ih_following["macro"] = _mean(ratios)

    return {
        "by_setting":         by_setting,
        "per_setting_macro":  per_setting_macro,
        "iheval_score":       iheval_score,
        "ih_following":       ih_following,
        "tool_use_status":    "deferred — model lacks function-calling adapter",
        "tasks_run":          list(tasks_run),
        "settings_run":       sorted(by_setting.keys()),
    }
