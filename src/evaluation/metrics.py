"""Top-level metric orchestration: aggregate all metric streams.

Pure aggregation only — generation, judging, and refusal classification
happen upstream. Inputs are already-judged conflict records, already-
judged reference records, and aligned-control records carrying refusal
flags. Outputs the full metrics dict and writes it to JSON.

When raw response records are supplied, a complementary "non-empty" view
of the metrics is also emitted: PPA/WHS/utility-delta restricted to rows
where the model produced a non-empty response, plus per-split response
completion rates. This separates "did the model emit text at all?" from
"given that it did, did it pick the right level?".
"""

import json
from pathlib import Path

from src.evaluation.over_refusal import compute_orr
from src.evaluation.ppa import compute_ppa
from src.evaluation.utility_delta import compute_utility_delta
from src.evaluation.whs import compute_whs


def _empty_ids(responses: list[dict]) -> set[str]:
    """IDs whose response is empty after .strip() (i.e. the RESP_END artifact)."""
    return {
        r["id"] for r in responses
        if not (r.get("response") or "").strip()
    }


def _completion_rate(responses: list[dict]) -> float:
    """Fraction of responses with non-empty content."""
    if not responses:
        return 0.0
    return 1.0 - len(_empty_ids(responses)) / len(responses)


def aggregate_all_metrics(
    judged_conflicts: list[dict],
    judged_reference: list[dict],
    aligned_refusals: list[dict],
    pair_lookup: dict[str, str],
    reward_metrics: dict | None = None,
    text_similarity: dict | None = None,
    *,
    conflict_responses: list[dict] | None = None,
    reference_responses: list[dict] | None = None,
    aligned_responses: list[dict] | None = None,
) -> dict:
    """Aggregate all per-instance signals into the final metrics dict.

    Args:
        judged_conflicts: Output of ``score_responses`` on the conflict split.
        judged_reference: Output of ``score_responses`` on the reference
            (flat-text) split. May be empty.
        aligned_refusals: List of ``{"matched_conflict_id": ..., "is_refusal":
            bool}`` produced by ``classify_refusal`` over the aligned split.
        pair_lookup: Map from conflict id to conflict pair string.
        reward_metrics: Optional reward-accuracy / margin block.
        text_similarity: Optional BERTScore / ROUGE-L block.
        conflict_responses: Optional raw response records for the conflict
            split (each ``{"id": ..., "response": ...}``). When supplied,
            ``response_completion_rate`` and ``*_non_empty`` PPA/WHS variants
            are emitted.
        reference_responses: Same for the reference split.
        aligned_responses: Same for the aligned-control split (used only for
            ``response_completion_rate``).

    Returns:
        Flat dict of all metrics.
    """
    ppa = compute_ppa(judged_conflicts)
    whs = compute_whs(ppa["per_pair"], ppa["per_pair_count"])
    orr = compute_orr(aligned_refusals, pair_lookup)

    if judged_reference:
        ref_ppa = compute_ppa(judged_reference)
        # Restrict utility delta to pairs that are populated in BOTH splits,
        # otherwise pairs only seen in the conflict split would dominate Δ
        # with a baseline of 0.
        common = {
            p for p, c in ref_ppa["per_pair_count"].items() if c > 0
        } & {
            p for p, c in ppa["per_pair_count"].items() if c > 0
        }
        delta = compute_utility_delta(
            {p: ppa["per_pair"][p] for p in common},
            {p: ref_ppa["per_pair"][p] for p in common},
        )
    else:
        ref_ppa = {
            "per_pair": {}, "per_pair_count": {},
            "asr_per_pair": {}, "macro_avg": 0.0,
        }
        delta = {"per_pair_delta": {}, "mean_delta": 0.0, "mean_abs_delta": 0.0}

    out: dict = {
        "ppa_per_pair": ppa["per_pair"],
        "ppa_per_pair_count": ppa["per_pair_count"],
        "asr_per_pair": ppa["asr_per_pair"],
        "ppa_macro": ppa["macro_avg"],
        "whs": whs["whs"],
        "per_gap_avg": whs["per_gap_avg"],
        "per_gap_count": whs["per_gap_count"],
        "orr_overall": orr["orr_overall"],
        "orr_per_pair": orr["orr_per_pair"],
        "orr_per_pair_count": orr["per_pair_count"],
        "reference_ppa_per_pair": ref_ppa["per_pair"],
        "reference_ppa_macro": ref_ppa["macro_avg"],
        "utility_delta_per_pair": delta["per_pair_delta"],
        "utility_delta_mean": delta["mean_delta"],
        "utility_delta_mean_abs": delta["mean_abs_delta"],
    }

    # Optional non-empty / completion-rate decomposition. Computed only when
    # raw response records are supplied so existing call sites stay valid.
    completion: dict[str, float] = {}
    empty_conflict_ids: set[str] = set()
    empty_reference_ids: set[str] = set()

    if conflict_responses is not None:
        completion["conflicts"] = _completion_rate(conflict_responses)
        empty_conflict_ids = _empty_ids(conflict_responses)

        ne_judged = [
            j for j in judged_conflicts if j.get("id") not in empty_conflict_ids
        ]
        ne_ppa = compute_ppa(ne_judged)
        ne_whs = compute_whs(ne_ppa["per_pair"], ne_ppa["per_pair_count"])
        out["ppa_per_pair_non_empty"] = ne_ppa["per_pair"]
        out["ppa_per_pair_count_non_empty"] = ne_ppa["per_pair_count"]
        out["asr_per_pair_non_empty"] = ne_ppa["asr_per_pair"]
        out["ppa_macro_non_empty"] = ne_ppa["macro_avg"]
        out["whs_non_empty"] = ne_whs["whs"]
        out["per_gap_avg_non_empty"] = ne_whs["per_gap_avg"]
        out["per_gap_count_non_empty"] = ne_whs["per_gap_count"]

    if reference_responses is not None:
        completion["reference"] = _completion_rate(reference_responses)
        empty_reference_ids = _empty_ids(reference_responses)
        if judged_reference:
            ne_ref_judged = [
                j for j in judged_reference
                if j.get("id") not in empty_reference_ids
            ]
            ne_ref_ppa = compute_ppa(ne_ref_judged)
            out["reference_ppa_per_pair_non_empty"] = ne_ref_ppa["per_pair"]
            out["reference_ppa_per_pair_count_non_empty"] = ne_ref_ppa["per_pair_count"]
            out["reference_ppa_macro_non_empty"] = ne_ref_ppa["macro_avg"]

    if aligned_responses is not None:
        completion["aligned"] = _completion_rate(aligned_responses)

    if completion:
        out["response_completion_rate"] = completion

    # Non-empty utility delta needs both conflict and reference non-empty
    # views; only emit when both are present.
    if (
        "ppa_per_pair_non_empty" in out
        and "reference_ppa_per_pair_non_empty" in out
    ):
        ne_common = {
            p for p, c in out["reference_ppa_per_pair_count_non_empty"].items()
            if c > 0
        } & {
            p for p, c in out["ppa_per_pair_count_non_empty"].items() if c > 0
        }
        ne_delta = compute_utility_delta(
            {p: out["ppa_per_pair_non_empty"][p] for p in ne_common},
            {p: out["reference_ppa_per_pair_non_empty"][p] for p in ne_common},
        )
        out["utility_delta_per_pair_non_empty"] = ne_delta["per_pair_delta"]
        out["utility_delta_mean_non_empty"] = ne_delta["mean_delta"]
        out["utility_delta_mean_abs_non_empty"] = ne_delta["mean_abs_delta"]

    if reward_metrics is not None:
        out["reward_metrics"] = reward_metrics
    if text_similarity is not None:
        out["text_similarity"] = text_similarity
    return out


def write_metrics_json(metrics: dict, path: Path) -> None:
    """Persist a metrics dict to JSON (parents created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
