"""Integration-style test for the metrics aggregator.

Generation, scoring, and refusal classification are mocked; we verify
the orchestrator wires the aggregators correctly and emits a complete
metrics.json with all required keys.
"""

import json
import math
from pathlib import Path

from src.evaluation.metrics import aggregate_all_metrics, write_metrics_json


def test_aggregate_all_metrics_emits_required_keys() -> None:
    judged_conflicts = [
        {"id": "eval_0001", "conflict_type": "L0_vs_L4",
         "satisfies_higher_level": True, "follows_lower_level": False},
        {"id": "eval_0002", "conflict_type": "L1_vs_L3",
         "satisfies_higher_level": False, "follows_lower_level": True},
        {"id": "eval_0003", "conflict_type": "L1_vs_L3",
         "satisfies_higher_level": True, "follows_lower_level": False},
    ]
    judged_reference = [
        {"id": "ref_0001", "conflict_type": "L0_vs_L4",
         "satisfies_higher_level": True, "follows_lower_level": False},
        {"id": "ref_0002", "conflict_type": "L1_vs_L3",
         "satisfies_higher_level": True, "follows_lower_level": False},
    ]
    aligned_refusals = [
        {"matched_conflict_id": "eval_0001", "is_refusal": False},
        {"matched_conflict_id": "eval_0002", "is_refusal": True},
        {"matched_conflict_id": "eval_0003", "is_refusal": False},
    ]
    pair_lookup = {
        "eval_0001": "L0_vs_L4",
        "eval_0002": "L1_vs_L3",
        "eval_0003": "L1_vs_L3",
    }

    out = aggregate_all_metrics(
        judged_conflicts=judged_conflicts,
        judged_reference=judged_reference,
        aligned_refusals=aligned_refusals,
        pair_lookup=pair_lookup,
    )
    # Required top-level keys
    for key in (
        "ppa_per_pair", "asr_per_pair", "ppa_macro", "ppa_per_pair_count",
        "whs", "per_gap_avg",
        "orr_overall", "orr_per_pair",
        "utility_delta_per_pair", "utility_delta_mean", "utility_delta_mean_abs",
    ):
        assert key in out, f"missing key {key}"

    # PPA on conflicts: L1_vs_L3 = 1/2 = 0.5; L0_vs_L4 = 1/1 = 1.0
    assert out["ppa_per_pair"]["L1_vs_L3"] == 0.5
    assert out["ppa_per_pair"]["L0_vs_L4"] == 1.0
    assert out["asr_per_pair"]["L1_vs_L3"] == 0.5

    # ORR: 1 / 3
    assert out["orr_overall"] == 1 / 3

    # Utility Δ: PPA_conflict − PPA_reference for L0_vs_L4: 1.0-1.0=0;
    # L1_vs_L3: 0.5-1.0=-0.5
    assert out["utility_delta_per_pair"]["L1_vs_L3"] == -0.5


def test_aggregate_all_metrics_handles_no_reference() -> None:
    """Reference split is optional; utility delta returns empty when absent."""
    out = aggregate_all_metrics(
        judged_conflicts=[
            {"id": "eval_0001", "conflict_type": "L0_vs_L4",
             "satisfies_higher_level": True, "follows_lower_level": False},
        ],
        judged_reference=[],
        aligned_refusals=[],
        pair_lookup={},
    )
    assert out["utility_delta_per_pair"] == {}
    assert out["utility_delta_mean"] == 0.0


def test_write_metrics_json_round_trips(tmp_path: Path) -> None:
    metrics = {"whs": 0.5, "ppa_macro": 0.4}
    out_path = tmp_path / "metrics.json"
    write_metrics_json(metrics, out_path)
    loaded = json.loads(out_path.read_text())
    assert loaded == metrics


def test_aggregate_all_metrics_includes_optional_blocks() -> None:
    """Reward + similarity blocks are forwarded when supplied."""
    out = aggregate_all_metrics(
        judged_conflicts=[],
        judged_reference=[],
        aligned_refusals=[],
        pair_lookup={},
        reward_metrics={"macro_avg_accuracy": 0.7, "mean_reward_margin": 1.2},
        text_similarity={"bertscore_f1": 0.85, "rougeL": 0.4},
    )
    assert out["reward_metrics"]["macro_avg_accuracy"] == 0.7
    assert out["text_similarity"]["bertscore_f1"] == 0.85


def test_aggregate_emits_completion_and_non_empty_when_responses_given() -> None:
    """When response records are supplied, completion rate + non-empty PPA
    are emitted, and the empty rows are excluded from the non-empty view."""
    judged_conflicts = [
        # eval_0001: empty response → counted as wrong in full PPA, dropped from non-empty
        {"id": "eval_0001", "conflict_type": "L1_vs_L3",
         "satisfies_higher_level": False, "follows_lower_level": False},
        # eval_0002: non-empty, attacker won → wrong in both views
        {"id": "eval_0002", "conflict_type": "L1_vs_L3",
         "satisfies_higher_level": False, "follows_lower_level": True},
        # eval_0003: non-empty, correct
        {"id": "eval_0003", "conflict_type": "L1_vs_L3",
         "satisfies_higher_level": True, "follows_lower_level": False},
    ]
    conflict_responses = [
        {"id": "eval_0001", "response": ""},
        {"id": "eval_0002", "response": "some long answer"},
        {"id": "eval_0003", "response": "another answer"},
    ]
    aligned_responses = [
        {"id": "ctrl_0001", "response": ""},
        {"id": "ctrl_0002", "response": "ok"},
    ]
    out = aggregate_all_metrics(
        judged_conflicts=judged_conflicts,
        judged_reference=[],
        aligned_refusals=[],
        pair_lookup={},
        conflict_responses=conflict_responses,
        aligned_responses=aligned_responses,
    )

    # Full PPA counts the empty as wrong: 1/3
    assert math.isclose(out["ppa_per_pair"]["L1_vs_L3"], 1 / 3)
    # Non-empty PPA drops eval_0001: 1/2
    assert out["ppa_per_pair_non_empty"]["L1_vs_L3"] == 0.5
    assert out["ppa_per_pair_count_non_empty"]["L1_vs_L3"] == 2
    assert out["ppa_macro_non_empty"] == 0.5
    assert "whs_non_empty" in out

    # Completion rates
    assert math.isclose(out["response_completion_rate"]["conflicts"], 2 / 3)
    assert out["response_completion_rate"]["aligned"] == 0.5


def test_aggregate_non_empty_utility_delta_when_both_splits_present() -> None:
    """Non-empty utility delta is computed when both conflict and reference
    responses are supplied."""
    judged_conflicts = [
        {"id": "eval_0001", "conflict_type": "L1_vs_L3",
         "satisfies_higher_level": False, "follows_lower_level": False},
        {"id": "eval_0002", "conflict_type": "L1_vs_L3",
         "satisfies_higher_level": True, "follows_lower_level": False},
    ]
    judged_reference = [
        {"id": "ref_0001", "conflict_type": "L1_vs_L3",
         "satisfies_higher_level": True, "follows_lower_level": False},
        {"id": "ref_0002", "conflict_type": "L1_vs_L3",
         "satisfies_higher_level": True, "follows_lower_level": False},
    ]
    out = aggregate_all_metrics(
        judged_conflicts=judged_conflicts,
        judged_reference=judged_reference,
        aligned_refusals=[],
        pair_lookup={},
        conflict_responses=[
            {"id": "eval_0001", "response": ""},
            {"id": "eval_0002", "response": "answer"},
        ],
        reference_responses=[
            {"id": "ref_0001", "response": "answer"},
            {"id": "ref_0002", "response": "answer"},
        ],
    )
    # Non-empty conflict PPA: 1/1 = 1.0; non-empty reference PPA: 2/2 = 1.0;
    # delta = 0
    assert out["ppa_per_pair_non_empty"]["L1_vs_L3"] == 1.0
    assert out["reference_ppa_per_pair_non_empty"]["L1_vs_L3"] == 1.0
    assert out["utility_delta_per_pair_non_empty"]["L1_vs_L3"] == 0.0
    assert out["utility_delta_mean_non_empty"] == 0.0


def test_aggregate_no_responses_omits_non_empty_keys() -> None:
    """Without response records, non-empty / completion fields are NOT emitted
    (back-compat with existing callers)."""
    out = aggregate_all_metrics(
        judged_conflicts=[
            {"id": "eval_0001", "conflict_type": "L0_vs_L4",
             "satisfies_higher_level": True, "follows_lower_level": False},
        ],
        judged_reference=[],
        aligned_refusals=[],
        pair_lookup={},
    )
    assert "ppa_macro_non_empty" not in out
    assert "response_completion_rate" not in out
    assert "whs_non_empty" not in out
