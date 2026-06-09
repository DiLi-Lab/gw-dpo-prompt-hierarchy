"""XSTest metrics aggregator."""

from src.evaluation.external.xstest.metrics import aggregate_xstest_metrics


def _safe(judge_label: str, type_: str = "homonyms") -> dict:
    return {"label": "safe", "judge_label": judge_label, "type": type_}


def _unsafe(judge_label: str, type_: str = "contrast_homonyms") -> dict:
    return {"label": "unsafe", "judge_label": judge_label, "type": type_}


def test_compliance_and_refusal_rates() -> None:
    records = [
        _safe("1_full_compliance"),
        _safe("1_full_compliance"),
        _safe("2_full_refusal"),
        _safe("3_partial_refusal"),
        _unsafe("2_full_refusal"),
        _unsafe("1_full_compliance"),
    ]
    m = aggregate_xstest_metrics(records)
    assert m["n_safe"] == 4
    assert m["n_unsafe"] == 2
    assert m["compliance_rate_safe"] == 0.5            # 2 / 4
    assert m["full_refusal_rate_safe"] == 0.25         # 1 / 4
    assert m["partial_refusal_rate_safe"] == 0.25      # 1 / 4
    assert m["false_refusal_rate_safe"] == 0.5         # full + partial
    assert m["refusal_rate_unsafe"] == 0.5             # 1 / 2
    assert m["compliance_rate_unsafe"] == 0.5


def test_per_type_breakdown() -> None:
    records = [
        _safe("1_full_compliance", type_="homonyms"),
        _safe("2_full_refusal", type_="homonyms"),
        _safe("1_full_compliance", type_="figurative_language"),
    ]
    m = aggregate_xstest_metrics(records)
    by_type = m["by_type"]
    assert by_type["homonyms"]["compliance_rate_safe"] == 0.5
    assert by_type["figurative_language"]["compliance_rate_safe"] == 1.0


def test_parse_errors_reported() -> None:
    records = [_safe("unknown"), _safe("1_full_compliance")]
    m = aggregate_xstest_metrics(records)
    assert m["parse_errors"] == 1
    # Compliance/refusal rates compute over labelled records only.
    assert m["compliance_rate_safe"] == 1.0


def test_empty_inputs_return_zero_n_and_zero_rates() -> None:
    m = aggregate_xstest_metrics([])
    assert m["n_safe"] == 0
    assert m["n_unsafe"] == 0
    assert m["compliance_rate_safe"] == 0.0
    assert m["refusal_rate_unsafe"] == 0.0
