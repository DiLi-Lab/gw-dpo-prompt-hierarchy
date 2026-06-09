"""SEP metric aggregator."""

from src.evaluation.external.sep.metrics import aggregate_sep_metrics


def _rec(
    *,
    id: int,
    domain: str = "qa",
    witness_match: bool,
    non_trivial: bool,
    skip_reason: str | None = None,
) -> dict:
    return {
        "id": id,
        "domain": domain,
        "witness": "W",
        "response": "...",
        "witness_match": witness_match,
        "non_trivial": non_trivial,
        "skip_reason": skip_reason,
    }


def test_aggregator_basic_rates() -> None:
    records = [
        _rec(id=1, witness_match=True,  non_trivial=True),
        _rec(id=2, witness_match=False, non_trivial=True),
        _rec(id=3, witness_match=False, non_trivial=True),
        _rec(id=4, witness_match=True,  non_trivial=True),
    ]
    m = aggregate_sep_metrics(records)
    assert m["n_total"] == 4
    assert m["n_non_trivial"] == 4
    assert m["n_skipped"] == 0
    assert m["n_witness_match"] == 2
    assert m["probe_execution_rate_evaluable"] == 0.5
    assert m["probe_execution_rate_all"] == 0.5
    assert m["non_trivial_response_rate"] == 1.0


def test_aggregator_with_skipped_responses() -> None:
    records = [
        _rec(id=1, witness_match=True,  non_trivial=True),
        _rec(id=2, witness_match=False, non_trivial=True),
        _rec(id=3, witness_match=False, non_trivial=False, skip_reason="too_short"),
        _rec(id=4, witness_match=False, non_trivial=False, skip_reason="refusal_pattern_match"),
    ]
    m = aggregate_sep_metrics(records)
    assert m["n_total"] == 4
    assert m["n_non_trivial"] == 2
    assert m["n_skipped"] == 2
    assert m["n_witness_match"] == 1
    # Headline: 1 / 2 evaluable
    assert m["probe_execution_rate_evaluable"] == 0.5
    # All-responses: 1 / 4
    assert m["probe_execution_rate_all"] == 0.25
    assert m["non_trivial_response_rate"] == 0.5
    assert m["skip_reason_breakdown"] == {
        "too_short": 1,
        "refusal_pattern_match": 1,
    }


def test_aggregator_handles_refuse_everywhere_without_div_zero() -> None:
    records = [
        _rec(id=1, witness_match=False, non_trivial=False, skip_reason="refusal_pattern_match"),
        _rec(id=2, witness_match=False, non_trivial=False, skip_reason="refusal_pattern_match"),
    ]
    m = aggregate_sep_metrics(records)
    assert m["n_non_trivial"] == 0
    # Convention: empty denominator → 0.0, not NaN.
    assert m["probe_execution_rate_evaluable"] == 0.0
    assert m["probe_execution_rate_all"] == 0.0
    assert m["non_trivial_response_rate"] == 0.0


def test_aggregator_per_domain_breakdown() -> None:
    records = [
        _rec(id=1, domain="qa",            witness_match=True,  non_trivial=True),
        _rec(id=2, domain="qa",            witness_match=False, non_trivial=True),
        _rec(id=3, domain="summarization", witness_match=True,  non_trivial=True),
        _rec(id=4, domain="summarization", witness_match=True,  non_trivial=True),
        _rec(id=5, domain="summarization", witness_match=False, non_trivial=False,
             skip_reason="too_short"),
    ]
    m = aggregate_sep_metrics(records)
    qa = m["by_domain"]["qa"]
    assert qa == {
        "n": 2, "n_non_trivial": 2, "n_witness_match": 1,
        "probe_execution_rate_evaluable": 0.5,
        "probe_execution_rate_all": 0.5,
    }
    summ = m["by_domain"]["summarization"]
    assert summ == {
        "n": 3, "n_non_trivial": 2, "n_witness_match": 2,
        "probe_execution_rate_evaluable": 1.0,
        "probe_execution_rate_all": 2 / 3,
    }


def test_aggregator_handles_empty_input() -> None:
    m = aggregate_sep_metrics([])
    assert m["n_total"] == 0
    assert m["n_non_trivial"] == 0
    assert m["n_skipped"] == 0
    assert m["n_witness_match"] == 0
    assert m["probe_execution_rate_evaluable"] == 0.0
    assert m["probe_execution_rate_all"] == 0.0
    assert m["non_trivial_response_rate"] == 0.0
    assert m["skip_reason_breakdown"] == {}
    assert m["by_domain"] == {}
