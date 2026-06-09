"""MT-Bench per-question score record.

Bundles the two per-turn judge outcomes into a single record. The
runner constructs one of these per question after both judge calls
have completed (or parse-failed). The aggregator works directly off
per-turn scores in scoring.jsonl, so this dataclass exists primarily
for diagnostic reading of scoring.jsonl rather than for the headline
aggregation.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MTBenchQuestionScore:
    question_id: int
    category: str
    turn1_score: float | None
    turn2_score: float | None
    mean_score: float | None


def build_question_score(
    *,
    question_id: int,
    category: str,
    turn1_score: float | None,
    turn2_score: float | None,
) -> MTBenchQuestionScore:
    """Bundle two per-turn judge scores into a per-question record.

    ``mean_score`` is ``None`` whenever either turn parse-failed — a
    half-scored question never contributes a synthetic mean to the
    diagnostic per-question view.
    """
    if turn1_score is not None and turn2_score is not None:
        mean: float | None = (turn1_score + turn2_score) / 2.0
    else:
        mean = None
    return MTBenchQuestionScore(
        question_id=question_id,
        category=category,
        turn1_score=turn1_score,
        turn2_score=turn2_score,
        mean_score=mean,
    )
