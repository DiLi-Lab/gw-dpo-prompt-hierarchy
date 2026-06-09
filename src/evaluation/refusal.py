"""Two-stage refusal classifier: regex fast-path + judge confirmation.

Stage A: ``regex_refusal_match`` checks for known refusal patterns from
``src.data.dpo.response_utils.REFUSAL_PATTERNS``. If no match, the response
is not a refusal — we are done.

Stage B: ``needs_judge_confirmation`` returns True for responses long
enough that a hedge-then-comply pattern is plausible. For these, an
LLM judge is asked to confirm whether the response is genuinely a refusal
(returns True) or just a hedged compliance (returns False).

This split keeps judge calls cheap: short refusals don't need confirmation,
non-matches don't need any LLM call at all.
"""

from typing import Callable

from src.data.dpo.response_utils import is_refusal


def regex_refusal_match(response: str) -> bool:
    """Stage A: exact-pattern match against REFUSAL_PATTERNS."""
    return is_refusal(response)


def needs_judge_confirmation(response: str, min_chars: int) -> bool:
    """Stage B trigger: regex matched AND response long enough that a
    hedge-then-comply pattern is plausible."""
    return regex_refusal_match(response) and len(response) >= min_chars


def classify_refusal(
    response: str,
    min_chars: int,
    judge_fn: Callable[[str], bool] | None,
) -> dict:
    """Classify a response as refusal or not.

    Args:
        response: Model output to classify.
        min_chars: Length threshold above which a regex match triggers
            judge confirmation.
        judge_fn: Callable taking the response text and returning True iff
            it is a genuine refusal. Pass ``None`` to skip the judge stage
            (useful for tests and when the API is unavailable).

    Returns:
        Dict with keys ``is_refusal`` (bool) and ``judge_used`` (bool).
    """
    if not regex_refusal_match(response):
        return {"is_refusal": False, "judge_used": False}

    if not needs_judge_confirmation(response, min_chars=min_chars):
        return {"is_refusal": True, "judge_used": False}

    if judge_fn is None:
        # Without a judge, conservatively trust the regex match.
        return {"is_refusal": True, "judge_used": False}

    return {"is_refusal": bool(judge_fn(response)), "judge_used": True}
