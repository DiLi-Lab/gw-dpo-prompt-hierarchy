"""MT-Bench JSONL loaders.

The vendored files at ``data/external/mt_bench/`` are pinned to a
specific FastChat commit (see ``data/external/mt_bench/NOTICE``) and are
the source of truth for MT-Bench evaluation. The schemas are fixed by
this loader; upstream drift fails fast with a clear error.
"""

import json
from dataclasses import dataclass
from pathlib import Path

MATH_CATEGORIES: frozenset[str] = frozenset({"math", "coding", "reasoning"})

_EXPECTED_CATEGORIES: frozenset[str] = frozenset({
    "writing", "roleplay", "extraction", "reasoning",
    "math", "coding", "stem", "humanities",
})

_EXPECTED_JUDGE_TEMPLATES: frozenset[str] = frozenset({
    "single-v1",
    "single-math-v1",
    "single-v1-multi-turn",
    "single-math-v1-multi-turn",
})


@dataclass(frozen=True)
class MTBenchQuestion:
    question_id: int
    category: str
    turns: tuple[str, str]


@dataclass(frozen=True)
class ReferenceAnswer:
    question_id: int
    turns: tuple[str, str]


@dataclass(frozen=True)
class JudgePromptTemplate:
    name: str
    system_prompt: str
    prompt_template: str


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_questions(
    path: str | Path,
    *,
    expect_count: int | None = 80,
) -> list[MTBenchQuestion]:
    """Load MT-Bench questions; fail fast on schema drift.

    ``expect_count`` is the canonical 80 for the vendored file. Pass
    ``None`` from tests using subset fixtures.
    """
    rows = _read_jsonl(path)
    questions: list[MTBenchQuestion] = []
    for row in rows:
        turns = row["turns"]
        if not isinstance(turns, list) or len(turns) != 2:
            msg = f"question_id={row.get('question_id')} has != 2 turns"
            raise ValueError(msg)
        if row["category"] not in _EXPECTED_CATEGORIES:
            msg = (
                f"Unexpected category {row['category']!r} for "
                f"question_id={row.get('question_id')}"
            )
            raise ValueError(msg)
        questions.append(MTBenchQuestion(
            question_id=int(row["question_id"]),
            category=row["category"],
            turns=(turns[0], turns[1]),
        ))
    if expect_count is not None and len(questions) != expect_count:
        msg = (
            f"Loaded {len(questions)} questions from {path}; "
            f"expected {expect_count}."
        )
        raise ValueError(msg)
    return questions


def load_reference_answers(path: str | Path) -> dict[int, ReferenceAnswer]:
    """Load gpt-4 reference answers keyed by question_id.

    Only the math / coding / reasoning categories have references. The
    upstream JSONL nests the turns under choices[0]['turns'].
    """
    rows = _read_jsonl(path)
    out: dict[int, ReferenceAnswer] = {}
    for row in rows:
        choices = row.get("choices", [])
        if not choices:
            msg = f"question_id={row.get('question_id')} has no choices"
            raise ValueError(msg)
        turns = choices[0].get("turns", [])
        if not isinstance(turns, list) or len(turns) != 2:
            msg = f"question_id={row.get('question_id')} ref has != 2 turns"
            raise ValueError(msg)
        out[int(row["question_id"])] = ReferenceAnswer(
            question_id=int(row["question_id"]),
            turns=(turns[0], turns[1]),
        )
    return out


def load_judge_prompts(path: str | Path) -> dict[str, JudgePromptTemplate]:
    """Load the 4 MT-Bench single-mode judge templates.

    Upstream ``judge_prompts.jsonl`` carries 8 templates (4 pair-mode
    plus 4 single-mode). We need only the single-mode set; pair-mode
    rows are filtered out via the ``type`` field. The vendored file is
    pinned to a specific FastChat commit, so a missing ``type`` field
    is structural drift and must fail hard.
    """
    rows = _read_jsonl(path)
    out: dict[str, JudgePromptTemplate] = {}
    for row in rows:
        if "type" not in row:
            msg = (
                f"judge_prompts row {row.get('name')!r} missing 'type' field; "
                "upstream schema may have changed — re-pin and update loader."
            )
            raise ValueError(msg)
        if row["type"] != "single":
            continue
        name = row["name"]
        out[name] = JudgePromptTemplate(
            name=name,
            system_prompt=row["system_prompt"],
            prompt_template=row["prompt_template"],
        )
    missing = _EXPECTED_JUDGE_TEMPLATES - set(out.keys())
    if missing:
        msg = f"judge_prompts.jsonl missing required templates: {sorted(missing)}"
        raise ValueError(msg)
    return out
