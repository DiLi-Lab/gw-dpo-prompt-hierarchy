"""Unit tests for MT-Bench prompt builders."""

from src.evaluation.external.mt_bench.data import MTBenchQuestion
from src.evaluation.external.mt_bench.prompt import (
    build_turn1_prompt,
    build_turn2_prompt,
)


class _StubTokenizer:
    """Captures messages and returns a deterministic rendered string."""

    def __init__(self) -> None:
        self.captured: list[dict] = []
        self.captured_kwargs: dict = {}

    def apply_chat_template(self, messages, **kwargs):
        self.captured = messages
        self.captured_kwargs = kwargs
        rendered_parts = []
        for m in messages:
            rendered_parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        if kwargs.get("add_generation_prompt"):
            rendered_parts.append("<assistant>")
        return "".join(rendered_parts)


def _q() -> MTBenchQuestion:
    return MTBenchQuestion(
        question_id=42,
        category="writing",
        turns=("First turn user message.", "Second turn user message."),
    )


def test_turn1_emits_user_only_no_system() -> None:
    tok = _StubTokenizer()
    rendered = build_turn1_prompt(tok, _q())
    assert tok.captured == [
        {"role": "user", "content": "First turn user message."},
    ]
    assert tok.captured_kwargs.get("add_generation_prompt") is True
    assert tok.captured_kwargs.get("tokenize") is False
    assert "<user>First turn user message.</user>" in rendered
    assert "<assistant>" in rendered  # generation marker


def test_turn2_emits_user_assistant_user_no_system() -> None:
    tok = _StubTokenizer()
    rendered = build_turn2_prompt(tok, _q(), turn1_response="First answer.")
    assert tok.captured == [
        {"role": "user",      "content": "First turn user message."},
        {"role": "assistant", "content": "First answer."},
        {"role": "user",      "content": "Second turn user message."},
    ]
    assert tok.captured_kwargs.get("add_generation_prompt") is True
    assert tok.captured_kwargs.get("tokenize") is False
    assert rendered.endswith("<assistant>")


def test_turn1_does_not_inject_system_prompt() -> None:
    tok = _StubTokenizer()
    build_turn1_prompt(tok, _q())
    roles = [m["role"] for m in tok.captured]
    assert "system" not in roles


def test_turn2_does_not_inject_system_prompt() -> None:
    tok = _StubTokenizer()
    build_turn2_prompt(tok, _q(), turn1_response="ok")
    roles = [m["role"] for m in tok.captured]
    assert "system" not in roles
