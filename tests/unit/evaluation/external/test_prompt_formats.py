"""Delimited and chat-template prompt builders for external benchmarks."""

from src.evaluation.external.prompt_formats import (
    build_chat_template,
    build_delimited,
)


def test_build_delimited_l3_only() -> None:
    out = build_delimited(l3="What is 2+2?")
    assert out == "<|L3_START|>What is 2+2?<|L3_END|>"


def test_build_delimited_l1_l3() -> None:
    out = build_delimited(l1="You are a maths tutor.", l3="What is 2+2?")
    assert "<|L1_START|>You are a maths tutor.<|L1_END|>" in out
    assert "<|L3_START|>What is 2+2?<|L3_END|>" in out
    # L1 must precede L3
    assert out.index("L1_START") < out.index("L3_START")


def test_build_delimited_with_l4_history() -> None:
    out = build_delimited(
        l1="System.",
        l3="Final user question.",
        l4="USER: prior turn 1\nASSISTANT: prior reply 1",
    )
    assert out.index("L1_START") < out.index("L3_START") < out.index("L4_START")


def test_build_chat_template_user_only() -> None:
    class _T:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is False  # critical: must request string output
            return repr(messages)

    out = build_chat_template(_T(), user="hello")
    assert "'role': 'user'" in out
    assert "'content': 'hello'" in out


def test_build_chat_template_full_payload() -> None:
    class _T:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return repr(messages)

    out = build_chat_template(
        _T(),
        system="You are S.",
        history=[("user", "u1"), ("assistant", "a1")],
        user="final question",
    )
    s_idx = out.index("'role': 'system'")
    u1_idx = out.index("'content': 'u1'")
    a1_idx = out.index("'content': 'a1'")
    final_idx = out.index("'content': 'final question'")
    assert s_idx < u1_idx < a1_idx < final_idx


def test_build_chat_template_passes_tokenize_false() -> None:
    """Regression: must pass tokenize=False so apply_chat_template returns a
    string. Default tokenize=True returns list[int], which breaks the
    downstream tokenizer(prompts, ...) call at run time."""
    captured: dict = {}

    class _T:
        def apply_chat_template(self, messages, **kwargs):
            captured.update(kwargs)
            return "ok"

    out = build_chat_template(_T(), user="hello")
    assert isinstance(out, str)
    assert captured.get("tokenize") is False
    assert captured.get("add_generation_prompt") is True
