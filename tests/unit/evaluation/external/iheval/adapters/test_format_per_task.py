"""Per-task IHEval prompt adapters."""

from src.evaluation.external.iheval.adapters import format_record_for_format
from src.evaluation.external.iheval.data import IHEvalRecord


def _record(**kwargs) -> IHEvalRecord:
    base = dict(
        task="single-turn", setting="conflict", sub="default", id=1,
        system="SYS", instruction="USER", conversation_history=None, tool=None,
        answer={},
    )
    base.update(kwargs)
    return IHEvalRecord(**base)


def test_single_turn_delimited_puts_system_at_l1_user_at_l3() -> None:
    out = format_record_for_format(_record(), fmt="delimited", tokenizer=None)
    assert "<|L1_START|>SYS<|L1_END|>" in out
    assert "<|L3_START|>USER<|L3_END|>" in out
    assert out.index("L1_START") < out.index("L3_START")


def test_multi_turn_delimited_history_goes_to_l4() -> None:
    rec = _record(
        task="multi-turn",
        conversation_history=["prior user turn", "prior assistant turn"],
        instruction="final question",
    )
    out = format_record_for_format(rec, fmt="delimited", tokenizer=None)
    assert "<|L4_START|>" in out
    assert "prior user turn" in out
    assert "prior assistant turn" in out
    # final instruction sits at L3, history at L4 (lower priority)
    assert out.index("L3_START") < out.index("L4_START")
    assert "<|L3_START|>final question<|L3_END|>" in out


def test_safety_delimited_uses_l1_l3_only() -> None:
    rec = _record(task="user-prompt-hijack", system="SECRET", instruction="ATTACK")
    out = format_record_for_format(rec, fmt="delimited", tokenizer=None)
    assert "<|L1_START|>SECRET<|L1_END|>" in out
    assert "<|L3_START|>ATTACK<|L3_END|>" in out


def test_chat_template_dispatches_to_tokenizer() -> None:
    class _T:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert tokenize is False
            return repr(messages)

    rec = _record(
        task="multi-turn",
        conversation_history=["u", "a", "u2", "a2"],
        instruction="final",
    )
    out = format_record_for_format(rec, fmt="chat_template", tokenizer=_T())
    assert "'role': 'system'" in out
    # Conversation history alternates user/assistant starting with user.
    assert out.index("'content': 'u'") < out.index("'content': 'a'") < out.index("'content': 'final'")
