"""SEP Mapping-A prompt builders.

Mapping A places the SEP instruction in L1 (system) and the data slot
(with the embedded probe) in L3 (user). This is the published-leaderboard
comparable mapping per the design (§4.3).

Mapping B (instruction → L3, data → L4) is a v2 follow-up that adds new
builders here without touching the existing ones.
"""

from typing import Any

from src.evaluation.external.prompt_formats import build_chat_template, build_delimited
from src.evaluation.external.sep.data import SEPRecord


def build_sep_delimited_mapping_a(record: SEPRecord) -> str:
    """Wrap (instruction, data) in <|L1_*|> + <|L3_*|> delimiters.

    Returns the level-tagged prompt body. The runner appends the
    ``<|RESP_START|>`` marker, matching the XSTest pattern.
    """
    return build_delimited(
        l1=record.instruction,
        l3=record.data_with_witness,
    )


def build_sep_chat_template(tokenizer: Any, record: SEPRecord) -> str:
    """Render (instruction, data) as a system+user chat template prompt."""
    return build_chat_template(
        tokenizer,
        system=record.instruction,
        user=record.data_with_witness,
    )
