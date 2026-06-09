"""SFT dataset construction for the 5-level instruction hierarchy.

Provides builders for aligned, partial-level, and trivially misaligned
training examples, plus assembly and I/O utilities.
"""

from src.data.sft.assembly import assemble_instance, assemble_sft_example
from src.data.sft.build_sft_dataset import (
    compute_sft_stats,
    load_sft_dataset,
    save_sft_dataset,
)
from src.data.sft.domain_classifier import classify_domain, select_matched_l1
from src.data.sft.row_utils import get_input, get_output

__all__ = [
    "assemble_instance",
    "assemble_sft_example",
    "classify_domain",
    "compute_sft_stats",
    "get_input",
    "get_output",
    "load_sft_dataset",
    "save_sft_dataset",
    "select_matched_l1",
]
