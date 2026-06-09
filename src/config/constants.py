"""Constants for the 5-level instruction hierarchy.

Defines hierarchy level names, special delimiter tokens, and segment IDs
used throughout the project. These are immutable and shared across all
modules.
"""

HIERARCHY_LEVELS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4")
NUM_LEVELS: int = 5

LEVEL_NAMES: dict[str, str] = {
    "L0": "Platform Governance",
    "L1": "Developer System Prompt",
    "L2": "Per-User Configuration",
    "L3": "User Messages",
    "L4": "Data/Tool Outputs",
}

SPECIAL_TOKENS: list[str] = [
    "<|L0_START|>", "<|L0_END|>",
    "<|L1_START|>", "<|L1_END|>",
    "<|L2_START|>", "<|L2_END|>",
    "<|L3_START|>", "<|L3_END|>",
    "<|L4_START|>", "<|L4_END|>",
    "<|RESP_START|>", "<|RESP_END|>",
]

NUM_SEGMENTS: int = 6
RESPONSE_SEGMENT_ID: int = 5

LEVEL_TO_SEGMENT_ID: dict[str, int] = {
    f"L{i}": i for i in range(NUM_LEVELS)
}
LEVEL_TO_SEGMENT_ID["response"] = RESPONSE_SEGMENT_ID

LLAMA_MODEL_PATH: str = "meta-llama/Llama-3.1-8B-Instruct"
