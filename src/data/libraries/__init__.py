"""Content libraries for the 5-level instruction hierarchy.

Provides loaders and generators for L0 rules, L1 developer system prompts,
L2 per-user configuration, L3 user messages, L4 tool outputs, and injection
templates.
"""

from src.data.libraries.injection_templates import (
    InjectionTemplate,
    inject_into_content,
    load_injection_templates,
    sample_injection_prefix,
)
from src.data.libraries.l0_rules import L0Rule, load_l0_rules, sample_l0_rules
from src.data.libraries.l1_prompts import (
    generate_l1_library,
    load_l1_library,
)
from src.data.libraries.l2_templates import (
    L2Config,
    generate_l2,
    generate_l2_batch,
    generate_l2_config,
    generate_l2_for_conflict,
)
from src.data.libraries.l3_user_messages import (
    L3Message,
    filter_l3_candidates,
    load_l3_pool,
    sample_l3_message,
    validate_l3_pool,
)
from src.data.libraries.l4_tool_outputs import (
    L4Entry,
    MIN_CONTENT_CHARS,
    PLACEHOLDER_PATTERNS,
    build_l4_wrapped,
    is_placeholder,
    load_l4_library,
    save_l4_library,
    validate_l4_library,
    wrap_as_l4,
)

__all__ = [
    "InjectionTemplate",
    "L0Rule",
    "L2Config",
    "L3Message",
    "L4Entry",
    "MIN_CONTENT_CHARS",
    "PLACEHOLDER_PATTERNS",
    "build_l4_wrapped",
    "is_placeholder",
    "filter_l3_candidates",
    "generate_l1_library",
    "generate_l2",
    "generate_l2_batch",
    "generate_l2_config",
    "generate_l2_for_conflict",
    "inject_into_content",
    "load_injection_templates",
    "load_l0_rules",
    "load_l1_library",
    "load_l3_pool",
    "load_l4_library",
    "sample_injection_prefix",
    "sample_l0_rules",
    "sample_l3_message",
    "save_l4_library",
    "validate_l3_pool",
    "validate_l4_library",
    "wrap_as_l4",
]
