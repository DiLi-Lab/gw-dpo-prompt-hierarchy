"""Model nickname → checkpoint path resolver for external-benchmark CLIs.

Exposes a single function :func:`resolve_model` that maps a stable
nickname (``gw_dpo``, ``bilateral``, ...) to a checkpoint path plus
loader hints. Unknown nicknames raise ``ValueError`` with the full list
of known options to fail fast and aid debugging.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config.paths import PathsConfig

ModelKind = Literal["base", "trained"]


@dataclass(frozen=True)
class ResolvedModel:
    """Information needed to load a model for external-benchmark evaluation."""

    nickname: str
    model_path: str
    ise_weights_path: Path | None
    requires_special_tokens: bool
    kind: ModelKind


# Single source of truth. Paths are joined with the project's models_dir at
# resolution time (except base_stock, which is a HuggingFace id).
_RegistryEntry = tuple[str, bool, ModelKind]

KNOWN_MODELS: dict[str, _RegistryEntry] = {
    "base_stock":       ("__hf__:meta-llama/Llama-3.1-8B-Instruct", False, "base"),
    "base_with_tokens": ("base-with-tokens",                        True,  "base"),
    "sft_only":         ("llama-3.1-8b-sft-merged",                 True,  "trained"),
    "standard_dpo":     ("llama-3.1-8b-standard-dpo-final",         True,  "trained"),
    "gw_dpo":           ("llama-3.1-8b-gw-dpo-final",               True,  "trained"),
    "bilateral":        ("llama-3.1-8b-gw-dpo-bilateral-final",     True,  "trained"),
    "three_level":      ("llama-3.1-8b-3level-gw-dpo-final",        True,  "trained"),
    "tokens_only":      ("llama-3.1-8b-gw-dpo-no-ise-final",        True,  "trained"),
}


def resolve_model(nickname: str, paths: PathsConfig) -> ResolvedModel:
    """Resolve a model nickname to a ``ResolvedModel``.

    Args:
        nickname: One of the keys in ``KNOWN_MODELS``.
        paths: Project ``PathsConfig`` for prefixing local paths.

    Raises:
        ValueError: If ``nickname`` is not in ``KNOWN_MODELS``.
    """
    if nickname not in KNOWN_MODELS:
        known = ", ".join(sorted(KNOWN_MODELS))
        msg = (
            f"Unknown model nickname '{nickname}'. Known nicknames: {known}."
        )
        raise ValueError(msg)

    raw_path, requires_special_tokens, kind = KNOWN_MODELS[nickname]

    # ISE weights live next to the merged model directory by convention; the
    # loader auto-detects them. tokens_only is the (f) ablation and explicitly
    # has no ISE.
    ise_weights_path: Path | None = None

    if raw_path.startswith("__hf__:"):
        model_path = raw_path[len("__hf__:") :]
    else:
        model_path = str(paths.models_dir / raw_path)

    return ResolvedModel(
        nickname=nickname,
        model_path=model_path,
        ise_weights_path=ise_weights_path,
        requires_special_tokens=requires_special_tokens,
        kind=kind,
    )
