"""Thin wrappers around the upstream IHEval scorers under vendor/iheval/.

The upstream package is rooted at ``vendor/iheval/src/...``. The naïve
strategy of adding ``vendor/iheval/`` to :data:`sys.path` collides with
this project's own ``src/`` regular package — Python resolves
``from src.task_execution.evaluate ...`` against our ``src/`` and never
falls through to the namespace-package candidate inside the submodule.

Workaround: load the four standalone scorer files directly via
:func:`importlib.util.spec_from_file_location`, giving each a unique
module name (``iheval_eval_*``) so there is no collision.

IFEval is handled differently because the upstream
``rule_following/evaluate/instructions_registry.py`` imports its sibling
``instructions.py`` with the bare ``import instructions`` statement,
which only works when the IFEval evaluate/ directory is on
:data:`sys.path`. We add it once at module-import time. The two upstream
module names (``instructions``, ``instructions_registry``) are unique
to IFEval; they do not shadow anything in this project's ``src/`` tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

# Project root is four parents above this file (src/evaluation/external/iheval/).
_VENDOR_SRC = Path(__file__).resolve().parents[4] / "vendor" / "iheval" / "src"
_IFEVAL_EVAL_DIR = _VENDOR_SRC / "rule_following" / "evaluate"


def _load_module_from_file(unique_name: str, relpath: Path) -> Any:
    """Load a Python file as a uniquely-named module via importlib."""
    file_path = _VENDOR_SRC / relpath
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    if spec is None or spec.loader is None:
        msg = (
            f"Could not load {file_path} — is the IHEval submodule "
            f"initialised? Run `git submodule update --init vendor/iheval`."
        )
        raise ImportError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Standalone scorer modules — load lazily so the import error message is
# helpful when the submodule isn't checked out.
_eval_translation_mod: Any = None
_eval_tensortrust_mod: Any = None
_eval_lang_detect_mod: Any = None
_eval_verb_extract_mod: Any = None
_ifeval_registry_mod: Any = None


def _ensure_modules_loaded() -> None:
    global _eval_translation_mod, _eval_tensortrust_mod
    global _eval_lang_detect_mod, _eval_verb_extract_mod
    global _ifeval_registry_mod

    if _eval_translation_mod is None:
        _eval_translation_mod = _load_module_from_file(
            "iheval_eval_translation",
            Path("task_execution/evaluate/eval_translation.py"),
        )
    if _eval_tensortrust_mod is None:
        _eval_tensortrust_mod = _load_module_from_file(
            "iheval_eval_tensortrust",
            Path("safety/evaluate/eval_tensortrust.py"),
        )
    if _eval_lang_detect_mod is None:
        _eval_lang_detect_mod = _load_module_from_file(
            "iheval_eval_lang_detect",
            Path("task_execution/evaluate/eval_lang_detect.py"),
        )
    if _eval_verb_extract_mod is None:
        _eval_verb_extract_mod = _load_module_from_file(
            "iheval_eval_verb_extract",
            Path("task_execution/evaluate/eval_verb_extract.py"),
        )
    if _ifeval_registry_mod is None:
        # IFEval's instructions_registry uses bare `import instructions`,
        # so the evaluate/ directory must be on sys.path when it loads.
        ifeval_dir = str(_IFEVAL_EVAL_DIR)
        if ifeval_dir not in sys.path:
            sys.path.insert(0, ifeval_dir)
        # Use the standard import machinery — the registry pulls in its
        # `instructions` sibling, and both names are unique to IFEval.
        import importlib

        _ifeval_registry_mod = importlib.import_module("instructions_registry")


def _wrap_translation(answer: Any, response: str) -> dict:
    raw = _eval_translation_mod.eval_translation(answer, response)
    return {"score": float(raw), "details": {}}


def _wrap_verb_extract(answer: Any, response: str) -> dict:
    raw = _eval_verb_extract_mod.eval_verb_extract(answer, response)
    return {"score": float(raw), "details": {}}


def _wrap_lang_detect(answer: Any, response: str) -> dict:
    raw = _eval_lang_detect_mod.eval_lang_detect(answer, response)
    return {"score": float(bool(raw)), "details": {}}


def _wrap_tensortrust(answer: Any, response: str) -> dict:
    raw = _eval_tensortrust_mod.eval_tensortrust(answer, response)
    return {"score": float(bool(raw)), "details": {}}


def _wrap_ifeval(answer: Any, response: str) -> dict:
    """Per-record IFEval scoring.

    Each IFEval row carries ``instruction_id_list`` and ``kwargs``; the
    response is scored against each instruction's ``check_following``,
    and the per-record score is the fraction of instructions satisfied
    (the IFEval ``prompt_level_loose_acc`` style).
    """
    inst_ids = answer.get("instruction_id_list") or []
    kwargs_list = answer.get("kwargs") or [{}] * len(inst_ids)
    if not inst_ids:
        return {
            "score": 1.0,
            "details": {"per_instruction": [], "n_instructions": 0},
        }
    registry = _ifeval_registry_mod.INSTRUCTION_DICT
    per_instruction: list[bool] = []
    for inst_id, kw in zip(inst_ids, kwargs_list):
        cls = registry.get(inst_id)
        if cls is None:
            per_instruction.append(False)
            continue
        try:
            inst = cls(inst_id)
            inst.build_description(**(kw or {}))
            per_instruction.append(bool(inst.check_following(response)))
        except Exception:  # noqa: BLE001 — IFEval's checkers raise on edge inputs
            per_instruction.append(False)
    score = sum(per_instruction) / len(per_instruction)
    return {
        "score": float(score),
        "details": {
            "per_instruction": per_instruction,
            "n_instructions": len(per_instruction),
        },
    }


def score(task: str, answer: Any, response: str) -> dict:
    """Score one (answer, response) pair for the named task.

    Returns a uniform ``{"score": float, "details": dict}`` envelope.

    Raises:
        KeyError: If ``task`` is not one of the seven in-scope IHEval tasks.
    """
    _ensure_modules_loaded()
    dispatch = {
        "translation":            _wrap_translation,
        "verb-extract":           _wrap_verb_extract,
        "lang-detect":            _wrap_lang_detect,
        "user-prompt-hijack":     _wrap_tensortrust,
        "system-prompt-extract":  _wrap_tensortrust,
        "single-turn":            _wrap_ifeval,
        "multi-turn":             _wrap_ifeval,
    }
    if task not in dispatch:
        msg = (
            f"No IHEval scorer wired for task={task!r}. In-scope tasks: "
            f"{sorted(dispatch)}."
        )
        raise KeyError(msg)
    return dispatch[task](answer, response)
