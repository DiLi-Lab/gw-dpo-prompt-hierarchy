"""IHEval data loader.

Walks the upstream ``benchmark/<category>/<task>/<setting>/<sub>/input_data.json``
tree and emits typed :class:`IHEvalRecord` instances. Task ↔ category
mapping is encoded inline.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# Maps a task name to its enclosing category directory, mirroring the
# upstream layout under benchmark/<category>/<task>/.
TASK_TO_CATEGORY: dict[str, str] = {
    "single-turn":            "rule-following",
    "multi-turn":             "rule-following",
    "verb-extract":           "task-execution",
    "translation":            "task-execution",
    "lang-detect":            "task-execution",
    "user-prompt-hijack":     "safety",
    "system-prompt-extract":  "safety",
    # tool-use tasks deliberately omitted — see design doc §2.
}


@dataclass(frozen=True)
class IHEvalRecord:
    """One IHEval input row enriched with provenance."""

    task: str
    setting: str        # aligned | conflict | reference
    sub: str            # the strictness sub-folder (e.g. "default", "strong_defense")
    id: str | int
    system: str | None
    instruction: str
    conversation_history: list[str] | None
    tool: dict | None
    answer: Any

    @property
    def uid(self) -> str:
        """Stable id across the whole benchmark.

        IHEval row ids only need to be unique within an input_data.json, so
        the uid combines task / setting / sub / id to disambiguate."""
        return f"{self.task}::{self.setting}::{self.sub}::{self.id}"


def _iter_subs(task_root: Path, setting: str) -> Iterator[Path]:
    setting_root = task_root / setting
    if not setting_root.exists():
        return
    for sub_dir in sorted(setting_root.iterdir()):
        if sub_dir.is_dir() and (sub_dir / "input_data.json").exists():
            yield sub_dir


def iter_iheval_records(
    benchmark_root: Path,
    tasks: tuple[str, ...],
    settings: tuple[str, ...],
) -> Iterator[IHEvalRecord]:
    """Yield IHEval records under the chosen tasks × settings.

    Args:
        benchmark_root: Path to ``vendor/iheval/benchmark/``.
        tasks: Task names (must be keys of :data:`TASK_TO_CATEGORY`).
        settings: ``aligned`` / ``conflict`` / ``reference``.
    """
    for task in tasks:
        if task not in TASK_TO_CATEGORY:
            continue
        category = TASK_TO_CATEGORY[task]
        task_root = benchmark_root / category / task
        if not task_root.exists():
            continue
        for setting in settings:
            for sub_dir in _iter_subs(task_root, setting):
                payload = json.loads((sub_dir / "input_data.json").read_text())
                for row in payload:
                    # Try to convert id to int if possible, otherwise keep as string
                    id_val = row["id"]
                    if isinstance(id_val, str):
                        try:
                            id_val = int(id_val)
                        except ValueError:
                            pass  # Keep as string if conversion fails
                    yield IHEvalRecord(
                        task=task,
                        setting=setting,
                        sub=sub_dir.name,
                        id=id_val,
                        system=row.get("system"),
                        instruction=row["instruction"],
                        conversation_history=row.get("conversation_history"),
                        tool=row.get("tool"),
                        answer=row.get("answer"),
                    )
