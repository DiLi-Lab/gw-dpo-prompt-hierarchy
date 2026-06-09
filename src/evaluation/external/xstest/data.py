"""XSTest CSV loader.

The upstream dataset is a single CSV with columns
``id, prompt, type, label, focus, note``. This loader reads it into
typed :class:`XSTestRecord` instances and validates the label set.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

XSTestLabel = Literal["safe", "unsafe"]
_VALID_LABELS: frozenset[str] = frozenset({"safe", "unsafe"})


@dataclass(frozen=True)
class XSTestRecord:
    """One row of the XSTest dataset."""

    id: int
    prompt: str
    type: str
    label: XSTestLabel
    focus: str
    note: str


def load_xstest_csv(csv_path: str | Path) -> list[XSTestRecord]:
    """Load and validate the XSTest prompts CSV.

    Args:
        csv_path: Path to ``xstest_prompts.csv``.

    Returns:
        List of :class:`XSTestRecord`, one per CSV row.

    Raises:
        ValueError: If a row contains a ``label`` value outside
            ``{"safe", "unsafe"}``.
    """
    records: list[XSTestRecord] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if label not in _VALID_LABELS:
                msg = (
                    f"Unexpected label {label!r} in row id={row.get('id')}."
                    " Expected 'safe' or 'unsafe'."
                )
                raise ValueError(msg)
            records.append(
                XSTestRecord(
                    id=int(row["id"]),
                    prompt=row["prompt"],
                    type=row["type"],
                    label=label,  # type: ignore[arg-type]
                    focus=row.get("focus", ""),
                    note=row.get("note", ""),
                ),
            )
    return records
