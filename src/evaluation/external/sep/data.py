"""SEP CSV loader.

The vendored subsample CSV at ``data/external/sep/sep_subsample.csv`` is
produced by ``bin/build_sep_subsample.py`` and is the source of truth
for runtime SEP evaluation. Schema is fixed by this loader and does
not depend on the upstream HuggingFace field names.
"""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SEPRecord:
    """One row of the SEP subsample."""

    id: int
    domain: str
    instruction: str
    data_with_witness: str
    witness: str
    probe_type: str
    source_index: int


def load_sep_csv(csv_path: str | Path) -> list[SEPRecord]:
    """Load and validate the SEP subsample CSV.

    Args:
        csv_path: Path to ``sep_subsample.csv``.

    Returns:
        List of :class:`SEPRecord`, one per CSV row.

    Raises:
        ValueError: If a row has empty ``witness``, ``instruction``,
            or ``data_with_witness``.
    """
    records: list[SEPRecord] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            instruction = row["instruction"].strip()
            data_with_witness = row["data_with_witness"].strip()
            witness = row["witness"].strip()
            row_id = row.get("id")
            if not witness:
                msg = f"Empty witness in row id={row_id}."
                raise ValueError(msg)
            if not instruction:
                msg = f"Empty instruction in row id={row_id}."
                raise ValueError(msg)
            if not data_with_witness:
                msg = f"Empty data_with_witness in row id={row_id}."
                raise ValueError(msg)
            records.append(
                SEPRecord(
                    id=int(row["id"]),
                    domain=row["domain"],
                    instruction=row["instruction"],
                    data_with_witness=row["data_with_witness"],
                    witness=row["witness"],
                    probe_type=row["probe_type"],
                    source_index=int(row["source_index"]),
                ),
            )
    return records
