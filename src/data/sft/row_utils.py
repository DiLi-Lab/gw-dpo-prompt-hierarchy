"""Shared row accessor helpers for base dataset rows.

Handles field name differences between Alpaca (output, input) and
Dolly (response, context) schemas, providing a unified interface
for all SFT category builders.
"""


def get_output(row: dict[str, str]) -> str:
    """Get the output/response field from a base dataset row.

    Prefers Alpaca's ``output`` over Dolly's ``response``.
    Returns an empty string when neither field is present.
    """
    return row.get("output") or row.get("response", "")


def get_input(row: dict[str, str]) -> str:
    """Get the input/context field from a base dataset row.

    Prefers Alpaca's ``input`` over Dolly's ``context``.
    Returns an empty string when neither field is present.
    """
    return row.get("input") or row.get("context", "")
