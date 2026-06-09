"""Shared output-dir / resume helpers for the external eval CLIs.

All three benchmarks (IHEval, SEP, XSTest) follow the same on-disk
layout::

    <external_runs>/<bench>/<key>/run_<UTC-ts>/
        responses*.jsonl       (incrementally appended, resumable)
        scoring*.jsonl         (incrementally appended, resumable)
        metrics.json           (written once, at successful completion)
        run_args.json          (written once, on first invocation)

A run is "complete" iff ``metrics.json`` exists. Resumable runs use the
``run_args.json`` sidecar to fail fast if a follow-up invocation would
combine partial outputs from one parameter set with new work from
another (e.g. resuming a run with a different ``--limit``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

ResolveMode = Literal["explicit", "fresh", "resume_partial", "resume_complete"]

_RUN_PREFIX = "run_"
_METRICS_FILE = "metrics.json"
_RUN_ARGS_FILE = "run_args.json"


class ResumeArgsMismatch(RuntimeError):
    """Raised when --resume targets a dir whose run_args.json disagrees."""


def latest_run_dir(base_dir: Path) -> Path | None:
    """Return the lexicographically newest ``run_*`` subdir, or None.

    UTC-timestamp directory names sort lexicographically the same as
    chronologically, so a plain ``sorted(reverse=True)[0]`` is correct.
    """
    if not base_dir.exists():
        return None
    candidates = sorted(
        (
            p for p in base_dir.iterdir()
            if p.is_dir() and p.name.startswith(_RUN_PREFIX)
        ),
        reverse=True,
    )
    return candidates[0] if candidates else None


def is_complete(run_dir: Path) -> bool:
    """A run is complete iff ``metrics.json`` exists at its root."""
    return (run_dir / _METRICS_FILE).exists()


def resolve_output_dir(
    *,
    output_dir_arg: str | None,
    resume: bool,
    base_dir: Path,
    timestamp: str,
) -> tuple[Path, ResolveMode]:
    """Decide the run directory + mode for a CLI invocation.

    Args:
        output_dir_arg: The literal ``--output-dir`` value if the user
            passed one; otherwise ``None``. When set, it short-circuits
            (the user is in full control of the path).
        resume: ``--resume`` flag. When ``False``, always create a new
            timestamped subdir of ``base_dir``.
        base_dir: Canonical per-(model, format[, mapping]) parent that
            holds the ``run_<ts>/`` siblings, e.g.
            ``evaluation/external/iheval/gw_dpo__delimited/``.
        timestamp: UTC string in the agreed ``YYYYMMDD_HHMMSS`` format,
            used as the ``run_<ts>`` suffix when starting fresh.

    Returns:
        ``(output_dir, mode)``. Mode ``resume_complete`` signals the CLI
        to log + exit 0 without loading the model. Mode ``resume_partial``
        means reuse the dir; the runner's per-step caches will skip
        already-finished work. ``fresh`` means the dir does not yet
        exist; the runner will ``mkdir`` it. ``explicit`` defers all
        decisions to the user.
    """
    if output_dir_arg is not None:
        return Path(output_dir_arg), "explicit"
    if resume:
        latest = latest_run_dir(base_dir)
        if latest is not None:
            return latest, ("resume_complete" if is_complete(latest)
                            else "resume_partial")
    return base_dir / f"{_RUN_PREFIX}{timestamp}", "fresh"


def save_run_args(run_dir: Path, args: dict) -> None:
    """Persist a JSON sidecar of the args that defined this run.

    Idempotent: writing the same dict twice is a no-op. Writing a
    *different* dict overwrites silently — the caller must validate
    first via :func:`validate_run_args`.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / _RUN_ARGS_FILE).write_text(json.dumps(args, indent=2, sort_keys=True))


def validate_run_args(run_dir: Path, args: dict) -> None:
    """Raise :class:`ResumeArgsMismatch` if a sidecar disagrees with ``args``.

    Missing sidecar = legacy partial run; pass silently. The caller
    decides whether to write a fresh sidecar after validation.
    """
    sidecar = run_dir / _RUN_ARGS_FILE
    if not sidecar.exists():
        return
    on_disk = json.loads(sidecar.read_text())
    diffs = []
    keys = set(on_disk) | set(args)
    for key in sorted(keys):
        prior = on_disk.get(key)
        current = args.get(key)
        if prior != current:
            diffs.append(f"{key}: prior={prior!r} new={current!r}")
    if diffs:
        msg = (
            f"--resume target {run_dir} was created with different args.\n  "
            + "\n  ".join(diffs)
            + "\nRe-run without --resume (or with a fresh --output-dir) to "
            "start over with the new args."
        )
        raise ResumeArgsMismatch(msg)
