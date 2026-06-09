"""bin/build_sep_subsample.py — stratification & determinism tests."""

import csv
import importlib.util
import json
from pathlib import Path


def _import_build_module():
    repo_root = Path(__file__).resolve().parents[5]
    module_path = repo_root / "bin" / "build_sep_subsample.py"
    spec = importlib.util.spec_from_file_location(
        "build_sep_subsample", module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_dataset_rows(n: int = 200) -> list[dict]:
    """200-row fake SEP dataset across 4 domains (50 each)."""
    domains = ["qa", "summarization", "code", "translation"]
    rows = []
    for i in range(n):
        domain = domains[i % len(domains)]
        rows.append({
            "instruction": f"Instruction {i}",
            "data_with_witness": f"Data with witness W{i}",
            "witness": f"W{i}",
            "domain": domain,
            "probe_type": "direct",
        })
    return rows


def test_build_subsample_is_byte_deterministic(tmp_path: Path) -> None:
    build = _import_build_module()
    out_csv_a = tmp_path / "sep_a.csv"
    out_csv_b = tmp_path / "sep_b.csv"

    rows = _fake_dataset_rows(200)

    def fake_loader():
        return rows

    build.build_subsample(
        loader=fake_loader,
        out_csv_path=out_csv_a,
        manifest_path=tmp_path / "manifest_a.json",
        seed=42,
        size=40,
        strata_field="domain",
        upstream_revision="test",
    )
    build.build_subsample(
        loader=fake_loader,
        out_csv_path=out_csv_b,
        manifest_path=tmp_path / "manifest_b.json",
        seed=42,
        size=40,
        strata_field="domain",
        upstream_revision="test",
    )

    assert out_csv_a.read_bytes() == out_csv_b.read_bytes()


def test_build_subsample_stratifies_proportionally(tmp_path: Path) -> None:
    build = _import_build_module()
    out_csv = tmp_path / "sep.csv"

    rows = _fake_dataset_rows(200)
    build.build_subsample(
        loader=lambda: rows,
        out_csv_path=out_csv,
        manifest_path=tmp_path / "manifest.json",
        seed=42,
        size=40,
        strata_field="domain",
        upstream_revision="test",
    )

    with out_csv.open() as f:
        reader = csv.DictReader(f)
        sampled = list(reader)
    by_domain: dict[str, int] = {}
    for r in sampled:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
    assert len(sampled) == 40
    # 4 equal-size groups of 50 → exactly 10 per domain at target 40 (no rounding).
    for domain, n in by_domain.items():
        assert n == 10, f"domain {domain} got {n} rows (expected exactly 10)"


def test_build_subsample_writes_manifest(tmp_path: Path) -> None:
    build = _import_build_module()
    rows = _fake_dataset_rows(200)
    out_csv = tmp_path / "sep.csv"
    manifest_path = tmp_path / "manifest.json"

    build.build_subsample(
        loader=lambda: rows,
        out_csv_path=out_csv,
        manifest_path=manifest_path,
        seed=42,
        size=40,
        strata_field="domain",
        upstream_revision="test-rev-abc",
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 42
    assert manifest["total"] == 40
    assert manifest["upstream_revision"] == "test-rev-abc"
    assert "n_per_domain" in manifest
    assert "built_at" in manifest


def test_build_subsample_absorbs_rounding_slack(tmp_path: Path) -> None:
    """Slack-absorption path: when per-group rounding doesn't sum to size,
    the largest group absorbs the difference so the total exactly equals size."""
    build = _import_build_module()

    # Three unequal groups of 33 / 33 / 34, target size 11.
    # round(11 * 33/100) = round(3.63) = 4 for each of the size-33 groups,
    # round(11 * 34/100) = round(3.74) = 4 for the size-34 group,
    # → naive sum is 4 + 4 + 4 = 12, which is 1 over target.
    # Slack absorption pulls -1 from the largest group → 4 + 4 + 3 = 11.
    rows: list[dict] = []
    for i in range(33):
        rows.append({
            "instruction": f"i{i}", "data_with_witness": f"d{i}",
            "witness": f"w{i}", "domain": "alpha", "probe_type": "p",
        })
    for i in range(33):
        rows.append({
            "instruction": f"i{i}", "data_with_witness": f"d{i}",
            "witness": f"w{i}", "domain": "beta", "probe_type": "p",
        })
    for i in range(34):
        rows.append({
            "instruction": f"i{i}", "data_with_witness": f"d{i}",
            "witness": f"w{i}", "domain": "gamma", "probe_type": "p",
        })

    out_csv = tmp_path / "sep.csv"
    build.build_subsample(
        loader=lambda: rows,
        out_csv_path=out_csv,
        manifest_path=tmp_path / "manifest.json",
        seed=42,
        size=11,
        strata_field="domain",
        upstream_revision="test",
    )

    with out_csv.open() as f:
        reader = csv.DictReader(f)
        sampled = list(reader)
    assert len(sampled) == 11
    by_domain: dict[str, int] = {}
    for r in sampled:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
    # gamma is the largest group (34 rows) so it absorbs the -1 slack.
    assert by_domain == {"alpha": 4, "beta": 4, "gamma": 3}
