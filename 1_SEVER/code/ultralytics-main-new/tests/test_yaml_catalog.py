"""Integrity tests for the organized citrus model YAML catalog."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
YAML_ROOT = ROOT / "0_orange_yaml"


def test_model_index_matches_every_yaml_exactly():
    """The central index must have one and only one row for every model YAML."""
    actual = {path.relative_to(YAML_ROOT).as_posix() for path in YAML_ROOT.rglob("*.yaml")}
    with (YAML_ROOT / "MODEL_INDEX.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = [row["yaml"] for row in rows]
    assert len(actual) == 234
    assert len(indexed) == len(set(indexed))
    assert set(indexed) == actual


def test_each_model_series_has_a_readme_and_root_has_no_model_yaml():
    """Every series must be self-describing and model YAMLs must not leak into the catalog root."""
    assert not list(YAML_ROOT.glob("*.yaml"))
    model_series = {path.parent for path in YAML_ROOT.rglob("*.yaml")}
    for directory in model_series:
        assert (directory / "README.md").is_file(), f"Missing README: {directory}"
