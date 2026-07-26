"""Tests for SWC header metadata parsing and formatting."""

import json
import warnings
from pathlib import Path

import pytest

from NeuRosetta.api import Tree
from NeuRosetta.io.swc_meta import (
    default_swc_header,
    parse_swc_header,
    units_from_swc_header,
    warn_if_export_dimensionless,
)
from NeuRosetta.io.swc_utils import export_swc, import_swc


def test_parse_swc_header_meta_json(tmp_path):
    swc = tmp_path / "1.swc"
    swc.write_text('# Meta: {"units": "nm"}\n1 1 0.0 0.0 0.0 0.5 -1\n')

    meta = parse_swc_header(swc)
    assert meta["units"] == "nm"
    assert units_from_swc_header(meta) == "nm"


def test_parse_swc_header_units_comment(tmp_path):
    swc = tmp_path / "1.swc"
    swc.write_text("# units: micrometer\n1 1 0.0 0.0 0.0 0.5 -1\n")

    meta = parse_swc_header(swc)
    assert units_from_swc_header(meta) == "micrometer"


def test_parse_swc_header_navis_dimensionless(tmp_path):
    swc = tmp_path / "1.swc"
    swc.write_text('# Meta: {"units": "1 dimensionless"}\n1 1 0.0 0.0 0.0 0.5 -1\n')

    meta = parse_swc_header(swc)
    assert units_from_swc_header(meta) == "1 dimensionless"


def test_import_swc_reads_header_units(tmp_path):
    swc = tmp_path / "1.swc"
    swc.write_text('# Meta: {"units": "nm"}\n1 1 0.0 0.0 0.0 0.5 -1\n')

    tree = import_swc(swc)
    assert tree.metadata["units"] == "nanometer"


def test_import_swc_set_units_overrides_header(tmp_path):
    swc = tmp_path / "1.swc"
    swc.write_text('# Meta: {"units": "nm"}\n1 1 0.0 0.0 0.0 0.5 -1\n')

    tree = import_swc(swc, set_units="micron")
    assert tree.metadata["units"] == "micron"


def test_export_swc_writes_meta_units(tmp_path, simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    path = tmp_path / "1.swc"

    export_swc(tree, path)
    text = path.read_text()
    meta = parse_swc_header(path)

    assert '"units"' in text
    assert meta["units"] == "nanometer"
    assert meta["id"] == "1"


def test_export_import_swc_units_round_trip(tmp_path, simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    path = tmp_path / "1.swc"

    export_swc(tree, path)
    loaded = import_swc(path)

    assert loaded.metadata["units"] == "nanometer"


def test_export_swc_warns_dimensionless(tmp_path, simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    path = tmp_path / "1.swc"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_swc(tree, path)

    assert any("dimensionless" in str(w.message).lower() for w in caught)


def test_default_swc_header_includes_units(simple_tree):
    tree = Tree(ID=7, metadata={"units": "nm"}, graph=simple_tree)
    header = default_swc_header(tree)
    payload = json.loads(header.splitlines()[1].split(":", 1)[1].strip())

    assert payload["id"] == "7"
    assert payload["units"] == "nanometer"


def test_warn_if_export_dimensionless(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_export_dimensionless(tree)

    assert caught
    assert "dimensionless" in str(caught[0].message).lower()


def test_parse_docs_sample_header():
    sample = Path(__file__).resolve().parents[2] / "docs" / "data" / "1.swc"
    if not sample.exists():
        pytest.skip("sample SWC not available")

    meta = parse_swc_header(sample)
    assert units_from_swc_header(meta) == "1 dimensionless"
