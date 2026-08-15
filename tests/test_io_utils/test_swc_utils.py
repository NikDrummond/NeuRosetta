# tests/test_io_utils/test_swc_utils.py
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from numpy import array, array_equal, ones_like, sort
from pandas import DataFrame

from neurosetta.api import Tree
from neurosetta.io.io_utils import _check_swc_columns, _table_from_swc
from neurosetta.io.swc_utils import export_swc, import_swc


def _props_by_id(graph, prop):
    ids = graph.vp["ids"].a
    return {int(ids[v]): graph.vp[prop].a[v] for v in graph.get_vertices()}


def _parent_map(graph):
    ids = graph.vp["ids"].a
    parents = {int(ids[v]): -1 for v in graph.get_vertices()}
    for source, target in graph.get_edges():
        parents[int(ids[target])] = int(ids[source])
    return parents


def _coords_by_id(graph):
    ids = graph.vp["ids"].a
    return {
        int(ids[v]): (
            graph.vp["x"].a[v],
            graph.vp["y"].a[v],
            graph.vp["z"].a[v],
        )
        for v in graph.get_vertices()
    }


def test_check_swc_columns_valid():
    df = DataFrame(
        {
            "node_id": [1, 2],
            "type": [1, 1],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "z": [0.0, 1.0],
            "radius": [0.5, 0.5],
            "parent_id": [-1, 1],
        }
    )
    _check_swc_columns(df)  # Should not raise


def test_check_swc_columns_missing():
    df = DataFrame({"node_id": [1, 2], "type": [1, 1]})
    with pytest.raises(ValueError) as exc_info:
        _check_swc_columns(df)
    assert "Missing required columns" in str(exc_info.value)


def test_table_from_swc(tmp_path):
    # Create a temporary SWC file
    swc_content = "1 1 0.0 0.0 0.0 0.5 -1\n2 1 1.0 1.0 1.0 0.5 1\n"
    swc_file = tmp_path / "test.swc"
    swc_file.write_text(swc_content)

    df = _table_from_swc(str(swc_file))
    assert len(df) == 2
    assert list(df.columns) == ["node_id", "type", "x", "y", "z", "radius", "parent_id"]


def test_swc_read_write(simple_tree):

    # test neuron
    tree = Tree(ID=1, metadata={}, graph=simple_tree)

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "1.swc"

        # Write then read
        with pytest.warns(UserWarning, match="dimensionless"):
            export_swc(tree=tree, fpath=path)
        result = import_swc(path)

    # assert metadata and ID
    assert result.ID == 1
    assert "ID" not in result.metadata
    assert result.metadata["units"] == "dimensionless"
    assert not result.metadata["isReduced"]
    assert result.metadata["file_path"] == str(path)
    assert result.metadata is result.graph.gp["metadata"]
    assert int(result.graph.gp["ID"]) == 1

    # assert graph topology and geometry are preserved in SWC id space
    assert array_equal(sort(tree.graph.vp["ids"].a), sort(result.graph.vp["ids"].a))
    assert _parent_map(tree.graph) == _parent_map(result.graph)
    assert _coords_by_id(tree.graph) == _coords_by_id(result.graph)

    n_types = array([-1, 0, 5, 6, 0, 0, 0, 5, 0, 0, 6, 5, 6, 6, 6, 0, 6])
    ones_like(n_types)

    assert _props_by_id(tree.graph, "node_type") == _props_by_id(result.graph, "node_type")
    assert _props_by_id(tree.graph, "radius") == _props_by_id(result.graph, "radius")


def test_swc_roundtrip_non_contiguous_ids(tmp_path):
    swc_file = tmp_path / "42.swc"
    swc_file.write_text("10 1 0 0 0 1 -1\n11 1 1 0 0 1 10\n12 1 2 0 0 1 11\n20 1 3 0 0 1 10\n")

    original = import_swc(swc_file)
    out_file = tmp_path / "99.swc"
    with pytest.warns(UserWarning, match="dimensionless"):
        export_swc(original, out_file)
    result = import_swc(out_file)

    assert array_equal(original.graph.vp["ids"].a, result.graph.vp["ids"].a)
    assert _parent_map(original.graph) == _parent_map(result.graph)
    assert _coords_by_id(original.graph) == _coords_by_id(result.graph)


def test_swc_roundtrip_reordered_ids(tmp_path):
    swc_file = tmp_path / "42.swc"
    swc_file.write_text("1 1 0 0 0 1 -1\n2 1 1 0 0 1 1\n3 1 2 0 0 1 2\n")

    original = import_swc(swc_file)
    out_file = tmp_path / "99.swc"
    with pytest.warns(UserWarning, match="dimensionless"):
        export_swc(original, out_file)
    result = import_swc(out_file)

    assert array_equal(original.graph.vp["ids"].a, result.graph.vp["ids"].a)
    assert _parent_map(original.graph) == _parent_map(result.graph)
    df = _table_from_swc(str(out_file))
    assert list(df.parent_id) == [-1, 1, 2]
    assert list(df.node_id) == [1, 2, 3]


def test_import_swc_set_units(tmp_path, simple_tree):
    swc_file = tmp_path / "1.swc"
    swc_file.write_text("1 1 0.0 0.0 0.0 0.5 -1\n")

    result = import_swc(swc_file, set_units="nm")

    assert result.metadata["units"] == "nanometer"
