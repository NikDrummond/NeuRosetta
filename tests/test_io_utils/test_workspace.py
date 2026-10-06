"""Tests for Workspace API and ``.nrw`` I/O."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import neurosetta as nr
from neurosetta.api import Workspace
from neurosetta.io.workspace_errors import (
    WorkspaceFormatError,
    WorkspaceIntegrityError,
    WorkspaceVersionError,
)
from neurosetta.io.workspace_utils import WORKSPACE_FORMAT, WORKSPACE_SCHEMA_VERSION
from neurosetta.testing import make_synthetic_forest, make_synthetic_tree


@pytest.fixture
def forest():
    return make_synthetic_forest(3, n=12, seed=7, units="nm")


@pytest.fixture
def workspace(forest):
    return Workspace(
        forest,
        name="T4/T5 analysis",
        description="Morphology analysis",
        metadata={"dataset": "synthetic", "note": "unit-test"},
    )


def test_workspace_basics(workspace, forest):
    assert workspace.forest is forest
    assert workspace.name == "T4/T5 analysis"
    assert workspace.description == "Morphology analysis"
    assert workspace.metadata["dataset"] == "synthetic"
    assert "trees=3" in repr(workspace)
    assert "selections=0" in repr(workspace)


def test_workspace_requires_forest():
    with pytest.raises(TypeError, match="Forest"):
        Workspace([1, 2, 3])  # type: ignore[arg-type]


def test_selections(workspace, forest):
    workspace.add_selection("first_two", forest[:2])
    workspace.add_selection("by_ids", [1, 3])
    workspace.add_selection("empty", [])

    assert workspace.list_selections() == ["first_two", "by_ids", "empty"]
    sel = workspace.get_selection("first_two")
    assert isinstance(sel, nr.Forest)
    assert sel.ids() == [1, 2]
    assert workspace.get_selection("empty").ids() == []

    with pytest.raises(ValueError, match="already exists"):
        workspace.add_selection("first_two", forest[:1])

    workspace.add_selection("first_two", forest[:1], overwrite=True)
    assert workspace.get_selection("first_two").ids() == [1]

    with pytest.raises(KeyError, match="not present"):
        workspace.add_selection("bad", [999])

    workspace.remove_selection("by_ids")
    assert "by_ids" not in workspace.list_selections()


def test_results(workspace):
    workspace.add_result("params", {"method": "pca", "n": 3})
    workspace.add_result("arr", np.arange(6).reshape(2, 3))
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    workspace.add_result("table", df)

    assert workspace.list_results() == ["params", "arr", "table"]
    assert workspace.get_result("params")["method"] == "pca"

    with pytest.raises(ValueError, match="already exists"):
        workspace.add_result("params", {"other": True})

    workspace.add_result("params", {"other": True}, overwrite=True)
    assert workspace.get_result("params") == {"other": True}

    workspace.remove_result("arr")
    assert "arr" not in workspace.list_results()


def test_round_trip(workspace, forest, tmp_path):
    workspace.add_selection("subset", forest[:2])
    workspace.add_selection("last", [3])
    coords = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    workspace.add_result("coords", coords)
    workspace.add_result("params", {"alpha": 0.5, "nested": [1, {"k": "v"}]})
    df = pd.DataFrame({"n_nodes": [10, 20], "label": ["a", "b"]}, index=[10, 20])
    workspace.add_result("metrics", df)

    # User metadata on a tree should survive via .nr
    forest[0].set_meta("cell_type", "T4a")

    path = nr.save_workspace(workspace, tmp_path / "analysis")
    assert path.suffix == ".nrw"
    assert path.is_file()

    loaded = nr.load_workspace(path)
    assert isinstance(loaded, Workspace)
    assert loaded.name == workspace.name
    assert loaded.description == workspace.description
    assert loaded.metadata == workspace.metadata
    assert len(loaded.forest) == len(forest)
    assert loaded.forest.ids() == forest.ids()
    assert [t.name for t in loaded.forest] == [t.name for t in forest]
    assert loaded.forest[0].get_meta("cell_type") == "T4a"
    assert loaded.forest[0].count_nodes() == forest[0].count_nodes()

    assert loaded.list_selections() == ["subset", "last"]
    assert loaded.get_selection("subset").ids() == [1, 2]
    assert loaded.get_selection("last").ids() == [3]

    np.testing.assert_array_equal(loaded.get_result("coords"), coords)
    assert loaded.get_result("params") == {"alpha": 0.5, "nested": [1, {"k": "v"}]}
    assert_frame_equal(loaded.get_result("metrics"), df)


def test_numpy_object_dtype_rejected(workspace, tmp_path):
    workspace.add_result("bad", np.array([{"a": 1}], dtype=object))
    with pytest.raises(TypeError, match="bad"):
        nr.save_workspace(workspace, tmp_path / "bad.nrw")


def test_unsupported_result_rejected(workspace, tmp_path):
    workspace.add_result("obj", object())
    with pytest.raises(TypeError, match="obj"):
        nr.save_workspace(workspace, tmp_path / "bad.nrw")
    assert not (tmp_path / "bad.nrw").exists()


def test_non_json_metadata_rejected(forest, tmp_path):
    ws = Workspace(forest, metadata={"bad": object()})
    with pytest.raises(TypeError, match="metadata"):
        nr.save_workspace(ws, tmp_path / "bad.nrw")


def test_empty_forest(tmp_path):
    ws = Workspace(nr.Forest([]), name="empty")
    path = nr.save_workspace(ws, tmp_path / "empty.nrw")
    loaded = nr.load_workspace(path)
    assert len(loaded.forest) == 0
    assert loaded.name == "empty"


def test_inspect_does_not_require_trees(workspace, tmp_path):
    workspace.add_selection("s", workspace.forest[:1])
    workspace.add_result("x", [1, 2, 3])
    path = nr.save_workspace(workspace, tmp_path / "i.nrw")

    info = nr.inspect_workspace(path)
    assert info["name"] == workspace.name
    assert info["description"] == workspace.description
    assert info["schema_version"] == WORKSPACE_SCHEMA_VERSION
    assert info["neurosetta_version"] == nr.__version__
    assert info["tree_count"] == 3
    assert info["selections"] == {"s": 1}
    assert info["results"] == {"x": "json"}
    assert info["metadata"]["dataset"] == "synthetic"


def test_file_path_not_temp(workspace, tmp_path):
    path = nr.save_workspace(workspace, tmp_path / "src.nrw")
    loaded = nr.load_workspace(path)
    for tree in loaded.forest:
        fp = tree.metadata["file_path"]
        assert fp == str(path.resolve())
        assert "nrw-load-" not in fp


def test_invalid_archives(tmp_path, workspace):
    not_zip = tmp_path / "x.nrw"
    not_zip.write_text("not a zip", encoding="utf-8")
    with pytest.raises(WorkspaceFormatError, match="ZIP"):
        nr.load_workspace(not_zip)

    good = nr.save_workspace(workspace, tmp_path / "good.nrw")

    # Missing manifest
    missing_manifest = tmp_path / "no_manifest.nrw"
    with zipfile.ZipFile(good, "r") as src, zipfile.ZipFile(missing_manifest, "w") as dst:
        for info in src.infolist():
            if info.filename != "manifest.json":
                dst.writestr(info, src.read(info.filename))
    with pytest.raises(WorkspaceFormatError, match="manifest"):
        nr.load_workspace(missing_manifest)

    # Wrong format
    bad_fmt = tmp_path / "bad_fmt.nrw"
    with zipfile.ZipFile(good, "r") as src, zipfile.ZipFile(bad_fmt, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "manifest.json":
                manifest = json.loads(data)
                manifest["format"] = "not-a-workspace"
                data = json.dumps(manifest).encode()
            dst.writestr(info, data)
    with pytest.raises(WorkspaceFormatError, match="format"):
        nr.load_workspace(bad_fmt)

    # Unsupported future schema
    future = tmp_path / "future.nrw"
    with zipfile.ZipFile(good, "r") as src, zipfile.ZipFile(future, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "manifest.json":
                manifest = json.loads(data)
                manifest["schema_version"] = WORKSPACE_SCHEMA_VERSION + 99
                data = json.dumps(manifest).encode()
            dst.writestr(info, data)
    with pytest.raises(WorkspaceVersionError, match="newer"):
        nr.load_workspace(future)

    # Missing tree member
    missing_tree = tmp_path / "missing_tree.nrw"
    with zipfile.ZipFile(good, "r") as src, zipfile.ZipFile(missing_tree, "w") as dst:
        for info in src.infolist():
            if info.filename.startswith("forest/"):
                continue
            dst.writestr(info, src.read(info.filename))
    with pytest.raises(WorkspaceIntegrityError, match="missing"):
        nr.load_workspace(missing_tree)


def test_missing_artifact(workspace, tmp_path):
    workspace.add_result("x", {"a": 1})
    good = nr.save_workspace(workspace, tmp_path / "with_art.nrw")
    missing_art = tmp_path / "missing_art.nrw"
    with zipfile.ZipFile(good, "r") as src, zipfile.ZipFile(missing_art, "w") as dst:
        for info in src.infolist():
            if info.filename.startswith("artifacts/"):
                continue
            dst.writestr(info, src.read(info.filename))
    with pytest.raises(WorkspaceIntegrityError, match="artifact"):
        nr.load_workspace(missing_art)


def test_zip_traversal_rejected(tmp_path, workspace):
    good = nr.save_workspace(workspace, tmp_path / "base.nrw")
    evil = tmp_path / "evil.nrw"
    with zipfile.ZipFile(good, "r") as src, zipfile.ZipFile(evil, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "manifest.json":
                manifest = json.loads(data)
                manifest["forest"]["members"][0]["path"] = "../escape.nr"
                data = json.dumps(manifest).encode()
            dst.writestr(info, data)
        dst.writestr("../escape.nr", b"nope")
    with pytest.raises(WorkspaceFormatError, match="unsafe"):
        nr.load_workspace(evil)


def test_hash_mismatch(tmp_path, workspace):
    good = nr.save_workspace(workspace, tmp_path / "hashed.nrw")
    corrupt = tmp_path / "corrupt.nrw"
    with zipfile.ZipFile(good, "r") as src, zipfile.ZipFile(corrupt, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename.startswith("forest/"):
                data = data + b"\x00"
            dst.writestr(info, data)
    with pytest.raises(WorkspaceIntegrityError, match="SHA-256"):
        nr.load_workspace(corrupt)


def test_atomic_replace_leaves_existing(tmp_path, workspace, monkeypatch):
    path = nr.save_workspace(workspace, tmp_path / "keep.nrw")
    original = path.read_bytes()

    ws2 = Workspace(make_synthetic_forest(2, n=5, seed=99), name="replacement")

    def boom(*_args, **_kwargs):
        raise RuntimeError("forced failure before replace")

    monkeypatch.setattr(zipfile.ZipFile, "writestr", boom)
    with pytest.raises(RuntimeError, match="forced failure"):
        nr.save_workspace(ws2, path)

    assert path.read_bytes() == original
    # Temp leftovers cleaned up
    leftovers = list(tmp_path.glob(".keep.*.nrw.tmp")) + list(tmp_path.glob("*.nrw.tmp"))
    assert leftovers == []


def test_duplicate_tree_names_ok(tmp_path):
    t1 = make_synthetic_tree(8, tree_id=10, seed=1)
    t2 = make_synthetic_tree(8, tree_id=11, seed=2)
    t1.name = "same/name\\weird"
    t2.name = "same/name\\weird"
    forest = nr.Forest([t1, t2])
    ws = Workspace(forest, name="dup")
    ws.add_selection("both", forest)
    path = nr.save_workspace(ws, tmp_path / "dup.nrw")
    loaded = nr.load_workspace(path)
    assert loaded.forest.ids() == [10, 11]
    assert [t.name for t in loaded.forest] == ["same/name\\weird", "same/name\\weird"]


def test_public_exports():
    assert nr.Workspace is Workspace
    assert callable(nr.save_workspace)
    assert callable(nr.load_workspace)
    assert callable(nr.inspect_workspace)
    assert issubclass(nr.WorkspaceFormatError, nr.WorkspaceError)


def test_manifest_constants(workspace, tmp_path):
    path = nr.save_workspace(workspace, tmp_path / "c.nrw")
    with zipfile.ZipFile(path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert manifest["format"] == WORKSPACE_FORMAT
    assert manifest["schema_version"] == WORKSPACE_SCHEMA_VERSION
    assert manifest["forest"]["members"][0]["ref"] == "t000000"
    assert "sha256" in manifest["forest"]["members"][0]
