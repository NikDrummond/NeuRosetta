"""Identity model: name (artifact) vs ID (logical neuron)."""

from __future__ import annotations

from pathlib import Path

import pytest
from vedo import Sphere

from neurosetta.api import Forest, Tree, Tree_mesh
from neurosetta.core.tree_helpers import bind_tree_id, bind_tree_metadata, bind_tree_name
from neurosetta.io import export_mesh, export_swc, import_swc, load, save
from neurosetta.io.io_utils import _base_meta
from neurosetta.ops.tree_graphs.tree_editing import reduce_tree
from neurosetta.ops.tree_graphs.tree_mesh import _mesh_to_payload, _payload_to_mesh


def _tiny_swc(path: Path) -> Path:
    path.write_text("1 1 0 0 0 1 -1\n2 1 1 0 0 1 1\n")
    return path


@pytest.fixture
def simple_named_tree(simple_tree):
    return Tree(ID=1, metadata=_base_meta(), graph=simple_tree, name="cell_A_full")


# --- 1–4: SWC import / construction ---


def test_non_numeric_swc_stem(tmp_path):
    path = _tiny_swc(tmp_path / "cell_A.swc")
    tree = import_swc(path)
    assert tree.name == "cell_A"
    assert tree.ID == "cell_A"


def test_suffixed_numeric_stem_no_int_conversion(tmp_path):
    path = _tiny_swc(tmp_path / "720575940630123456_full.swc")
    tree = import_swc(path)
    assert tree.name == "720575940630123456_full"
    assert tree.ID == "720575940630123456_full"
    assert isinstance(tree.ID, str)


def test_explicit_string_logical_id(tmp_path):
    path = _tiny_swc(tmp_path / "720575940630123456_full.swc")
    tree = import_swc(path, ID="flywire:720575940630123456")
    assert tree.name == "720575940630123456_full"
    assert tree.ID == "flywire:720575940630123456"


def test_leading_zero_id_stays_string(simple_tree):
    tree = Tree(ID="00123", metadata={}, graph=simple_tree)
    assert tree.ID == "00123"
    assert tree.name == "00123"
    assert isinstance(tree.ID, str)


def test_integer_id_still_works(simple_tree):
    tree = Tree(ID=42, metadata={}, graph=simple_tree)
    assert tree.ID == 42
    assert tree.name == "42"


# --- 6–7: copy / reduce ---


def test_copy_preserves_name_and_id(simple_named_tree):
    clone = simple_named_tree.copy()
    assert clone.ID == simple_named_tree.ID == 1
    assert clone.name == simple_named_tree.name == "cell_A_full"
    assert clone.graph is not simple_named_tree.graph


def test_reduce_preserves_logical_id(simple_named_tree):
    simple_named_tree.ID = "abc"
    simple_named_tree.name = "abc_full"
    g = reduce_tree(simple_named_tree, inplace=False)
    reduced = Tree.from_graph(g)
    assert reduced.ID == "abc"
    assert reduced.name == "abc_full"


# --- 8–9: mesh binding ---


def test_mesh_bind_different_names_same_id(simple_tree):
    tree = Tree(ID="abc", metadata={"units": "micron"}, graph=simple_tree, name="abc_full")
    mesh = Tree_mesh(
        ID="abc",
        metadata={"units": "micron"},
        mesh=Sphere(r=1.0),
        name="abc_mesh",
    )
    bound = tree.set_mesh(mesh)
    assert bound.ID == tree.ID == "abc"
    assert bound.name == "abc_mesh"
    assert tree.name == "abc_full"


def test_mesh_bind_incompatible_ids_fail(simple_tree):
    tree = Tree(ID="abc", metadata={"units": "micron"}, graph=simple_tree)
    mesh = Tree_mesh(ID="xyz", metadata={"units": "micron"}, mesh=Sphere(r=1.0))
    with pytest.raises(ValueError, match="does not match"):
        tree.set_mesh(mesh)


# --- 10–12: .nr / mesh payload ---


def test_nr_roundtrip_preserves_name_and_id(simple_tree, tmp_path):
    tree = Tree(
        ID="flywire:123",
        metadata=_base_meta(),
        graph=simple_tree,
        name="123_full",
    )
    out = tmp_path / "round.nr"
    save(tree, out)
    loaded = load(out)
    assert loaded.ID == tree.ID == "flywire:123"
    assert loaded.name == tree.name == "123_full"
    assert isinstance(loaded.ID, str)


def test_legacy_nr_long_id_no_name(simple_tree, tmp_path):
    """Simulate a pre-name .nr: long gp['ID'], no name property."""
    g = simple_tree.copy()
    g.gp["ID"] = g.new_gp("long", 99)
    g.gp["metadata"] = g.new_gp("object", _base_meta())
    # Ensure no name key
    assert "name" not in g.gp
    path = tmp_path / "legacy.nr"
    g.save(str(path), fmt="gt")
    loaded = load(path)
    assert loaded.ID == 99
    assert loaded.name == "99"


def test_mesh_payload_preserves_name_and_id():
    mesh = Tree_mesh(ID="abc", metadata={}, mesh=Sphere(r=1.0), name="abc_mesh")
    payload = _mesh_to_payload(mesh)
    assert payload["ID"] == "abc"
    assert payload["name"] == "abc_mesh"
    rebuilt = _payload_to_mesh(payload)
    assert rebuilt.ID == "abc"
    assert rebuilt.name == "abc_mesh"


# --- 13–16: export filenames ---


def test_swc_export_uses_name(simple_named_tree, tmp_path):
    with pytest.warns(UserWarning, match="dimensionless"):
        export_swc(simple_named_tree, tmp_path)
    assert (tmp_path / "cell_A_full.swc").exists()


def test_nr_save_uses_name(simple_named_tree, tmp_path):
    save(simple_named_tree, tmp_path)
    assert (tmp_path / "cell_A_full.nr").exists()


def test_mesh_export_uses_name(tmp_path):
    mesh = Tree_mesh(ID="abc", metadata={}, mesh=Sphere(r=1.0), name="abc_mesh")
    export_mesh(mesh, tmp_path)
    assert (tmp_path / "abc_mesh.ply").exists()


def test_explicit_export_path_overrides_name(simple_named_tree, tmp_path):
    out = tmp_path / "custom_out.swc"
    with pytest.warns(UserWarning, match="dimensionless"):
        export_swc(simple_named_tree, out)
    assert out.exists()
    assert simple_named_tree.name == "cell_A_full"  # unchanged
    assert not (tmp_path / "cell_A_full.swc").exists()


# --- 17–18: Forest string IDs ---


def test_forest_string_id_ops(simple_tree):
    t1 = Tree(ID="abc", metadata={}, graph=simple_tree, name="abc_full")
    t2 = t1.copy()
    t2.ID = "def"
    t2.name = "def_full"
    forest = Forest([t1, t2])
    assert forest.ids() == ["abc", "def"]
    assert "abc" in forest
    assert forest.by_id("abc") is t1
    forest.remove_id("abc")
    assert forest.ids() == ["def"]


def test_forest_rejects_duplicate_logical_ids(simple_tree):
    t1 = Tree(ID="abc", metadata={}, graph=simple_tree, name="full")
    t2 = t1.copy()
    t2.name = "reduced"  # different artifact, same logical ID
    forest = Forest([t1])
    with pytest.raises(ValueError, match="Duplicate"):
        forest.append(t2)


# --- 19: directory resolver ---


def test_directory_import_id_resolver(tmp_path):
    swc_dir = tmp_path / "swc"
    swc_dir.mkdir()
    _tiny_swc(swc_dir / "neuron_abc_full.swc")
    _tiny_swc(swc_dir / "neuron_def_full.swc")
    forest = import_swc(
        swc_dir,
        id_resolver=lambda p: p.stem.removeprefix("neuron_").removesuffix("_full"),
    )
    assert isinstance(forest, Forest)
    assert set(forest.ids()) == {"abc", "def"}
    assert forest.by_id("abc").name == "neuron_abc_full"


# --- 20: cross-representation ---


def test_cross_representation_attach_and_persist(simple_tree, tmp_path):
    full = Tree(ID="123", metadata={"units": "micron"}, graph=simple_tree, name="123_full")
    reduced_g = reduce_tree(full, inplace=False)
    reduced = Tree.from_graph(reduced_g)
    reduced.name = "123_reduced"
    assert reduced.ID == full.ID == "123"

    mesh = Tree_mesh(
        ID="123",
        metadata={"units": "micron"},
        mesh=Sphere(r=1.0),
        name="123_mesh",
    )
    full.set_mesh(mesh)
    assert full.mesh.name == "123_mesh"
    assert full.mesh.ID == "123"

    out = tmp_path / "full.nr"
    save(full, out)
    loaded = load(out)
    assert loaded.ID == "123"
    assert loaded.name == "123_full"
    assert loaded.mesh is not None
    assert loaded.mesh.ID == "123"
    assert loaded.mesh.name == "123_mesh"


def test_mesh_directory_id_resolver_with_set_meshes(simple_tree, tmp_path):
    tree = Tree(ID="abc", metadata={"units": "micron"}, graph=simple_tree, name="abc_full")
    forest = Forest([tree])
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()
    Sphere(r=1.0).write(str(mesh_dir / "neuron_abc_surface.ply"))
    attached = forest.set_meshes(
        mesh_dir,
        set_units="micron",
        missing="error",
        id_resolver=lambda p: p.stem.split("_")[1],
    )
    assert "abc" in attached
    assert forest.by_id("abc").mesh.name == "neuron_abc_surface"
    assert forest.by_id("abc").mesh.ID == "abc"


def test_bind_tree_id_no_leading_zero_coercion(simple_tree):
    bind_tree_id(simple_tree, "00123")
    bind_tree_metadata(simple_tree, {})
    bind_tree_name(simple_tree, "artifact")
    tree = Tree.from_graph(simple_tree)
    assert tree.ID == "00123"
    assert tree.name == "artifact"
