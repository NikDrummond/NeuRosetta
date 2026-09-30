"""Mesh taxonomy: neuron vs neuropil vs AnatomicalFrame."""

from __future__ import annotations

import pytest
from vedo import Sphere

from neurosetta import AnatomicalFrame, Neuropil, Tree_mesh, import_mesh
from neurosetta.core.mesh import (
    MESH_KIND_NEURON,
    MESH_KIND_NEUROPIL,
    _Mesh,
    check_neuron_mesh_owner_id,
    is_neuron_mesh,
    is_neuropil_mesh,
    mesh_kind_of,
)
from neurosetta.utils.metrics.descriptors import _coerce_anatomical_frame


def test_tree_mesh_stamps_neuron_kind():
    m = Tree_mesh(ID=42, metadata={}, mesh=Sphere(r=1.0))
    assert m.mesh_kind == MESH_KIND_NEURON
    assert m.metadata["mesh_kind"] == MESH_KIND_NEURON
    assert is_neuron_mesh(m)
    assert mesh_kind_of(m) == MESH_KIND_NEURON


def test_neuropil_stamps_neuropil_kind():
    n = Neuropil(ID="AL", metadata={}, mesh=Sphere(r=1.0))
    assert n.mesh_kind == MESH_KIND_NEUROPIL
    assert is_neuropil_mesh(n)


def test_import_mesh_kinds(tmp_path):
    path = tmp_path / "x.ply"
    Sphere(r=1.0).write(str(path))
    neuron = import_mesh(path, mesh_type="Neuron")
    neuropil = import_mesh(path, mesh_type="Neuropil")
    assert is_neuron_mesh(neuron)
    assert is_neuropil_mesh(neuropil)


def test_anatomical_frame_rejects_tree_mesh():
    neuron = Tree_mesh(ID="n1", metadata={}, mesh=Sphere(r=1.0))
    with pytest.raises(TypeError, match="neuron mesh"):
        AnatomicalFrame(reference_mesh=neuron)


def test_anatomical_frame_accepts_neuropil_and_generic():
    neuropil = Neuropil(ID="AL", metadata={}, mesh=Sphere(r=1.0))
    generic = _Mesh(ID="g", metadata={}, mesh=Sphere(r=1.0))
    assert AnatomicalFrame(reference_mesh=neuropil).reference_mesh is neuropil
    assert AnatomicalFrame(reference_mesh=generic).reference_mesh is generic


def test_coerce_rejects_tree_mesh():
    neuron = Tree_mesh(ID="n1", metadata={}, mesh=Sphere(r=1.0))
    with pytest.raises(TypeError, match="neuron mesh"):
        _coerce_anatomical_frame(neuron)


def test_check_neuron_mesh_owner_id():
    m = Tree_mesh(ID=7, metadata={}, mesh=Sphere(r=1.0))
    check_neuron_mesh_owner_id(m, 7)
    with pytest.raises(ValueError, match="does not match"):
        check_neuron_mesh_owner_id(m, 8)
    neuropil = Neuropil(ID="AL", metadata={}, mesh=Sphere(r=1.0))
    with pytest.raises(TypeError, match="neuron mesh"):
        check_neuron_mesh_owner_id(neuropil, "AL")
