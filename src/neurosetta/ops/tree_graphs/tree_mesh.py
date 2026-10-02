"""Tree-level neuron-mesh facet (``tree.mesh``)."""

from __future__ import annotations

from typing import Any

import numpy as np
from vedo import Mesh

from ...core import _Tree
from ...core.mesh import (
    MESH_KIND_NEURON,
    check_neuron_mesh_owner_id,
    is_neuron_mesh,
    mesh_kind_of,
)
from ...core.synapses import owner_ids_compatible
from ...utils.graph_utils import g_has_property

_MESH_GP = "mesh"


def _mesh_to_payload(mesh) -> dict:
    """Serialize a neuron mesh for graph-tool / ``.nr`` persistence."""
    return {
        "ID": mesh.ID,
        "name": mesh.name,
        "metadata": dict(mesh.metadata),
        "vertices": np.asarray(mesh.mesh.vertices, dtype=np.float64).copy(),
        "faces": np.asarray(mesh.mesh.cells, dtype=np.int64).copy(),
    }


def _payload_to_mesh(payload: dict):
    """Rebuild a :class:`~neurosetta.api.Tree_mesh` from a serialized payload."""
    from ...api.tree_mesh_class import Tree_mesh

    return Tree_mesh(
        ID=payload["ID"],
        metadata=dict(payload.get("metadata") or {}),
        mesh=Mesh([payload["vertices"], payload["faces"]]),
        name=payload.get("name"),
    )


def _copy_mesh_value(value: Any) -> Any:
    """Deep-copy a live mesh or payload for tree ``copy()``."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {
            "ID": value["ID"],
            "name": value.get("name"),
            "metadata": dict(value.get("metadata") or {}),
            "vertices": np.asarray(value["vertices"], dtype=np.float64).copy(),
            "faces": np.asarray(value["faces"], dtype=np.int64).copy(),
        }
    # Live Tree_mesh — clone geometry
    return type(value)(
        ID=value.ID,
        metadata=dict(value.metadata),
        mesh=value.mesh.clone() if hasattr(value.mesh, "clone") else value.mesh,
        name=value.name,
    )

def freeze_mesh_for_save(tree: _Tree) -> None:
    """Replace a live mesh gp with a pickleable verts/faces payload (in place)."""
    if not g_has_property(tree.graph, _MESH_GP, "g"):
        return
    val = tree.graph.gp[_MESH_GP]
    if val is None or isinstance(val, dict):
        return
    tree.graph.gp[_MESH_GP] = _mesh_to_payload(val)


def _bind_mesh_gp(tree: _Tree, mesh) -> None:
    g = tree.graph
    if mesh is None:
        if _MESH_GP in g.gp:
            del g.gp[_MESH_GP]
        return
    if _MESH_GP not in g.gp:
        g.gp[_MESH_GP] = g.new_gp("object", mesh)
    else:
        g.gp[_MESH_GP] = mesh


def get_mesh(tree: _Tree):
    """Return the attached neuron mesh, or ``None``.

    After ``.nr`` load the gp may hold a verts/faces payload; this hydrates it
    to a live :class:`~neurosetta.api.Tree_mesh` and rebinds.
    """
    if not g_has_property(tree.graph, _MESH_GP, "g"):
        return None
    val = tree.graph.gp[_MESH_GP]
    if val is None:
        return None
    if isinstance(val, dict):
        mesh = _payload_to_mesh(val)
        _bind_mesh_gp(tree, mesh)
        return mesh
    return val


def has_mesh(tree: _Tree) -> bool:
    """Return True when a neuron mesh facet is attached."""
    return get_mesh(tree) is not None


def clear_mesh(tree: _Tree) -> None:
    """Detach the neuron mesh facet."""
    _bind_mesh_gp(tree, None)


def _coerce_mesh(data: Any, *, tree_id) -> Any:
    """Accept ``Tree_mesh`` or ``vedo.Mesh``; reject neuropil / wrong kinds."""
    from ...api.tree_mesh_class import Tree_mesh
    from ...core.mesh import _Mesh

    if isinstance(data, Tree_mesh):
        return data
    if isinstance(data, _Mesh):
        if not is_neuron_mesh(data):
            raise TypeError(
                f"tree.mesh requires a neuron mesh (mesh_kind={MESH_KIND_NEURON!r}); "
                f"got mesh_kind={mesh_kind_of(data)!r} ({type(data).__name__})"
            )
        # Generic _Mesh stamped neuron — wrap as Tree_mesh
        return Tree_mesh(
            ID=data.ID,
            metadata=dict(data.metadata),
            mesh=data.mesh,
            name=getattr(data, "name", None),
        )
    if isinstance(data, Mesh):
        return Tree_mesh(ID=tree_id, metadata={}, mesh=data)
    raise TypeError(f"Unsupported mesh input; expected Tree_mesh or vedo.Mesh, got {type(data)!r}")


def _prepare_mesh_bind(mesh, tree: _Tree, *, context: str) -> None:
    """ID check, unit check/warn; stamp logical ID (not name) from *tree*."""
    from ..units.mesh_facet_units import (
        check_mesh_tree_units,
        stamp_mesh_units_from_tree,
    )

    if not owner_ids_compatible(mesh.ID, tree.ID):
        raise ValueError(f"{context}: mesh ID {mesh.ID!r} does not match tree.ID={tree.ID!r}")
    # Prefer exact tree ID after compatible numeric/string match.
    # Artifact name is intentionally left untouched.
    if mesh.ID != tree.ID:
        mesh.ID = tree.ID
    check_neuron_mesh_owner_id(mesh, tree.ID)
    check_mesh_tree_units(mesh, tree, context=context)
    stamp_mesh_units_from_tree(mesh, tree)

def set_mesh(
    tree: _Tree,
    data: Any,
    *,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
):
    """Attach (replace) a neuron mesh facet on *tree*.

    Preferred entry point for neuron surfaces — same pattern as
    :func:`~neurosetta.ops.tree_graphs.tree_synapses.set_synapses`.
    :class:`~neurosetta.api.Tree_mesh` remains the I/O / payload type; analyzing
    a standalone mesh as the primary neuron object is soft-deprecated.

    Parameters
    ----------
    tree : _Tree
        Morphology to bind to.
    data : Tree_mesh, vedo.Mesh, or path
        Neuron surface, or a **single-file** path loaded via
        :func:`~neurosetta.import_mesh` (``mesh_type=\"Neuron\"``). Directories
        are rejected — use :func:`~neurosetta.set_meshes` on a Forest.
        ``ID`` must match ``tree.ID`` (numeric coercion allowed). Units must
        agree when both sides declare them.
    set_units, voxel_size, voxel_unit
        Forwarded to :func:`~neurosetta.import_mesh` when *data* is a path.

    Returns
    -------
    Tree_mesh
        The bound mesh (same object when a ``Tree_mesh`` was passed).
    """
    from pathlib import Path

    if isinstance(data, (str, Path)):
        from ...io.mesh_utils import import_mesh

        path = Path(data)
        if path.is_dir():
            raise TypeError(
                "set_mesh() does not accept a directory; use "
                "Forest.set_meshes(...) / neurosetta.set_meshes(...) instead."
            )
        data = import_mesh(
            path,
            mesh_type="Neuron",
            set_units=set_units,
            voxel_size=voxel_size,
            voxel_unit=voxel_unit,
        )
    elif set_units is not None or voxel_size is not None or voxel_unit is not None:
        raise ValueError("set_units / voxel_size / voxel_unit only apply when data is a path")

    mesh = _coerce_mesh(data, tree_id=tree.ID)
    _prepare_mesh_bind(mesh, tree, context="set_mesh")
    _bind_mesh_gp(tree, mesh)
    return mesh


def sync_attached_mesh_units(tree: _Tree) -> None:
    """After a tree unit change, stamp attached mesh units to match (if any)."""
    from ..units.mesh_facet_units import stamp_mesh_units_from_tree

    mesh = get_mesh(tree)
    if mesh is None:
        return
    stamp_mesh_units_from_tree(mesh, tree)
    _bind_mesh_gp(tree, mesh)


__all__ = [
    "_MESH_GP",
    "_bind_mesh_gp",
    "_copy_mesh_value",
    "_mesh_to_payload",
    "clear_mesh",
    "freeze_mesh_for_save",
    "get_mesh",
    "has_mesh",
    "set_mesh",
    "sync_attached_mesh_units",
]
