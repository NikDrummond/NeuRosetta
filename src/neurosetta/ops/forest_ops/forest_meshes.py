"""Attach neuron meshes to Forest members by ID."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ...core import _Forest
from ...core.synapses import owner_ids_compatible
from ...ops.tree_graphs.tree_mesh import set_mesh

if TYPE_CHECKING:
    from ...api.forest_mesh_class import Forest_mesh
    from ...api.tree_mesh_class import Tree_mesh

MissingPolicy = Literal["error", "warn", "ignore"]
UnusedPolicy = Literal["error", "warn", "ignore"]


def _mesh_ids_match(mesh_id: Hashable, tree_id: Hashable) -> bool:
    """True when mesh and tree IDs refer to the same neuron (int/str coerce)."""
    # owner_ids_compatible(None, …) is True for unbound synapses — not wanted here.
    if mesh_id is None:
        return False
    return owner_ids_compatible(mesh_id, tree_id)


def _as_mesh_list(
    meshes: Forest_mesh | Tree_mesh | Mapping[Hashable, Tree_mesh] | Iterable[Tree_mesh],
) -> list[Tree_mesh]:
    from ...api.forest_mesh_class import Forest_mesh
    from ...api.tree_mesh_class import Tree_mesh

    if isinstance(meshes, Tree_mesh):
        return [meshes]
    if isinstance(meshes, Forest_mesh):
        return list(meshes)
    if isinstance(meshes, Mapping):
        out: list[Tree_mesh] = []
        for key, mesh in meshes.items():
            if not isinstance(mesh, Tree_mesh):
                raise TypeError(
                    f"Mapping values must be Tree_mesh; got {type(mesh)!r} for key {key!r}"
                )
            # Mapping key is the match ID (may coerce vs mesh.ID via set_mesh).
            if key != mesh.ID:
                mesh = Tree_mesh(
                    ID=key,
                    metadata=dict(mesh.metadata),
                    mesh=mesh.mesh,
                )
            out.append(mesh)
        return out
    items = list(meshes)
    for m in items:
        if not isinstance(m, Tree_mesh):
            raise TypeError(f"Expected Tree_mesh members; got {type(m)!r}")
    return items


def _load_meshes(
    source: str | Path,
    *,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> list[Any]:
    from ...api.forest_mesh_class import Forest_mesh
    from ...api.tree_mesh_class import Tree_mesh
    from ...io.mesh_utils import import_mesh

    loaded = import_mesh(
        source,
        mesh_type="Neuron",
        set_units=set_units,
        voxel_size=voxel_size,
        voxel_unit=voxel_unit,
    )
    if isinstance(loaded, Forest_mesh):
        return list(loaded)
    if isinstance(loaded, Tree_mesh):
        return [loaded]
    raise TypeError(f"Expected neuron mesh(es) from {source!r}; got {type(loaded)!r}")


def _match_mesh(tree_id: Hashable, meshes: list[Any]) -> Any | None:
    hits = [m for m in meshes if _mesh_ids_match(m.ID, tree_id)]
    if not hits:
        return None
    if len(hits) > 1:
        ids = [m.ID for m in hits]
        raise ValueError(
            f"Multiple meshes match tree ID {tree_id!r}: {ids}. "
            f"Ensure mesh file stems / IDs are unique per neuron."
        )
    return hits[0]


def set_meshes(
    forest: _Forest,
    meshes: (
        str | Path | Forest_mesh | Tree_mesh | Mapping[Hashable, Tree_mesh] | Iterable[Tree_mesh]
    ),
    *,
    missing: MissingPolicy = "warn",
    unused: UnusedPolicy = "ignore",
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> dict[Hashable, Tree_mesh]:
    """Attach neuron meshes to Forest trees by matching ``ID``.

    Parameters
    ----------
    forest : Forest
        Target morphology collection.
    meshes : path, Forest_mesh, Tree_mesh, mapping, or iterable
        Neuron meshes. A directory / file path is loaded via
        ``import_mesh(..., mesh_type=\"Neuron\")``. IDs are matched with the
        same int/str coercion as synapse ``owner_id`` (e.g. ``7`` ↔ ``\"7\"``).
    missing : {\"error\", \"warn\", \"ignore\"}, optional
        Policy when a tree has no matching mesh. By default ``\"warn\"``.
    unused : {\"error\", \"warn\", \"ignore\"}, optional
        Policy when a mesh matches no tree. By default ``\"ignore\"``.
    set_units, voxel_size, voxel_unit
        Forwarded to :func:`~neurosetta.import_mesh` when *meshes* is a path.

    Returns
    -------
    dict
        ``{tree.ID: bound Tree_mesh}`` for successfully attached members.

    Raises
    ------
    ValueError
        On ambiguous ID matches, or when *missing* / *unused* is ``\"error\"``.
    TypeError
        If *meshes* has the wrong type.
    """
    if missing not in ("error", "warn", "ignore"):
        raise ValueError(f"missing={missing!r} invalid")
    if unused not in ("error", "warn", "ignore"):
        raise ValueError(f"unused={unused!r} invalid")

    if isinstance(meshes, (str, Path)):
        mesh_list = _load_meshes(
            meshes,
            set_units=set_units,
            voxel_size=voxel_size,
            voxel_unit=voxel_unit,
        )
    else:
        if set_units is not None or voxel_size is not None or voxel_unit is not None:
            raise ValueError("set_units / voxel_size / voxel_unit only apply when meshes is a path")
        mesh_list = _as_mesh_list(meshes)

    attached: dict[Hashable, Any] = {}
    used: set[int] = set()

    for tree in forest:
        match = _match_mesh(tree.ID, mesh_list)
        if match is None:
            msg = f"No mesh found for tree ID {tree.ID!r}"
            if missing == "error":
                raise ValueError(msg)
            if missing == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        bound = set_mesh(tree, match)
        attached[tree.ID] = bound
        used.add(id(match))

    unused_meshes = [m for m in mesh_list if id(m) not in used]
    if unused_meshes:
        ids = [m.ID for m in unused_meshes]
        msg = f"Meshes with no matching tree were unused: {ids}"
        if unused == "error":
            raise ValueError(msg)
        if unused == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)

    return attached


__all__ = ["set_meshes"]
