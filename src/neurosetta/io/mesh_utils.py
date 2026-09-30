"""Mesh import and export utilities using vedo."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

from vedo import load as vd_load_mesh
from vedo import write

from ..core import _Forest, _Mesh
from .io_utils import _base_meta, _foreach_with_progress

MeshTypeName = Literal["Neuron", "Neuropil"]


def _apply_mesh_import_units(
    obj: _Mesh,
    set_units: str | None,
    *,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> None:
    """Stamp spatial units onto an imported mesh without rescaling vertices."""
    if set_units is None:
        return
    from ..ops.units.mesh_units import set_units as _set_mesh_units

    _set_mesh_units(
        obj,
        set_units,
        convert=False,
        voxel_size=voxel_size,
        voxel_unit=voxel_unit,
    )


def _mesh_meta_for_path(path: Path) -> dict:
    meta = _base_meta()
    meta["file_path"] = str(path)
    return meta


@overload
def import_mesh(
    fpath: str | Path,
    *,
    mesh_type: MeshTypeName = "Neuron",
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> _Mesh: ...


@overload
def import_mesh(
    fpath: str | Path,
    *,
    mesh_type: MeshTypeName = "Neuron",
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> _Forest: ...


def import_mesh(
    fpath: str | Path,
    *,
    mesh_type: MeshTypeName = "Neuron",
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
):
    """Import one or more mesh files as neurosetta mesh objects.

    For **neuron** meshes, the preferred follow-up is to attach onto
    morphologies — :meth:`~neurosetta.api.Tree.set_mesh` (single file or
    ``Tree_mesh``) / :meth:`~neurosetta.api.Forest.set_meshes` (directory) —
    rather than treating ``Tree_mesh`` / ``Forest_mesh`` as the primary neuron
    API. Neuropil imports are compartment geometry and stay separate.

    Parameters
    ----------
    fpath : str or pathlib.Path
        Path to a mesh file, or to a directory of mesh files supported by vedo
        (for example ``.ply``, ``.obj``, ``.stl``, ``.vtk``).
    mesh_type : {\"Neuron\", \"Neuropil\"}, optional
        Object type to construct. ``\"Neuron\"`` yields
        :class:`~neurosetta.api.Tree_mesh` /
        :class:`~neurosetta.api.Forest_mesh`. ``\"Neuropil\"`` yields
        :class:`~neurosetta.api.Neuropil` /
        :class:`~neurosetta.api.Neuropils`. Default is ``\"Neuron\"``.
    set_units : str or None, optional
        Declare spatial units for imported coordinates (no vertex rescaling).
        By default leave ``metadata[\"units\"]`` as dimensionless from
        :func:`~neurosetta.io.io_utils._base_meta`.
    voxel_size, voxel_unit
        Required together when ``set_units=\"voxel\"``.

    Returns
    -------
    Tree_mesh, Neuropil, Forest_mesh, or Neuropils
        A single mesh object for a file path, or a collection for a directory.

    Raises
    ------
    ValueError
        If ``mesh_type`` is invalid or voxel kwargs are incomplete.
    FileNotFoundError
        If ``fpath`` does not exist.

    Examples
    --------
    Import and attach a neuron mesh::

        tree.set_mesh(\"42.ply\", set_units=\"nm\")
        # or: tree.set_mesh(import_mesh(\"42.ply\", mesh_type=\"Neuron\", set_units=\"nm\"))

    Batch-attach a directory onto a Forest::

        forest.set_meshes(\"meshes/\", set_units=\"um\")

    Import neuropil meshes from a directory::

        neuropils = import_mesh(\"meshes/\", mesh_type=\"Neuropil\")
    """
    if mesh_type not in ("Neuron", "Neuropil"):
        raise ValueError(f"mesh_type must be 'Neuron' or 'Neuropil', not {mesh_type!r}")

    # Import here — avoid circular imports
    from ..api import Forest_mesh, Neuropil, Neuropils, Tree_mesh

    p = Path(fpath)

    def _import_one(path: Path) -> _Mesh:
        mesh = vd_load_mesh(str(path))
        mesh_id = path.stem
        meta = _mesh_meta_for_path(path)
        if mesh_type == "Neuron":
            obj: _Mesh = Tree_mesh(ID=mesh_id, metadata=meta, mesh=mesh)
        else:
            obj = Neuropil(ID=mesh_id, metadata=meta, mesh=mesh)
        _apply_mesh_import_units(obj, set_units, voxel_size=voxel_size, voxel_unit=voxel_unit)
        return obj

    if p.is_file():
        return _import_one(p)

    if not p.is_dir():
        raise FileNotFoundError(f"Path not found: {p}")

    loaded = vd_load_mesh(str(p))
    if not isinstance(loaded, (list, tuple)):
        loaded = [loaded]

    objs: list[_Mesh] = []
    for m in loaded:
        fname = getattr(m, "filename", None) or getattr(m, "name", None) or "mesh"
        path = Path(fname)
        mesh_id = path.stem if path.stem else str(fname)
        meta = _mesh_meta_for_path(path if path.suffix else p / f"{mesh_id}.ply")
        if mesh_type == "Neuron":
            obj = Tree_mesh(ID=mesh_id, metadata=meta, mesh=m)
        else:
            obj = Neuropil(ID=mesh_id, metadata=meta, mesh=m)
        _apply_mesh_import_units(obj, set_units, voxel_size=voxel_size, voxel_unit=voxel_unit)
        objs.append(obj)

    if mesh_type == "Neuron":
        return Forest_mesh(objs)
    return Neuropils(objs)


@overload
def export_mesh(
    mesh: _Mesh,
    fpath: str | Path | None = None,
) -> Path: ...


@overload
def export_mesh(
    mesh: _Forest,
    fpath: str | Path,
    *,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> list[Path]: ...


def export_mesh(
    mesh: _Mesh | _Forest,
    fpath: str | Path | None = None,
    fileoutput: str = ".ply",
    *,
    binary: bool = True,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
):
    """Export one or more meshes to disk.

    Parameters
    ----------
    mesh : Tree_mesh, Neuropil, Forest_mesh, or Neuropils
        Mesh object or collection to export.
    fpath : str or pathlib.Path or None, optional
        Output file or directory. If ``None``, writes next to the current
        working directory using the object ID. For collections, must be a
        directory path.
    fileoutput : str, optional
        File extension / suffix used when building output names.
        Default is ``\".ply\"``.
    binary : bool, optional
        Write binary mesh files when supported. Default is True.
    parallel : bool, optional
        Export collection members in parallel. Default is False.
    max_workers : int or None, optional
        Worker count when ``parallel=True``.
    show_progress : bool, optional
        Show a progress bar for collections. Default is False.

    Returns
    -------
    pathlib.Path or list of pathlib.Path
        Written path for a single mesh, or a list of paths for a collection.

    Raises
    ------
    ValueError
        If a collection is exported to a single file path.

    Examples
    --------
    Export a neuropil mesh::

        export_mesh(neuropil, \"out/AL.ply\")

    Export all neuron meshes in a collection::

        export_mesh(forest_mesh, \"out_meshes/\", show_progress=True)
    """

    def _save_one(t, base: Path) -> Path:
        out = base / f"{t.ID}{fileoutput}"
        out.parent.mkdir(parents=True, exist_ok=True)
        write(t.mesh, str(out), binary=binary)
        return out

    # ---- Single Mesh ----
    if not isinstance(mesh, _Forest):
        if fpath is None:
            out = Path.cwd() / f"{mesh.ID}{fileoutput}"
        else:
            p = Path(fpath)
            if p.exists() and p.is_dir():
                out = p / f"{mesh.ID}{fileoutput}"
            else:
                if p.suffix:
                    out = p
                else:
                    p.mkdir(parents=True, exist_ok=True)
                    out = p / f"{mesh.ID}{fileoutput}"

        out.parent.mkdir(parents=True, exist_ok=True)
        write(mesh.mesh, str(out), binary=binary)
        return out

    # ---- Forest ----
    base = Path.cwd() if fpath is None else Path(fpath)

    if base.suffix:
        raise ValueError("Cannot export a Forest to a single file path")

    base.mkdir(parents=True, exist_ok=True)

    items = list(mesh)
    out_paths: list[Path] = []

    def _wrapped_save(t):
        p = _save_one(t, base)
        out_paths.append(p)

    _foreach_with_progress(
        _wrapped_save,
        items,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
        desc="Exporting mesh files",
    )

    return out_paths
