"""Mesh import and export utilities using vedo."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import Literal, overload

from vedo import load as vd_load_mesh
from vedo import write

from ..core import _Forest, _Mesh
from .io_utils import _base_meta, _foreach_with_progress, resolve_import_id, safe_export_stem

MeshTypeName = Literal["Neuron", "Neuropil"]
IdResolver = Callable[[Path], Hashable]


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
    ID: Hashable | None = None,
    id_map: Mapping[str, Hashable] | None = None,
    id_resolver: IdResolver | None = None,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> _Mesh: ...


@overload
def import_mesh(
    fpath: str | Path,
    *,
    mesh_type: MeshTypeName = "Neuron",
    ID: Hashable | None = None,
    id_map: Mapping[str, Hashable] | None = None,
    id_resolver: IdResolver | None = None,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> _Forest: ...


def import_mesh(
    fpath: str | Path,
    *,
    mesh_type: MeshTypeName = "Neuron",
    ID: Hashable | None = None,
    id_map: Mapping[str, Hashable] | None = None,
    id_resolver: IdResolver | None = None,
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

    Identity
    --------
    * ``name`` is always the file stem.
    * ``ID`` defaults to the same stem (string). Override with ``ID=``
      (single file), ``id_map``, or ``id_resolver`` — same API as
      :func:`~neurosetta.io.swc_utils.import_swc`.

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
    ID : hashable or None, optional
        Explicit logical ID for a single-file import.
    id_map, id_resolver
        Batch ID assignment (see :func:`~neurosetta.io.swc_utils.import_swc`).
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
        If ``mesh_type`` is invalid, voxel kwargs are incomplete, or ``ID=``
        is supplied for a multi-file directory import.
    FileNotFoundError
        If ``fpath`` does not exist.

    Examples
    --------
    Import with independent logical ID::

        mesh = import_mesh(\"neuron_abc_surface.ply\", ID=\"abc\")
        assert mesh.name == \"neuron_abc_surface\"
        assert mesh.ID == \"abc\"
    """
    if mesh_type not in ("Neuron", "Neuropil"):
        raise ValueError(f"mesh_type must be 'Neuron' or 'Neuropil', not {mesh_type!r}")

    # Import here — avoid circular imports
    from ..api import Forest_mesh, Neuropil, Neuropils, Tree_mesh

    p = Path(fpath)

    def _build(path: Path, mesh_obj, *, explicit_id: Hashable | None = None) -> _Mesh:
        mesh_id = resolve_import_id(
            path,
            ID=explicit_id,
            id_map=id_map,
            id_resolver=id_resolver,
        )
        name = path.stem if path.stem else str(path)
        meta = _mesh_meta_for_path(path)
        if mesh_type == "Neuron":
            obj: _Mesh = Tree_mesh(ID=mesh_id, metadata=meta, mesh=mesh_obj, name=name)
        else:
            obj = Neuropil(ID=mesh_id, metadata=meta, mesh=mesh_obj, name=name)
        _apply_mesh_import_units(obj, set_units, voxel_size=voxel_size, voxel_unit=voxel_unit)
        return obj

    if p.is_file():
        return _build(p, vd_load_mesh(str(p)), explicit_id=ID)

    if not p.is_dir():
        raise FileNotFoundError(f"Path not found: {p}")

    if ID is not None:
        raise ValueError(
            "ID= is only valid for single-file import; use id_map= or id_resolver= for directories"
        )

    loaded = vd_load_mesh(str(p))
    if not isinstance(loaded, (list, tuple)):
        loaded = [loaded]

    objs: list[_Mesh] = []
    for m in loaded:
        fname = getattr(m, "filename", None) or getattr(m, "name", None) or "mesh"
        path = Path(fname)
        stem = path.stem if path.stem else str(fname)
        meta_path = path if path.suffix else p / f"{stem}.ply"
        objs.append(_build(meta_path, m))

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

    Default filenames use ``mesh.name`` (not ``ID``). An explicit file path
    wins over either value and does not mutate ``name``.

    Parameters
    ----------
    mesh : Tree_mesh, Neuropil, Forest_mesh, or Neuropils
        Mesh object or collection to export.
    fpath : str or pathlib.Path or None, optional
        Output file or directory. If ``None``, writes next to the current
        working directory using the object name. For collections, must be a
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
    """

    def _save_one(t, base: Path) -> Path:
        out = base / f"{safe_export_stem(t.name)}{fileoutput}"
        out.parent.mkdir(parents=True, exist_ok=True)
        write(t.mesh, str(out), binary=binary)
        return out

    # ---- Single Mesh ----
    if not isinstance(mesh, _Forest):
        if fpath is None:
            out = Path.cwd() / f"{safe_export_stem(mesh.name)}{fileoutput}"
        else:
            p = Path(fpath)
            if p.exists() and p.is_dir():
                out = p / f"{safe_export_stem(mesh.name)}{fileoutput}"
            else:
                if p.suffix:
                    out = p
                else:
                    p.mkdir(parents=True, exist_ok=True)
                    out = p / f"{safe_export_stem(mesh.name)}{fileoutput}"

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
