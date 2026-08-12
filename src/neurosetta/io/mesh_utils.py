"""Mesh import and export utilities using vedo."""

from pathlib import Path
from typing import overload

from vedo import Mesh, write
from vedo import load as vd_load_mesh

from ..core import _Mesh, _Forest
from .io_utils import _base_meta, _foreach_with_progress


@overload
def import_mesh(
    fpath: str | Path,
    *,
    mesh_type: str = "Neuron",
) -> _Mesh: ...


@overload
def import_mesh(
    fpath: str | Path,
    *,
    mesh_type: str = "Neuron",
) -> _Forest: ...


def import_mesh(
    fpath: str | Path,
    *,
    mesh_type: str = "Neuron",
):
    """
    Import one or more mesh files as neurosetta mesh objects.

    Parameters
    ----------
    fpath : str or pathlib.Path
        Path to a mesh file, or to a directory of mesh files supported by vedo
        (for example ``.ply``, ``.obj``, ``.stl``, ``.vtk``).
    mesh_type : {"Neuron", "Neuropil"}, optional
        Object type to construct. ``"Neuron"`` yields :class:`~neurosetta.api.Tree_mesh`
        / :class:`~neurosetta.api.Forest_mesh`. ``"Neuropil"`` yields
        :class:`~neurosetta.api.Neuropil` / :class:`~neurosetta.api.Neuropils`.
        Default is ``"Neuron"``.

    Returns
    -------
    Tree_mesh, Neuropil, Forest_mesh, or Neuropils
        A single mesh object for a file path, or a collection for a directory.

    Raises
    ------
    AttributeError
        If ``mesh_type`` is not ``"Neuron"`` or ``"Neuropil"``.
    FileNotFoundError
        If ``fpath`` does not exist.

    Examples
    --------
    Import a neuron mesh::

        neuron_mesh = import_mesh("42.ply", mesh_type="Neuron")

    Import neuropil meshes from a directory::

        neuropils = import_mesh("meshes/", mesh_type="Neuropil")
    """
    if mesh_type not in ["Neuron", "Neuropil"]:
        raise AttributeError(f"mesh_type must be Neuron or Neuropil, not {mesh_type}")

    # Import here - avoid circular imports
    from ..api import Tree_mesh, Neuropil, Forest_mesh, Neuropils

    p = Path(fpath)

    def _import_one(path: Path, mesh_type: str) -> _Mesh:
        mesh = vd_load_mesh(path)
        mesh_id = path.stem
        meta = _base_meta()
        meta["file_path"] = str(path)
        if mesh_type == "Neuron":
            return Tree_mesh(ID=mesh_id, metadata=meta, mesh=mesh)

        return Neuropil(ID=mesh_id, metadata=meta, mesh=mesh)

    if p.is_file():
        return _import_one(p, mesh_type=mesh_type)

    if not p.is_dir():
        raise FileNotFoundError(f"Path not found: {p}")

    meshes = vd_load_mesh(p)
    ids = [Path(m.filename).stem for m in meshes]

    if mesh_type == "Neuron":
        meshes = [
            Tree_mesh(ID=ids[i], metadata={}, mesh=meshes[i]) for i in range(len(ids))
        ]
        return Forest_mesh(meshes)

    meshes = [Neuropil(ID=ids[i], metadata={}, mesh=meshes[i]) for i in range(len(ids))]
    return Neuropils(meshes)


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
    parallel: bool = False,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> list[Path]: ...


def export_mesh(
    mesh: _Mesh | _Forest,
    fpath: str | Path | None = None,
    fileoutput: str = ".ply",
    *,
    binary: bool = True,
    parallel: bool = False,
    max_workers: int | None = None,
    show_progress: bool = False,
):
    """
    Export one or more meshes to disk.

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
        Default is ``".ply"``.
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

        export_mesh(neuropil, "out/AL.ply")

    Export all neuron meshes in a collection::

        export_mesh(forest_mesh, "out_meshes/", progress=True)
    """

    def _save_one(t, base: Path) -> Path:
        out = base / f"{t.ID}{fileoutput}"
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

        write(mesh.mesh, str(out), binary=binary)
        return out

    # ---- Forest ----
    if fpath is None:
        base = Path.cwd()
    else:
        base = Path(fpath)

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
