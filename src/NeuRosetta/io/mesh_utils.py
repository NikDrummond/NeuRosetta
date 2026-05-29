""""""
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
    mesh_id: int | str | None,
    parallel: bool = False,
    max_workers: int | None = None,
    progress: bool = False,
) -> _Mesh: ...

@overload
def import_mesh(
    fpath: str | Path,
    *,
    mesh_id: int | str | None,
    parallel: bool = False,
    max_workers: int | None = None,
    progress: bool = False,
) -> _Forest: ...

def import_mesh(
    fpath: str | Path,
    *,
    mesh_type: str = 'Neuron',
):
    """"""

    if mesh_type not in ['Neuron', 'Neuropil']:
        raise AttributeError(f"mesh_type must be Neuron or Neuropil, not {mesh_type}")

    # When I make api classes for this I import them here
    from ..api import Tree_mesh, Neuropil, Forest_mesh, Neuropils

    p = Path(fpath)

    def _import_one(path: Path, mesh_type: str) -> _Mesh:
        mesh = vd_load_mesh(path)
        mesh_id = path.stem
        meta = _base_meta()
        meta['ID'] = mesh_id
        meta['file_path'] = str(path)
        if mesh_type == 'Neuron':
            return Tree_mesh(ID = mesh_id, metadata = meta, mesh = mesh)
        
        return Neuropil(ID = mesh_id, metadata = meta, mesh = mesh)

    if p.is_file():
        return _import_one(p, mesh_type = mesh_type)
    
    if not p.is_dir():
        raise FileNotFoundError(f"Path not found: {p}")

    meshes = vd_load_mesh(p)
    ids = [Path(m.filename).stem for m in meshes]

    if mesh_type == 'Neuron':
        meshes = [Tree_mesh(ID = ids[i], metadata = {}, mesh = meshes[i]) for i in range(len(ids))]
        return Forest_mesh(meshes)
        
    meshes = [Neuropil(ID = ids[i], metadata = {}, mesh = meshes[i]) for i in range(len(ids))]
    return Neuropils(meshes)

@overload
def export_mesh(
    tree: _Mesh,
    fpath: str | Path | None = None,
) -> Path: ...

@overload
def export_mesh(
    tree: _Forest,
    fpath: str | Path,
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    progress: bool = False,
) -> list[Path]: ...

def export_mesh(
    mesh: _Mesh | _Forest,
    fpath: str | Path | None = None,
    fileoutput: str = '.ply',
    *,
    binary: bool = True,
    parallel: bool = False,
    max_workers: int | None = None,
    progress: bool = False,
):

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
        progress=progress,
        desc="Exporting mesh files",
    )

    return out_paths