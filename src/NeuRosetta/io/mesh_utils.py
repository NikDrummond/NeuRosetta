""""""
from pathlib import Path
from typing import overload

from vedo import Mesh
from vedo import load as vd_load_mesh


from ..core import _Mesh, _Forest
from .io_utils import _base_meta

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
        raise AttributeError(f"mesh_type must be Neuron or Neuropil, not f{mesh_type}")

    # When I make api classes for this I import them here
    from ..api import Tree_mesh, Neuropil, Forest_mesh, Neuropils

    p = Path(fpath)

    def _import_one(path: Path, mesh_type: str) -> _Mesh:
        mesh = vd_load_mesh(path)
        mesh_id = path.stem
        meta = _base_meta()
        meta['ID'] = mesh_id
        meta['file_path'] = str(path)
        if mesh_type is 'Neuron':
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
