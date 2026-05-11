"""for now, initial needed functions. whole sections needs to be built out"""

from numpy import ndarray

from ...core import _Mesh
from ...utils.vedo_utils import surface_distance, mesh_surface_depth


def distance_from_neuropil_surface(mesh: _Mesh, points: ndarray) -> ndarray:
    """_summary_

    Parameters
    ----------
    mesh : _Mesh
        _description_
    points : ndarray
        _description_

    Returns
    -------
    ndarray
        _description_
    """
    dists, _ = surface_distance(points=points, mesh=mesh.mesh)
    return dists


def neuropil_point_depth(
    mesh: _Mesh, points: ndarray, t: float, surface: str = "inner", norm: bool = True
) -> ndarray:
    """_summary_

    Parameters
    ----------
    mesh : _Mesh
        _description_
    points : ndarray
        _description_
    t : float
        _description_
    surface : str, optional
        _description_, by default 'inner'
    norm : bool, optional
        _description_, by default True

    Returns
    -------
    ndarray
        _description_
    """
    return mesh_surface_depth(
        mesh=mesh.mesh, points=points, t=t, surface=surface, norm=norm
    )
