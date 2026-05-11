"""distance from mesh surface function"""

from numpy import asarray, ndarray, unique, cross, where, zeros, einsum
from numpy.linalg import norm
from scipy.spatial import KDTree
from vedo import Mesh


def build_submesh(mesh: Mesh, face_indices=None) -> Mesh:
    """_summary_

    Parameters
    ----------
    mesh : vedo.Mesh
        _description_
    face_indices : _type_, optional
        _description_, by default None

    Returns
    -------
    vedo.Mesh
        _description_
    """
    if face_indices is None:
        return mesh

    face_indices = asarray(face_indices)
    all_faces = asarray(mesh.cells)  # (F, 3)
    all_verts = asarray(mesh.vertices)  # (V, 3)

    sel_faces = all_faces[face_indices]  # (k, 3)

    # Remap to a compact vertex set
    unique_vids, inverse = unique(sel_faces, return_inverse=True)
    new_verts = all_verts[unique_vids]  # (k_verts, 3)
    new_faces = inverse.reshape(-1, 3)  # (k, 3)

    return Mesh([new_verts, new_faces])


def surface_distance(
    points: ndarray,
    mesh: Mesh,
    face_indices=None,
) -> tuple[ndarray, ndarray]:
    """_summary_

    Parameters
    ----------
    points : np.ndarray
        _description_
    mesh : vedo.Mesh
        _description_
    face_indices : _type_, optional
        _description_, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        _description_
    """
    target = build_submesh(mesh, face_indices)
    verts = asarray(target.vertices)  # (V, 3)
    points = asarray(points, dtype=float)

    kd = KDTree(verts)
    dists, idx = kd.query(points)  # vectorised, no Python loop
    closest_pts = verts[idx]

    return dists, closest_pts


def _compute_face_centroids_and_normals(mesh: Mesh, face_indices=None):
    """"""

    all_faces = asarray(mesh.cells)  # (F, 3)
    all_verts = asarray(mesh.vertices)  # (V, 3)

    if face_indices is not None:
        faces = all_faces[asarray(face_indices)]
    else:
        faces = all_faces

    v0 = all_verts[faces[:, 0]]
    v1 = all_verts[faces[:, 1]]
    v2 = all_verts[faces[:, 2]]

    centroids = (v0 + v1 + v2) / 3.0  # (k, 3)

    # Cross-product normals
    normals = cross(v1 - v0, v2 - v0)  # (k, 3)
    norms = norm(normals, axis=1, keepdims=True)
    normals /= where(norms > 0, norms, 1.0)  # safe in-place normalise

    return centroids, normals


def _get_face_subset_from_dot(
    mesh: Mesh, t: float, face: str = "both", origin: ndarray = None
) -> ndarray | tuple[ndarray, ndarray]:
    """_summary_

    Parameters
    ----------
    mesh : vd.Mesh
        _description_
    t : float
        _description_
    face : str, optional
        _description_, by default 'both'
    origin : np.ndarray, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    AttributeError
        _description_
    """
    # set origin to [0,0,0] if not provided
    if origin is None:
        origin = zeros(3)

    # get face cenrtoids and notmals
    centroids, normals = _compute_face_centroids_and_normals(mesh)

    # Unit vectors from origin to each centroid
    to_cent = centroids - origin
    to_cent /= norm(to_cent, axis=1, keepdims=True)

    dot = einsum("ij,ij->i", normals, to_cent)

    if face == "inner":
        return where(dot < -t)
    elif face == "outer":
        return where(dot > t)
    elif face == "both":
        return where(dot < -t), where(dot > t)
    else:
        raise AttributeError(
            f'face must be one of ["inner", "outer", "both"] for {face}'
        )


def mesh_surface_depth(
    mesh: Mesh, points: ndarray, t: float, surface: str = "inner", norm: bool = True
) -> ndarray:
    """_summary_

    Parameters
    ----------
    mesh : Mesh
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

    Raises
    ------
    AttributeError
        _description_
    """
    # get subsets of inner and outer faces
    inner, outer = _get_face_subset_from_dot(mesh, t=t)

    # duistancess from inner and outer surfaces
    in_dists, _ = surface_distance(points, mesh, face_indices=inner)
    out_dists, _ = surface_distance(points, mesh, face_indices=outer)

    if surface == "inner":
        dists = in_dists
    elif surface == "outer":
        dists = out_dists
    else:
        raise AttributeError(f'surface must be "inner" or "outer" not {surface}')

    if norm:
        dists = dists / (in_dists + out_dists)

    return dists
