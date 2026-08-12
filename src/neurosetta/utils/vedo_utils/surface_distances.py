"""Functions for computing distances from points to mesh surfaces."""

from numpy import asarray, cross, einsum, ndarray, unique, where, zeros
from numpy.linalg import norm
from scipy.spatial import KDTree
from vedo import Mesh


def build_submesh(mesh: Mesh, face_indices=None) -> Mesh:
    """Build a submesh from a subset of faces.

    Parameters
    ----------
    mesh : vedo.Mesh
        Input mesh.
    face_indices : array-like, optional
        Indices of faces to include in the submesh. If None, return the
        original mesh. By default None.

    Returns
    -------
    vedo.Mesh
        Submesh containing only the specified faces with remapped vertices.
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
    """Compute distances from points to the closest vertices on a mesh.

    Parameters
    ----------
    points : ndarray
        Query points with shape (N, 3).
    mesh : vedo.Mesh
        Target mesh.
    face_indices : array-like, optional
        Indices of faces to restrict the search to. If None, use all faces.
        By default None.

    Returns
    -------
    tuple[ndarray, ndarray]
        - dists: Array of distances with shape (N,).
        - closest_pts: Array of closest vertex coordinates with shape (N, 3).
    """
    target = build_submesh(mesh, face_indices)
    verts = asarray(target.vertices)  # (V, 3)
    points = asarray(points, dtype=float)

    kd = KDTree(verts)
    dists, idx = kd.query(points)  # vectorised, no Python loop
    closest_pts = verts[idx]

    return dists, closest_pts


def _compute_face_centroids_and_normals(mesh: Mesh, face_indices=None):
    """Compute centroids and normals for mesh faces.

    Parameters
    ----------
    mesh : vedo.Mesh
        Input mesh.
    face_indices : array-like, optional
        Indices of faces to compute for. If None, use all faces.

    Returns
    -------
    tuple[ndarray, ndarray]
        - centroids: Face centroids with shape (F, 3).
        - normals: Face normals with shape (F, 3), normalized to unit length.
    """
    all_faces = asarray(mesh.cells)  # (F, 3)
    all_verts = asarray(mesh.vertices)  # (V, 3)

    faces = all_faces[asarray(face_indices)] if face_indices is not None else all_faces

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
    """Get face indices based on dot product of normal with vector from origin.

    Parameters
    ----------
    mesh : vedo.Mesh
        Input mesh.
    t : float
        Threshold for dot product.
    face : str, optional
        Which faces to return: "inner" (dot < -t), "outer" (dot > t),
        or "both" (return both). By default "both".
    origin : ndarray, optional
        Origin point for vector computation. By default [0, 0, 0].

    Returns
    -------
    ndarray | tuple[ndarray, ndarray]
        Face indices matching the criteria. If face="both", returns a tuple
        of (inner_indices, outer_indices).

    Raises
    ------
    AttributeError
        If face is not one of "inner", "outer", or "both".
    """
    # set origin to [0,0,0] if not provided
    if origin is None:
        origin = zeros(3)

    # get face centroids and normals
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
        raise AttributeError(f'face must be one of ["inner", "outer", "both"] for {face}')


def mesh_surface_depth(
    mesh: Mesh, points: ndarray, t: float, surface: str = "inner", norm: bool = True
) -> ndarray:
    """Compute normalized depth of points relative to mesh surfaces.

    Computes distance from points to inner and/or outer surfaces of a mesh,
    optionally normalized by the sum of distances to both surfaces.

    Parameters
    ----------
    mesh : Mesh
        Input mesh with well-defined inner/outer faces.
    points : ndarray
        Query points with shape (N, 3).
    t : float
        Dot product threshold for classifying inner/outer faces.
    surface : str, optional
        Which surface to compute depth for: "inner" or "outer". By default "inner".
    norm : bool, optional
        If True, normalize distances by sum of inner and outer distances.
        By default True.

    Returns
    -------
    ndarray
        Array of depth values with shape (N,). Values in [0, 1] if normalized.

    Raises
    ------
    AttributeError
        If surface is not "inner" or "outer".
    """
    # get subsets of inner and outer faces
    inner, outer = _get_face_subset_from_dot(mesh, t=t)

    # distances from inner and outer surfaces
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
