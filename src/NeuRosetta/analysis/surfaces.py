"""Functions for generating surface meshes from neuron forests.

This module provides utilities for reconstructing 3D surface meshes from
point clouds of neuron morphologies, including voxelization, marching cubes,
and mesh cleaning operations.
"""

import vedo as vd
import numpy as np
# import jax.numpy as jnp
# from jax.ops import segment_sum
from sklearn.neighbors import NearestNeighbors
import trimesh
from skimage.measure import marching_cubes
from scipy.ndimage import binary_dilation, label

from ..api import Forest, Neuropil
from ..ops.units import ensure_forest_units


def is_point_inside_mesh(
    mesh: vd.Mesh, query_points: np.ndarray, invert: bool = False, **kwargs
) -> np.ndarray:
    """Determine whether each point lies inside a given mesh.

    Parameters
    ----------
    mesh : vd.Mesh
        A Vedo mesh object.
    query_points : np.ndarray
        Points to be tested, shape (N, 3).
    invert : bool, optional
        Invert inside-outside test. By default False.
    **kwargs : dict
        Additional keyword arguments passed to vedo.Mesh.inside_points.

    Returns
    -------
    np.ndarray
        Boolean array of shape (N,) indicating whether each point is inside.
    """
    assert isinstance(mesh, vd.Mesh), "Mesh input must be a Vedo Mesh object."
    if isinstance(query_points, np.ndarray):
        n_points = query_points.shape[0]
        query_points = (
            vd.Points(query_points) if query_points.ndim > 1 else vd.Point(query_points)
        )
    else:
        n_points = query_points.N()
    assert isinstance(
        query_points, vd.Points
    ), "Query points must be a numpy array or vd.Points."
    indices_inside = mesh.inside_points(
        query_points, invert=invert, return_ids=True, **kwargs
    )
    result = np.zeros(n_points, dtype=bool)
    result[indices_inside] = True
    return result


# def downsample_points_voxel_grid(points: jnp.ndarray, voxel_size: float) -> jnp.ndarray:

#     voxel_indices = jnp.floor(points / voxel_size).astype(jnp.int32)
#     factor = jnp.array([1_000_000, 1_000, 1], dtype=jnp.int32)
#     voxel_hash = jnp.dot(voxel_indices, factor)
#     unique_hashes, inv = jnp.unique(voxel_hash, return_inverse=True)
#     n_segments = unique_hashes.shape[0]
#     sums = segment_sum(points, inv, num_segments=n_segments)
#     counts = segment_sum(jnp.ones(points.shape[0]), inv, num_segments=n_segments).reshape(
#         -1, 1
#     )
#     return sums / counts


def downsample_points_voxel_grid(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsample points using voxel grid averaging.

    Parameters
    ----------
    points : np.ndarray
        Input points, shape (N, 3).
    voxel_size : float
        Size of each voxel.

    Returns
    -------
    np.ndarray
        Downsampled points, shape (M, 3) where M <= N.
    """
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    factor = np.array([1_000_000, 1_000, 1], dtype=np.int64)
    voxel_hash = voxel_indices @ factor

    unique_hashes, inv = np.unique(
        voxel_hash, return_inverse=True
    )

    sums = np.zeros((len(unique_hashes), 3), dtype=points.dtype)
    np.add.at(sums, inv, points)

    counts = np.bincount(inv)

    return sums / counts[:, None]

def remove_point_cloud_outliers(
    points: np.ndarray, k: int = 10, quantile: int = 95
) -> np.ndarray:
    """Remove outliers in a point cloud using local neighborhood distances.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud, shape (N, 3).
    k : int, optional
        Number of neighbors for distance computation. By default 10.
    quantile : int, optional
        Distance threshold quantile (0-100). By default 95.

    Returns
    -------
    np.ndarray
        Filtered point cloud, shape (M, 3) where M <= N.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    dists, _ = nbrs.kneighbors(points)
    mean_dists = dists.mean(axis=1)
    return points[mean_dists < np.percentile(mean_dists, quantile)]


def generate_voxel_grid(
    points: np.ndarray,
    voxel_size: float = 1.0,
    padding: int = 2,
    dilate_iter: int = 1,
    keep_largest: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generate a voxel grid from 3D points.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud, shape (N, 3).
    voxel_size : float, optional
        Edge length of each voxel. By default 1.0.
    padding : int, optional
        Number of voxel units to pad the bounding box. By default 2.
    dilate_iter : int, optional
        Number of dilation iterations. By default 1.
    keep_largest : bool, optional
        Keep only the largest connected component. By default False.

    Returns
    -------
    tuple
        ``(voxel_grid, grid_offset, voxel_size)`` where ``voxel_grid`` is a 3D
        boolean array, ``grid_offset`` is the (x, y, z) origin, and
        ``voxel_size`` is the edge length of voxels.
    """
    min_corner = points.min(axis=0) - voxel_size * padding
    max_corner = points.max(axis=0) + voxel_size * padding
    dims = np.ceil((max_corner - min_corner) / voxel_size).astype(int)
    offset = min_corner

    indices = np.floor((points - offset) / voxel_size).astype(int)
    indices = indices[(indices >= 0).all(axis=1) & (indices < dims).all(axis=1)]

    grid = np.zeros(dims, dtype=bool)
    grid[tuple(indices.T)] = True

    if dilate_iter > 0:
        grid = binary_dilation(grid, iterations=dilate_iter)

    if keep_largest:
        labeled, num = label(grid)
        if num == 0:
            raise ValueError("No connected components found.")
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        grid = labeled == sizes.argmax()

    return grid, offset, voxel_size


def voxel_grid_to_volume(
    grid: np.ndarray,
    offset: np.ndarray,
    voxel_size: float,
    as_uint8: bool = True,
) -> vd.Volume:
    """Convert a binary voxel grid into a vedo.Volume.

    Parameters
    ----------
    grid : np.ndarray
        3D boolean array of shape (nz, ny, nx).
    offset : np.ndarray
        The (x, y, z) coordinate of the voxel grid origin.
    voxel_size : float
        Edge length of each voxel.
    as_uint8 : bool, optional
        Whether to cast the grid to uint8 (0/1) for the Volume scalar field.
        By default True.

    Returns
    -------
    vd.Volume
        A volumetric object with correct spacing and position.
    """
    data = grid.astype(np.uint8 if as_uint8 else float)
    vol = vd.Volume(data, spacing=(voxel_size, voxel_size, voxel_size))
    vol.pos(offset.tolist())
    return vol


def surface_from_voxel_grid(
    grid: np.ndarray, offset: np.ndarray, voxel_size: float = 1.0
) -> trimesh.Trimesh:
    """Generate a mesh from a binary voxel grid using marching cubes.

    Parameters
    ----------
    grid : np.ndarray
        3D binary voxel grid.
    offset : np.ndarray
        Origin of the voxel grid.
    voxel_size : float, optional
        Size of each voxel. By default 1.0.

    Returns
    -------
    trimesh.Trimesh
        Surface mesh.
    """
    verts, faces, _, _ = marching_cubes(
        grid.astype(np.float32), level=0.5, spacing=(voxel_size,) * 3
    )
    verts += offset
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def clean_mesh(mesh: trimesh.Trimesh | vd.Mesh, voxel_size: float | None = None) -> vd.Mesh:
    """Clean mesh by removing degenerate elements and optionally revoxelizing.

    Parameters
    ----------
    mesh : trimesh.Trimesh or vd.Mesh
        Input mesh.
    voxel_size : float or None, optional
        If mesh is not watertight, revoxelize at this resolution. By default None.

    Returns
    -------
    vd.Mesh
        Cleaned, watertight mesh.
    """
    # Close holes and split into connected components
    mesh.fill_holes()
    components = mesh.split()
    if not components:
        raise ValueError("Mesh.split() returned no components.")
    # Keep only the largest component by volume
    mesh = max(components, key=lambda m: m.volume)
    mesh.remove_unreferenced_vertices()

    # If still non-watertight, re-voxelise & re-march
    if not mesh.is_watertight and voxel_size is not None:
        vox = mesh.voxelized(pitch=voxel_size)
        mesh = vox.marching_cubes

    # Final clean/check
    mesh.process(validate=False)
    if mesh.volume < 0:
        mesh.invert()

    # Return as your vd.Mesh wrapper
    return mesh


def reconstruct_surface_voxel(
    points: np.ndarray,
    voxel_size: float = 1.0,
    padding: int = 2,
    dilate: int = 1,
    largest_component: bool = False,
    clean: bool = True,
) -> trimesh.Trimesh:
    """Reconstruct surface via voxelization and marching cubes.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud, shape (N, 3).
    voxel_size : float, optional
        Voxel pitch. By default 1.0.
    padding : int, optional
        Padding in voxel units. By default 2.
    dilate : int, optional
        Number of dilation steps. By default 1.
    largest_component : bool, optional
        Keep only largest blob. By default False.
    clean : bool, optional
        Post-process the mesh. By default True.

    Returns
    -------
    trimesh.Trimesh
        Reconstructed mesh.
    """
    grid, offset, voxel_size = generate_voxel_grid(
        points, voxel_size, padding, dilate, largest_component
    )
    mesh = surface_from_voxel_grid(grid, offset, voxel_size)
    mesh = clean_mesh(mesh, voxel_size) if clean else mesh
    return mesh


def reconstruct_neuropil_surface(
    forest: Forest,
    name: str,
    voxel_size: int = 3,
    remove_outliers: bool = True,
    k: int = 10,
    quantile: int = 95,
    padding: int = 2,
    dilate: int = 1,
    largest_component: bool = False,
    clean: bool = True,
    smooth: bool = True,
    smooth_kwargs: dict | None = None,
) -> Neuropil:
    """Reconstruct a neuropil surface mesh from a forest of neuron trees.

    Parameters
    ----------
    forest : Forest
        Forest of neuron tree graphs to build mesh around.
    name : str
        Name to give the output neuropil.
    voxel_size : int, optional
        Size of voxels used to generate mesh, in microns. By default 3.
    remove_outliers : bool, optional
        If True, remove outliers from the underlying point cloud. By default True.
    k : int, optional
        Number of nearest neighbours to use for outlier detection. By default 10.
    quantile : int, optional
        Percentage quantile for outlier detection. By default 95.
    padding : int, optional
        Point cloud padding in voxel units. By default 2.
    dilate : int, optional
        Number of dilation steps. By default 1.
    largest_component : bool, optional
        Keep only the largest blob. By default False.
    clean : bool, optional
        If True, clean up the mesh. By default True.
    smooth : bool, optional
        If True, smooth the mesh surface. By default True.
    smooth_kwargs : dict | None, optional
        Keyword arguments passed to vedo.Mesh.smooth. By default None (uses defaults).

    Returns
    -------
    Neuropil
        Neuropil object containing the reconstructed surface mesh.

    Raises
    ------
    ValueError
        If any tree in the forest has dimensionless units after harmonization.
    """
    ensure_forest_units(forest)
    coords = np.vstack(forest.get_node_coordinates(show_progress=False))

    # downsample
    coords = downsample_points_voxel_grid(coords, voxel_size)

    # remove outliers
    if remove_outliers:
        coords = remove_point_cloud_outliers(coords, k, quantile)

    # construct surface
    surf = reconstruct_surface_voxel(
        points=coords,
        voxel_size=voxel_size,
        padding=padding,
        dilate=dilate,
        largest_component=largest_component,
        clean=clean,
    )

    # convert to vedo mesh
    mesh = vd.Mesh(surf)

    # smooth
    if smooth:
        if smooth_kwargs is None:
            smooth_kwargs = {}
        mesh.smooth(**smooth_kwargs)

    metadata = {"units": "micron"}
    # neuropil object
    return Neuropil(ID=name, metadata=metadata, mesh=mesh)