"""Functions for generating a surface mesh from a Forest"""

import vedo as vd
import numpy as np
import jax.numpy as jnp
from jax.ops import segment_sum
from sklearn.neighbors import NearestNeighbors
import trimesh
from skimage.measure import marching_cubes
from scipy.ndimage import binary_dilation, label

from ..api import Forest, Neuropil

### to be re-homed at some point, ok in analysis for now

def is_point_inside_mesh(
    mesh: vd.mesh.Mesh, query_points: np.ndarray, invert: bool = False, **kwargs
) -> np.ndarray:
    """
    Determine whether each point lies inside a given mesh.

    Parameters
    ----------
    mesh : vd.Mesh
        A Vedo mesh.
    query_points : np.ndarray
        Points to be tested.
    invert : bool, optional
        Invert inside-outside test.

    Returns
    -------
    np.ndarray
        Boolean array indicating whether each point is inside.
    """
    assert isinstance(mesh, vd.Mesh), "Mesh input must be a Vedo Mesh object."
    if isinstance(query_points, np.ndarray):
        query_points = (
            vd.Points(query_points) if query_points.ndim > 1 else vd.Point(query_points)
        )
    assert isinstance(
        query_points, vd.Points
    ), "Query points must be a numpy array or vd.Points."
    indices_inside = mesh.inside_points(
        query_points, invert=invert, return_ids=True, **kwargs
    )
    result = np.zeros(query_points.npoints, dtype=bool)
    result[indices_inside] = True
    return result

def downsample_points_voxel_grid(points: jnp.ndarray, voxel_size: float) -> jnp.ndarray:
    """
    Downsample points using voxel grid averaging.

    Parameters
    ----------
    points : jnp.ndarray
        Input points.
    voxel_size : float
        Size of each voxel.

    Returns
    -------
    jnp.ndarray
        Downsampled points.
    """
    voxel_indices = jnp.floor(points / voxel_size).astype(jnp.int32)
    factor = jnp.array([1_000_000, 1_000, 1], dtype=jnp.int32)
    voxel_hash = jnp.dot(voxel_indices, factor)
    unique_hashes, inv = jnp.unique(voxel_hash, return_inverse=True)
    N = unique_hashes.shape[0]
    sums = segment_sum(points, inv, num_segments=N)
    counts = segment_sum(jnp.ones(points.shape[0]), inv, num_segments=N).reshape(-1, 1)
    return sums / counts

def remove_point_cloud_outliers(
    points: np.ndarray, k: int = 10, quantile: int = 95
) -> np.ndarray:
    """
    Remove outliers in a point cloud using local neighborhood distances.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud.
    k : int
        Number of neighbors for distance computation.
    quantile : int
        Distance threshold quantile.

    Returns
    -------
    np.ndarray
        Filtered point cloud.
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
    """
    Generate a voxel grid from 3D points.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud.
    voxel_size : float
        Edge length of each voxel.
    padding : int
        Number of voxel units to pad the bounding box.
    dilate_iter : int
        Number of dilation iterations.
    keep_largest : bool
        Keep only the largest connected component.

    Returns
    -------
    tuple
        (voxel grid, grid offset, voxel size)
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
    """
    Convert a binary voxel grid into a vedo.Volume.

    Parameters
    ----------
    grid : np.ndarray
        3D boolean array of shape (nz, ny, nx).
    offset : np.ndarray
        The (x,y,z) coordinate of the voxel grid origin.
    voxel_size : float
        Edge length of each voxel.
    as_uint8 : bool
        Whether to cast the grid to uint8 (0/1) for the Volume scalar field.

    Returns
    -------
    vd.Volume
        A volumetric object with correct spacing and position.
    """
    # 1. Prepare scalar data (0–1) in the order Vedo expects (z fastest)
    data = grid.astype(np.uint8 if as_uint8 else float)
    # Vedo will interpret the first axis as z, then y, then x
    # so no need to transpose if grid is (nz, ny, nx).

    # 2. Create the Volume with proper spacing
    vol = vd.Volume(data, spacing=(voxel_size, voxel_size, voxel_size))

    # 3. Shift it so that its minimum corner is at `offset`
    vol.pos(offset.tolist())

    return vol

def surface_from_voxel_grid(
    grid: np.ndarray, offset: np.ndarray, voxel_size: float = 1.0
) -> vd.Mesh:
    """
    Generate a mesh from a binary voxel grid using marching cubes.

    Parameters
    ----------
    grid : np.ndarray
        3D binary voxel grid.
    offset : np.ndarray
        Origin of the voxel grid.
    voxel_size : float
        Size of each voxel.

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

def clean_mesh(mesh, voxel_size: float = None) -> vd.Mesh:
    """
    Clean mesh by removing degenerate elements and optionally revoxelizing.

    Parameters
    ----------
    mesh : trimesh.Trimesh or vd.Mesh
        Input mesh.
    voxel_size : float or None
        If not watertight, revoxelize at this resolution.

    Returns
    -------
    vd.Mesh
        Cleaned, watertight mesh.
    """
    #  Close holes and split into connected components
    mesh.fill_holes()
    components = mesh.split()
    if not components:
        raise ValueError("Mesh.split() returned no components.")
    # 3. Keep only the largest component by volume
    mesh = max(components, key=lambda m: m.volume)
    mesh.remove_unreferenced_vertices()

    # 4. If still non-watertight, re-voxelise & re-march
    if not mesh.is_watertight and voxel_size is not None:
        vox = mesh.voxelized(pitch=voxel_size)
        mesh = vox.marching_cubes

    # 5. Final clean/check
    mesh.process(validate=False)
    if mesh.volume < 0:
        mesh.invert()

    # 6. Return as your vd.Mesh wrapper
    return mesh

def reconstruct_surface_voxel(
    points: np.ndarray,
    voxel_size: float = 1.0,
    padding: int = 2,
    dilate: int = 1,
    largest_component: bool = False,
    clean: bool = True,
) -> trimesh.Trimesh:
    """
    Reconstruct surface via voxelization and marching cubes.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud.
    voxel_size : float
        Voxel pitch.
    padding : int
        Padding in voxel units.
    dilate : int
        Number of dilation steps.
    largest_component : bool
        Keep only largest blob.
    clean : bool
        Post-process the mesh.

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
    """_summary_

    Parameters
    ----------
    forest : Forest
        Forest of neuron tree graphs to build mesh aound
    name : str
        name to give the output neuropil
    voxel_size : int, optional
        size of voxels used to generate mesh, by default 3
    remove_outliers : bool, optional
        If True, will try to remove outliers fom the underlying point cloud, by default True
    k : int, optional
        Number of nearest neighbours to use for outlier detection, by default 10
    quantile : int, optional
        percentage quantile fo outlier detection, by default 95
    padding : int, optional
        point cloud padding in voxel units, by default 2
    dilate : int, optional
        number of dilation steps, by default 1
    largest_component : bool, optional
        keep only the largest blob, by default False
    clean : bool, optional
        If True, we try to clean up the mesh, by default True
    smooth : bool, optional
        if True, we smooth the mesh surface, by default True
    smooth_kwargs : dict | None, optional
        kwargs passed to vedo.Mesh.smooth, by default None (uses defaults)

    Returns
    -------
    Neuropil
        _description_

    Raises
    ------
    AttributeError
        _description_
    """

    # get coordinates
    coords = np.vstack(forest.forest_node_coordinates())

    if forest[0].metadata["units"] == "nm":
        coords = coords * 0.001
    else:
        raise AttributeError(
            "We are not sure about units but need nm inputs to convert to microns"
        )

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
        largest_component = largest_component,
        clean=clean,
    )

    # convert to vedo mesh
    mesh = vd.Mesh(surf)

    # smooth
    if smooth:
        if smooth_kwargs is None:
            smooth_kwargs = {}
        mesh.smooth(**smooth_kwargs)

    metadata = {"ID": name, "units": "microns"}
    # neuropil object
    return Neuropil(ID=name, metadata=metadata, mesh=mesh)
