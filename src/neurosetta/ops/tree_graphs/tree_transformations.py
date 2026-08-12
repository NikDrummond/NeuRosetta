"""Functions for spatial transformations on trees."""

from typing import Literal, Tuple, Union
from numpy import ndarray, array, asarray

from .coordinates import get_node_coordinates, get_root_coordinate
from ...utils.geometry_utils.pca import eig_decomp
from ...utils.geometry_utils.rotations import (
    compute_alignment_rotation,
    apply_rotation_steps,
    rotate,
    minimum_rotation_to_align,
)
from ...utils.geometry_utils.scaling import scale_along_basis
from ...utils.geometry_utils.transforms import (
    translate,
    recenter,
    scale_about,
    center_at_centroid,
)
from ...utils.graph_utils.gt_properties import _set_coords_prop
from ...core import _Tree

CenterMode = Literal["centroid", "root"]
SourceMode = Literal["pc1"]
SourceVector = Union[SourceMode, tuple, ndarray]


def _apply_node_coordinates(
    tree: _Tree,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    bind: bool,
) -> ndarray | None:
    """Write transformed node coordinates to the graph or return them."""
    if bind:
        _set_coords_prop(tree.graph, array([x, y, z]))
        return None
    return array([x, y, z]).T


def _scale_tree_radii(tree: _Tree, factor: float) -> None:
    """Scale cached node radii by a multiplicative factor."""
    if "radius" in tree.graph.vp:
        tree.graph.vp["radius"].a *= factor


def _resolve_transform_center(
    tree: _Tree,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    center: tuple | ndarray | None,
    center_mode: CenterMode,
) -> Tuple[float, float, float]:
    """Resolve the centre point used by rotation and scaling transforms."""
    if center is not None:
        cx, cy, cz = center
        return float(cx), float(cy), float(cz)

    if center_mode == "root":
        rc = get_root_coordinate(tree, SoA=False).ravel()
        return float(rc[0]), float(rc[1]), float(rc[2])

    _, _, _, cx, cy, cz = center_at_centroid(x, y, z)
    return cx, cy, cz


def _maybe_recenter_at_centroid(
    x: ndarray,
    y: ndarray,
    z: ndarray,
    recenter: bool,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Optionally translate coordinates so their centroid lies at the origin."""
    if recenter:
        x, y, z, _, _, _ = center_at_centroid(x, y, z)
    return x, y, z


def _resolve_scale_factors(
    scale: float | int | tuple | ndarray,
) -> Tuple[float, float, float]:
    """Normalize uniform or per-axis scale factors to ``(sx, sy, sz)``."""
    if isinstance(scale, (float, int)):
        factor = float(scale)
        return factor, factor, factor
    sx, sy, sz = scale
    return float(sx), float(sy), float(sz)


def align_coordinates(
    tree: _Tree,
    robust: bool = True,
    b1: tuple = (0.0, 1.0, 0.0),
    b2: tuple = (1.0, 0.0, 0.0),
    b3: tuple = (0.0, 0.0, 0.1),
    bind: bool = True,
) -> ndarray | None:
    """PCA-align a single tree to canonical axes.

    Recenters node coordinates at their centroid, then rotates them so the
    first three principal components align with *b1*, *b2*, and *b3*.
    Optionally writes the transformed coordinates back to the tree graph.

    Parameters
    ----------
    tree : _Tree
        Tree to align.
    robust : bool, optional
        Use robust PCA. By default True.
    b1 : tuple, optional
        Target unit vector for PC1. By default (0.0, 1.0, 0.0).
    b2 : tuple, optional
        Target unit vector for PC2. By default (1.0, 0.0, 0.0).
    b3 : tuple, optional
        Target unit vector for PC3. By default (0.0, 0.0, 0.1).
    bind : bool, optional
        If True, write rotated coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Rotated (N, 3) coordinates when bind=False; None when bind=True.

    Notes
    -----
    The tree is translated so that its centroid lies at the origin before PCA
    is computed. Node radii are not modified by this operation.
    """
    coords = get_node_coordinates(tree, SoA=True)
    coords -= coords.mean(axis=1, keepdims=True)
    x = coords[0]
    y = coords[1]
    z = coords[2]

    _, evecs = eig_decomp(x, y, z, robust=robust)

    ev1 = evecs[:, 0]
    ev2 = evecs[:, 1]
    ev3 = evecs[:, 2]

    step1, step2 = compute_alignment_rotation(ev1, ev2, ev3, b1, b2, b3)

    xr, yr, zr = apply_rotation_steps(x, y, z, step1, step2)

    return _apply_node_coordinates(tree, xr, yr, zr, bind)


def translate_coordinates(
    tree: _Tree,
    dx: float,
    dy: float,
    dz: float,
    bind: bool = True,
) -> ndarray | None:
    """Translate all node coordinates by a fixed displacement.

    Applies a rigid translation to every node in the tree. Node radii and
    cached geometric properties are unchanged.

    Parameters
    ----------
    tree : _Tree
        Tree to translate.
    dx : float
        Displacement along the x axis.
    dy : float
        Displacement along the y axis.
    dz : float
        Displacement along the z axis.
    bind : bool, optional
        If True, write translated coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Translated (N, 3) coordinates when bind=False; None when bind=True.
    """
    x, y, z = get_node_coordinates(tree, SoA=True)
    x, y, z = translate(x, y, z, dx, dy, dz)
    return _apply_node_coordinates(tree, x, y, z, bind)


def center_coordinates_at_centroid(
    tree: _Tree,
    bind: bool = True,
) -> ndarray | None:
    """Translate a tree so that its node centroid lies at the origin.

    The centroid is computed independently for each coordinate as the
    arithmetic mean of all node positions. This is equivalent to subtracting
    the centroid from every node coordinate.

    Parameters
    ----------
    tree : _Tree
        Tree to center.
    bind : bool, optional
        If True, write centered coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Centered (N, 3) coordinates when bind=False; None when bind=True.

    Notes
    -----
    Node radii are not modified by this operation. This is the same
    recentering step performed internally by :func:`align_coordinates` before PCA.
    """
    x, y, z = get_node_coordinates(tree, SoA=True)
    x, y, z, _, _, _ = center_at_centroid(x, y, z)
    return _apply_node_coordinates(tree, x, y, z, bind)


def center_coordinates_at_root(
    tree: _Tree,
    bind: bool = True,
) -> ndarray | None:
    """Translate a tree so that its root node lies at the origin.

    This is equivalent to subtracting the root coordinate from every node
    position. When the tree has multiple roots, the first root returned by
    :func:`get_root_coordinate` is used.

    Parameters
    ----------
    tree : _Tree
        Tree to center.
    bind : bool, optional
        If True, write centered coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Centered (N, 3) coordinates when bind=False; None when bind=True.

    Notes
    -----
    Node radii are not modified by this operation.
    """
    rc = get_root_coordinate(tree, SoA=False).ravel()
    return recenter_coordinates(
        tree,
        center=(float(rc[0]), float(rc[1]), float(rc[2])),
        bind=bind,
    )


def recenter_coordinates(
    tree: _Tree,
    center: tuple | ndarray,
    bind: bool = True,
) -> ndarray | None:
    """Translate a tree so that a specified point becomes the origin.

    Subtracts *center* from every node coordinate. This is equivalent to
    :func:`translate_coordinates` with displacement ``(-cx, -cy, -cz)``.

    Parameters
    ----------
    tree : _Tree
        Tree to recenter.
    center : tuple | ndarray
        Point ``(cx, cy, cz)`` to move to the origin.
    bind : bool, optional
        If True, write recentered coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Recentered (N, 3) coordinates when bind=False; None when bind=True.

    Notes
    -----
    Node radii are not modified by this operation.
    """
    x, y, z = get_node_coordinates(tree, SoA=True)
    cx, cy, cz = center
    x, y, z = recenter(x, y, z, cx, cy, cz)
    return _apply_node_coordinates(tree, x, y, z, bind)


def rotate_coordinates(
    tree: _Tree,
    axis: tuple | ndarray,
    angle: float,
    assume_normalized: bool = False,
    bind: bool = True,
) -> ndarray | None:
    """Rotate all node coordinates about an axis through the origin.

    Uses Rodrigues' rotation formula via :func:`rotate`. The rotation angle
    is in radians. Node radii are not modified.

    Parameters
    ----------
    tree : _Tree
        Tree to rotate.
    axis : tuple | ndarray
        Rotation axis ``(ax, ay, az)``.
    angle : float
        Rotation angle in radians.
    assume_normalized : bool, optional
        If True, treat *axis* as a unit vector. By default False.
    bind : bool, optional
        If True, write rotated coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Rotated (N, 3) coordinates when bind=False; None when bind=True.

    Notes
    -----
    This rotates about the global origin. To rotate about the tree centroid,
    root, or another point, use :func:`rotate_coordinates_about`.
    """
    x, y, z = get_node_coordinates(tree, SoA=True)
    ax, ay, az = axis
    x, y, z = rotate(x, y, z, ax, ay, az, angle, assume_normalized)
    return _apply_node_coordinates(tree, x, y, z, bind)


def rotate_coordinates_about(
    tree: _Tree,
    axis: tuple | ndarray,
    angle: float,
    center: tuple | ndarray | None = None,
    center_mode: CenterMode = "centroid",
    assume_normalized: bool = False,
    bind: bool = True,
) -> ndarray | None:
    """Rotate all node coordinates about an axis passing through a point.

    Translates coordinates so that *center* lies at the origin, applies the
    rotation, then translates back. Implemented as recenter → rotate →
    translate. The rotation angle is in radians. Node radii are not modified.

    Parameters
    ----------
    tree : _Tree
        Tree to rotate.
    axis : tuple | ndarray
        Rotation axis ``(ax, ay, az)``.
    angle : float
        Rotation angle in radians.
    center : tuple | ndarray | None, optional
        Point ``(cx, cy, cz)`` through which the rotation axis passes. When
        None, *center_mode* determines the centre. By default None.
    center_mode : {"centroid", "root"}, optional
        Centre to use when *center* is None. ``"centroid"`` uses the mean
        node position; ``"root"`` uses the root node coordinate.
        By default ``"centroid"``.
    assume_normalized : bool, optional
        If True, treat *axis* as a unit vector. By default False.
    bind : bool, optional
        If True, write rotated coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Rotated (N, 3) coordinates when bind=False; None when bind=True.

    Raises
    ------
    ValueError
        If *center_mode* is not ``"centroid"`` or ``"root"``.
    """
    if center_mode not in ("centroid", "root"):
        raise ValueError(
            f"center_mode must be 'centroid' or 'root', not {center_mode!r}"
        )

    x, y, z = get_node_coordinates(tree, SoA=True)
    cx, cy, cz = _resolve_transform_center(tree, x, y, z, center, center_mode)
    ax, ay, az = axis
    x, y, z = recenter(x, y, z, cx, cy, cz)
    x, y, z = rotate(x, y, z, ax, ay, az, angle, assume_normalized)
    x, y, z = translate(x, y, z, cx, cy, cz)
    return _apply_node_coordinates(tree, x, y, z, bind)


def scale_coordinates(
    tree: _Tree,
    factor: float,
    center: tuple | ndarray | None = None,
    center_mode: CenterMode = "centroid",
    scale_radii: bool = True,
    bind: bool = True,
) -> ndarray | None:
    """Uniformly scale all node coordinates about a point.

    Applies the same scale factor along each coordinate axis using
    :func:`scale_about`. When *scale_radii* is True and the graph has a
    ``radius`` vertex property, node radii are multiplied by *factor*.

    Parameters
    ----------
    tree : _Tree
        Tree to scale.
    factor : float
        Uniform scale factor applied along x, y, and z.
    center : tuple | ndarray | None, optional
        Point ``(cx, cy, cz)`` that remains fixed during scaling. When
        None, *center_mode* determines the centre. By default None.
    center_mode : {"centroid", "root"}, optional
        Centre to use when *center* is None. ``"centroid"`` uses the mean
        node position; ``"root"`` uses the root node coordinate.
        By default ``"centroid"``.
    scale_radii : bool, optional
        If True, multiply node radii by *factor* when the ``radius`` vertex
        property is present. By default True.
    bind : bool, optional
        If True, write scaled coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Scaled (N, 3) coordinates when bind=False; None when bind=True.

    Raises
    ------
    ValueError
        If *center_mode* is not ``"centroid"`` or ``"root"``.

    Notes
    -----
    A scale factor of ``1`` leaves coordinates unchanged. A scale factor of
    ``0`` collapses every coordinate onto *center*. Negative scale factors
    reflect coordinates through *center* while scaling their magnitude.
    """
    if center_mode not in ("centroid", "root"):
        raise ValueError(
            f"center_mode must be 'centroid' or 'root', not {center_mode!r}"
        )

    x, y, z = get_node_coordinates(tree, SoA=True)
    cx, cy, cz = _resolve_transform_center(tree, x, y, z, center, center_mode)
    x, y, z = scale_about(x, y, z, factor, factor, factor, cx, cy, cz)
    out = _apply_node_coordinates(tree, x, y, z, bind)
    if bind and scale_radii:
        _scale_tree_radii(tree, factor)
    return out


def scale_coordinates_about(
    tree: _Tree,
    scale: float | int | tuple | ndarray,
    center: tuple | ndarray | None = None,
    center_mode: CenterMode = "centroid",
    scale_radii: bool = False,
    bind: bool = True,
) -> ndarray | None:
    """Scale node coordinates independently along each axis about a point.

    Applies anisotropic scaling using :func:`scale_about`. The supplied centre
    remains fixed during the transformation.

    Parameters
    ----------
    tree : _Tree
        Tree to scale.
    scale : float | int | tuple | ndarray
        Scale factor(s) along x, y, and z. A scalar is applied uniformly to
        all three axes. Otherwise supply ``(sx, sy, sz)``.
    center : tuple | ndarray | None, optional
        Point ``(cx, cy, cz)`` that remains fixed during scaling. When
        None, *center_mode* determines the centre. By default None.
    center_mode : {"centroid", "root"}, optional
        Centre to use when *center* is None. ``"centroid"`` uses the mean
        node position; ``"root"`` uses the root node coordinate.
        By default ``"centroid"``.
    scale_radii : bool, optional
        If True, multiply node radii by the mean of ``(sx, sy, sz)`` when
        the ``radius`` vertex property is present. By default False because
        anisotropic scaling has no single natural radius scale factor.
    bind : bool, optional
        If True, write scaled coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Scaled (N, 3) coordinates when bind=False; None when bind=True.

    Raises
    ------
    ValueError
        If *center_mode* is not ``"centroid"`` or ``"root"``.

    Notes
    -----
    For uniform scaling with automatic radius rescaling, prefer
    :func:`scale_coordinates`.
    """
    if center_mode not in ("centroid", "root"):
        raise ValueError(
            f"center_mode must be 'centroid' or 'root', not {center_mode!r}"
        )

    if isinstance(scale, (float, int)):
        sx = sy = sz = float(scale)
    else:
        sx, sy, sz = scale

    x, y, z = get_node_coordinates(tree, SoA=True)
    cx, cy, cz = _resolve_transform_center(tree, x, y, z, center, center_mode)
    x, y, z = scale_about(x, y, z, sx, sy, sz, cx, cy, cz)
    out = _apply_node_coordinates(tree, x, y, z, bind)
    if bind and scale_radii:
        _scale_tree_radii(tree, (sx + sy + sz) / 3.0)
    return out


def align_coordinates_to_vector(
    tree: _Tree,
    target: tuple | ndarray = (0.0, 1.0, 0.0),
    source: SourceVector = "pc1",
    robust: bool = True,
    recenter: bool = True,
    bind: bool = True,
) -> ndarray | None:
    """Rotate a tree so a source direction aligns with a target direction.

    Computes the minimum rotation that maps *source* onto *target*, then
    applies it to all node coordinates. When *source* is ``"pc1"``, the
    source direction is the first principal component of the node
    coordinates.

    Parameters
    ----------
    tree : _Tree
        Tree to align.
    target : tuple | ndarray, optional
        Target unit direction ``(tx, ty, tz)``. By default (0.0, 1.0, 0.0).
    source : {"pc1"} | tuple | ndarray, optional
        Direction to rotate from. ``"pc1"`` uses the first PCA eigenvector
        of the (optionally recentered) node coordinates. Otherwise supply
        an explicit direction vector. By default ``"pc1"``.
    robust : bool, optional
        Use robust PCA when *source* is ``"pc1"``. By default True.
    recenter : bool, optional
        If True, translate coordinates so their centroid lies at the origin
        before computing the rotation. By default True.
    bind : bool, optional
        If True, write rotated coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Rotated (N, 3) coordinates when bind=False; None when bind=True.

    Raises
    ------
    ValueError
        If *source* is not ``"pc1"`` or a three-component vector.

    Notes
    -----
    Unlike :func:`align_coordinates`, only one direction is constrained. The
    rotation about the aligned axis remains free. Node radii are not
    modified by this operation.
    """
    x, y, z = get_node_coordinates(tree, SoA=True)
    x, y, z = _maybe_recenter_at_centroid(x, y, z, recenter)

    if source == "pc1":
        _, evecs = eig_decomp(x, y, z, robust=robust)
        v1 = evecs[:, 0]
    else:
        v1 = asarray(source, dtype=float).ravel()
        if v1.shape != (3,):
            raise ValueError(
                f"source must be 'pc1' or a three-component vector, not {source!r}"
            )

    tx, ty, tz = target
    axis, angle = minimum_rotation_to_align(*v1, tx, ty, tz)
    x, y, z = rotate(x, y, z, *axis, angle, assume_normalized=True)
    return _apply_node_coordinates(tree, x, y, z, bind)


def scale_coordinates_along_pca(
    tree: _Tree,
    scale: float | int | tuple | ndarray,
    robust: bool = True,
    recenter: bool = True,
    scale_radii: bool = False,
    bind: bool = True,
) -> ndarray | None:
    """Scale node coordinates along the tree PCA axes.

    Decomposes each node coordinate into its projections onto the first
    three principal components, then rescales those components
    independently. Uses :func:`scale_along_basis`.

    Parameters
    ----------
    tree : _Tree
        Tree to scale.
    scale : float | int | tuple | ndarray
        Scale factor(s) along PC1, PC2, and PC3. A scalar is applied
        uniformly to all three directions. Otherwise supply
        ``(s1, s2, s3)``.
    robust : bool, optional
        Use robust PCA when computing the basis. By default True.
    recenter : bool, optional
        If True, translate coordinates so their centroid lies at the origin
        before computing PCA and scaling. By default True.
    scale_radii : bool, optional
        If True, multiply node radii by the mean of the three scale factors
        when the ``radius`` vertex property is present. By default False
        because anisotropic scaling has no single natural radius scale
        factor.
    bind : bool, optional
        If True, write scaled coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Scaled (N, 3) coordinates when bind=False; None when bind=True.

    Notes
    -----
    The PCA basis is computed from the current node coordinates. If the
    tree has already been aligned with :func:`align_coordinates`, this operation
    scales along the same canonical axes. For uniform scaling with automatic
    radius rescaling, prefer :func:`scale_coordinates`.
    """
    sx, sy, sz = _resolve_scale_factors(scale)

    x, y, z = get_node_coordinates(tree, SoA=True)
    x, y, z = _maybe_recenter_at_centroid(x, y, z, recenter)

    _, evecs = eig_decomp(x, y, z, robust=robust)
    b1 = evecs[:, 0]
    b2 = evecs[:, 1]
    b3 = evecs[:, 2]

    x, y, z = scale_along_basis(x, y, z, (sx, sy, sz), b1, b2, b3)
    out = _apply_node_coordinates(tree, x, y, z, bind)
    if bind and scale_radii:
        _scale_tree_radii(tree, (sx + sy + sz) / 3.0)
    return out


def apply_rotation_steps_to_coordinates(
    tree: _Tree,
    step1: tuple,
    step2: tuple,
    recenter: bool = False,
    bind: bool = True,
) -> ndarray | None:
    """Apply two precomputed axis-angle rotations to all node coordinates.

    Applies the rotations in the order *step1* followed by *step2*, using
    :func:`apply_rotation_steps`. This is the same two-step rotation
    machinery used internally by :func:`align_coordinates`.

    Parameters
    ----------
    tree : _Tree
        Tree to rotate.
    step1 : tuple
        First rotation specified as ``(axis, angle)``, where ``axis`` is a
        three-component unit vector and ``angle`` is in radians.
    step2 : tuple
        Second rotation specified as ``(axis, angle)``, where ``axis`` is a
        three-component unit vector and ``angle`` is in radians.
    recenter : bool, optional
        If True, translate coordinates so their centroid lies at the origin
        before applying the rotations. By default False.
    bind : bool, optional
        If True, write rotated coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Rotated (N, 3) coordinates when bind=False; None when bind=True.

    Notes
    -----
    The rotation axes in *step1* and *step2* are assumed to be normalized.
    Node radii are not modified by this operation.

    See Also
    --------
    align_coordinates : Computes and applies PCA alignment steps automatically.
    compute_alignment_rotation : Computes the two rotation steps for a basis
        alignment.
    """
    x, y, z = get_node_coordinates(tree, SoA=True)
    x, y, z = _maybe_recenter_at_centroid(x, y, z, recenter)
    x, y, z = apply_rotation_steps(x, y, z, step1, step2)
    return _apply_node_coordinates(tree, x, y, z, bind)
