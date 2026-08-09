"""Functions for global coordinate transformation of forests."""

from typing import Tuple
from numpy import array, ndarray, asarray

from scipy.spatial import ConvexHull

from ...core import _Forest
from ...utils.graph_utils.gt_properties import _set_coords_prop
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
from ..tree_graphs.coordinates import get_root_coordinate
from ..tree_graphs.tree_transformations import (
    CenterMode,
    SourceVector,
    _maybe_recenter_at_centroid,
    _resolve_scale_factors,
)
from .utils import _split_array_vertex, _split_inds


def _set_global_coords(
    forest: _Forest, x: ndarray, y: ndarray, z: ndarray
) -> None:
    """Write pooled coordinate arrays back to individual forest tree graphs.

    Parameters
    ----------
    forest : _Forest
        Target forest.
    x, y, z : ndarray
        Pooled node coordinates for all trees concatenated along vertices.

    Returns
    -------
    None
    """
    s_index = _split_inds(forest)

    x = _split_array_vertex(x, s_index)
    y = _split_array_vertex(y, s_index)
    z = _split_array_vertex(z, s_index)

    for i in range(len(forest)):
        c = array([x[i], y[i], z[i]])
        _set_coords_prop(forest[i].graph, c)


def _get_pooled_coordinates(
    forest: _Forest,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Return pooled node coordinates as separate x, y, and z arrays."""
    coords = forest.get_node_coordinates(
        SoA=False, merge_axis=0, show_progress=False
    )
    return coords[:, 0], coords[:, 1], coords[:, 2]


def _return_forest_coordinates(
    forest: _Forest,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    bind: bool,
    unpack: bool,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Write or return transformed pooled coordinates."""
    if bind:
        _set_global_coords(forest, x, y, z)
        return None

    if unpack:
        s_index = _split_inds(forest)
        return (
            _split_array_vertex(x, s_index),
            _split_array_vertex(y, s_index),
            _split_array_vertex(z, s_index),
        )

    return x, y, z


def _scale_forest_radii(forest: _Forest, factor: float) -> None:
    """Scale cached node radii on every tree by a multiplicative factor."""
    for tree in forest:
        if "radius" in tree.graph.vp:
            tree.graph.vp["radius"].a *= factor


def _resolve_forest_transform_center(
    forest: _Forest,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    center: tuple | ndarray | None,
    center_mode: CenterMode,
) -> Tuple[float, float, float]:
    """Resolve the centre point used by global forest transforms."""
    if center is not None:
        cx, cy, cz = center
        return float(cx), float(cy), float(cz)

    if center_mode == "root":
        roots = array(
            [
                get_root_coordinate(tree, SoA=False).ravel()
                for tree in forest
            ]
        )
        cx, cy, cz = roots.mean(axis=0)
        return float(cx), float(cy), float(cz)

    _, _, _, cx, cy, cz = center_at_centroid(x, y, z)
    return cx, cy, cz


def _validate_center_mode(center_mode: CenterMode) -> None:
    if center_mode not in ("centroid", "root"):
        raise ValueError(
            f"center_mode must be 'centroid' or 'root', not {center_mode!r}"
        )


def align_forest(
    forest: _Forest,
    robust: bool = True,
    b1: tuple = (0.0, 1.0, 0.0),
    b2: tuple = (1.0, 0.0, 0.0),
    b3: tuple = (0.0, 0.0, 0.1),
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """PCA-align all trees in a forest to canonical axes.

    Recenters pooled node coordinates at their global centroid, then rotates
    them so the first three principal components align with *b1*, *b2*, and
    *b3*. Optionally writes rotated coordinates back to each tree graph.

    Parameters
    ----------
    forest : _Forest
        Forest to align.
    robust : bool, optional
        Use robust PCA. By default True.
    b1 : tuple, optional
        Target unit vector for PC1. By default (0.0, 1.0, 0.0).
    b2 : tuple, optional
        Target unit vector for PC2. By default (1.0, 0.0, 0.0).
    b3 : tuple, optional
        Target unit vector for PC3. By default (0.0, 0.0, 0.1).
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write rotated coordinates to forest graphs and return None.
        By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled rotated ``(x, y, z)`` arrays when bind=False; None when
        bind=True. When *unpack* is True, each array contains one segment
        per tree.

    Notes
    -----
    All trees are transformed using one global PCA computed from their pooled
    node coordinates. Node radii are not modified by this operation.
    """
    x, y, z = _get_pooled_coordinates(forest)
    x, y, z, _, _, _ = center_at_centroid(x, y, z)

    _, evecs = eig_decomp(x, y, z, robust=robust)

    ev1 = evecs[:, 0]
    ev2 = evecs[:, 1]
    ev3 = evecs[:, 2]

    step1, step2 = compute_alignment_rotation(ev1, ev2, ev3, b1, b2, b3)
    x, y, z = apply_rotation_steps(x, y, z, step1, step2)

    return _return_forest_coordinates(forest, x, y, z, bind, unpack)


def translate_forest(
    forest: _Forest,
    dx: float,
    dy: float,
    dz: float,
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Translate all node coordinates in a forest by a fixed displacement.

    Applies the same rigid translation to every node in every tree. Node
    radii and cached geometric properties are unchanged.

    Parameters
    ----------
    forest : _Forest
        Forest to translate.
    dx : float
        Displacement along the x axis.
    dy : float
        Displacement along the y axis.
    dz : float
        Displacement along the z axis.
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write translated coordinates to forest graphs and return
        None. By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled translated ``(x, y, z)`` arrays when bind=False; None when
        bind=True.
    """
    x, y, z = _get_pooled_coordinates(forest)
    x, y, z = translate(x, y, z, dx, dy, dz)
    return _return_forest_coordinates(forest, x, y, z, bind, unpack)


def center_forest_at_centroid(
    forest: _Forest,
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Translate a forest so the global node centroid lies at the origin.

    The centroid is computed from all node coordinates pooled across trees.
    This is equivalent to subtracting the global centroid from every node
    coordinate.

    Parameters
    ----------
    forest : _Forest
        Forest to center.
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write centered coordinates to forest graphs and return None.
        By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled centered ``(x, y, z)`` arrays when bind=False; None when
        bind=True.

    Notes
    -----
    Node radii are not modified by this operation. This is the same
    recentering step performed internally by :func:`align_forest` before PCA.
    """
    x, y, z = _get_pooled_coordinates(forest)
    x, y, z, _, _, _ = center_at_centroid(x, y, z)
    return _return_forest_coordinates(forest, x, y, z, bind, unpack)


def recenter_forest(
    forest: _Forest,
    center: tuple | ndarray,
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Translate a forest so a specified point becomes the origin.

    Subtracts *center* from every node coordinate in every tree. This is
    equivalent to :func:`translate_forest` with displacement
    ``(-cx, -cy, -cz)``.

    Parameters
    ----------
    forest : _Forest
        Forest to recenter.
    center : tuple | ndarray
        Point ``(cx, cy, cz)`` to move to the origin.
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write recentered coordinates to forest graphs and return
        None. By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled recentered ``(x, y, z)`` arrays when bind=False; None when
        bind=True.

    Notes
    -----
    Node radii are not modified by this operation.
    """
    x, y, z = _get_pooled_coordinates(forest)
    cx, cy, cz = center
    x, y, z = recenter(x, y, z, cx, cy, cz)
    return _return_forest_coordinates(forest, x, y, z, bind, unpack)


def rotate_forest(
    forest: _Forest,
    axis: tuple | ndarray,
    angle: float,
    assume_normalized: bool = False,
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Rotate all node coordinates in a forest about an axis through the origin.

    Uses Rodrigues' rotation formula via :func:`rotate`. The rotation angle
    is in radians. Node radii are not modified.

    Parameters
    ----------
    forest : _Forest
        Forest to rotate.
    axis : tuple | ndarray
        Rotation axis ``(ax, ay, az)``.
    angle : float
        Rotation angle in radians.
    assume_normalized : bool, optional
        If True, treat *axis* as a unit vector. By default False.
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write rotated coordinates to forest graphs and return None.
        By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled rotated ``(x, y, z)`` arrays when bind=False; None when
        bind=True.

    Notes
    -----
    This rotates about the global origin. To rotate about the forest centroid,
    mean root position, or another point, use :func:`rotate_forest_about`.
    """
    x, y, z = _get_pooled_coordinates(forest)
    ax, ay, az = axis
    x, y, z = rotate(x, y, z, ax, ay, az, angle, assume_normalized)
    return _return_forest_coordinates(forest, x, y, z, bind, unpack)


def rotate_forest_about(
    forest: _Forest,
    axis: tuple | ndarray,
    angle: float,
    center: tuple | ndarray | None = None,
    center_mode: CenterMode = "centroid",
    assume_normalized: bool = False,
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Rotate all node coordinates in a forest about an axis through a point.

    Translates coordinates so that *center* lies at the origin, applies the
    rotation, then translates back. Implemented as recenter → rotate →
    translate. The rotation angle is in radians. Node radii are not modified.

    Parameters
    ----------
    forest : _Forest
        Forest to rotate.
    axis : tuple | ndarray
        Rotation axis ``(ax, ay, az)``.
    angle : float
        Rotation angle in radians.
    center : tuple | ndarray | None, optional
        Point ``(cx, cy, cz)`` through which the rotation axis passes. When
        None, *center_mode* determines the centre. By default None.
    center_mode : {"centroid", "root"}, optional
        Centre to use when *center* is None. ``"centroid"`` uses the global
        mean node position across the forest; ``"root"`` uses the mean root
        coordinate across trees. By default ``"centroid"``.
    assume_normalized : bool, optional
        If True, treat *axis* as a unit vector. By default False.
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write rotated coordinates to forest graphs and return None.
        By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled rotated ``(x, y, z)`` arrays when bind=False; None when
        bind=True.

    Raises
    ------
    ValueError
        If *center_mode* is not ``"centroid"`` or ``"root"``.
    """
    _validate_center_mode(center_mode)

    x, y, z = _get_pooled_coordinates(forest)
    cx, cy, cz = _resolve_forest_transform_center(
        forest, x, y, z, center, center_mode
    )
    ax, ay, az = axis
    x, y, z = recenter(x, y, z, cx, cy, cz)
    x, y, z = rotate(x, y, z, ax, ay, az, angle, assume_normalized)
    x, y, z = translate(x, y, z, cx, cy, cz)
    return _return_forest_coordinates(forest, x, y, z, bind, unpack)


def scale_forest(
    forest: _Forest,
    factor: float,
    center: tuple | ndarray | None = None,
    center_mode: CenterMode = "centroid",
    scale_radii: bool = True,
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Uniformly scale all node coordinates in a forest about a point.

    Applies the same scale factor along each coordinate axis using
    :func:`scale_about`. When *scale_radii* is True and each tree has a
    ``radius`` vertex property, node radii are multiplied by *factor*.

    Parameters
    ----------
    forest : _Forest
        Forest to scale.
    factor : float
        Uniform scale factor applied along x, y, and z.
    center : tuple | ndarray | None, optional
        Point ``(cx, cy, cz)`` that remains fixed during scaling. When
        None, *center_mode* determines the centre. By default None.
    center_mode : {"centroid", "root"}, optional
        Centre to use when *center* is None. ``"centroid"`` uses the global
        mean node position across the forest; ``"root"`` uses the mean root
        coordinate across trees. By default ``"centroid"``.
    scale_radii : bool, optional
        If True, multiply node radii by *factor* when the ``radius`` vertex
        property is present. By default True.
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write scaled coordinates to forest graphs and return None.
        By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled scaled ``(x, y, z)`` arrays when bind=False; None when
        bind=True.

    Raises
    ------
    ValueError
        If *center_mode* is not ``"centroid"`` or ``"root"``.
    """
    _validate_center_mode(center_mode)

    x, y, z = _get_pooled_coordinates(forest)
    cx, cy, cz = _resolve_forest_transform_center(
        forest, x, y, z, center, center_mode
    )
    x, y, z = scale_about(x, y, z, factor, factor, factor, cx, cy, cz)
    out = _return_forest_coordinates(forest, x, y, z, bind, unpack)
    if bind and scale_radii:
        _scale_forest_radii(forest, factor)
    return out


def scale_forest_about(
    forest: _Forest,
    scale: float | int | tuple | ndarray,
    center: tuple | ndarray | None = None,
    center_mode: CenterMode = "centroid",
    scale_radii: bool = False,
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Scale node coordinates in a forest independently along each axis.

    Applies anisotropic scaling using :func:`scale_about`. The supplied
    centre remains fixed during the transformation.

    Parameters
    ----------
    forest : _Forest
        Forest to scale.
    scale : float | int | tuple | ndarray
        Scale factor(s) along x, y, and z. A scalar is applied uniformly to
        all three axes. Otherwise supply ``(sx, sy, sz)``.
    center : tuple | ndarray | None, optional
        Point ``(cx, cy, cz)`` that remains fixed during scaling. When
        None, *center_mode* determines the centre. By default None.
    center_mode : {"centroid", "root"}, optional
        Centre to use when *center* is None. ``"centroid"`` uses the global
        mean node position across the forest; ``"root"`` uses the mean root
        coordinate across trees. By default ``"centroid"``.
    scale_radii : bool, optional
        If True, multiply node radii by the mean of ``(sx, sy, sz)`` when
        the ``radius`` vertex property is present. By default False because
        anisotropic scaling has no single natural radius scale factor.
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write scaled coordinates to forest graphs and return None.
        By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled scaled ``(x, y, z)`` arrays when bind=False; None when
        bind=True.

    Raises
    ------
    ValueError
        If *center_mode* is not ``"centroid"`` or ``"root"``.

    Notes
    -----
    For uniform scaling with automatic radius rescaling, prefer
    :func:`scale_forest`.
    """
    _validate_center_mode(center_mode)

    sx, sy, sz = _resolve_scale_factors(scale)

    x, y, z = _get_pooled_coordinates(forest)
    cx, cy, cz = _resolve_forest_transform_center(
        forest, x, y, z, center, center_mode
    )
    x, y, z = scale_about(x, y, z, sx, sy, sz, cx, cy, cz)
    out = _return_forest_coordinates(forest, x, y, z, bind, unpack)
    if bind and scale_radii:
        _scale_forest_radii(forest, (sx + sy + sz) / 3.0)
    return out


def align_forest_to_vector(
    forest: _Forest,
    target: tuple | ndarray = (0.0, 1.0, 0.0),
    source: SourceVector = "pc1",
    robust: bool = True,
    recenter: bool = True,
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Rotate a forest so a source direction aligns with a target direction.

    Computes the minimum rotation that maps *source* onto *target* using
    pooled node coordinates, then applies it to every node in every tree.
    When *source* is ``"pc1"``, the source direction is the first principal
    component of the pooled coordinates.

    Parameters
    ----------
    forest : _Forest
        Forest to align.
    target : tuple | ndarray, optional
        Target unit direction ``(tx, ty, tz)``. By default (0.0, 1.0, 0.0).
    source : {"pc1"} | tuple | ndarray, optional
        Direction to rotate from. ``"pc1"`` uses the first PCA eigenvector
        of the (optionally recentered) pooled coordinates. Otherwise supply
        an explicit direction vector. By default ``"pc1"``.
    robust : bool, optional
        Use robust PCA when *source* is ``"pc1"``. By default True.
    recenter : bool, optional
        If True, translate coordinates so their global centroid lies at the
        origin before computing the rotation. By default True.
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write rotated coordinates to forest graphs and return None.
        By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled rotated ``(x, y, z)`` arrays when bind=False; None when
        bind=True.

    Raises
    ------
    ValueError
        If *source* is not ``"pc1"`` or a three-component vector.

    Notes
    -----
    Unlike :func:`align_forest`, only one direction is constrained. The
    rotation about the aligned axis remains free. Node radii are not modified
    by this operation.
    """
    x, y, z = _get_pooled_coordinates(forest)
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
    return _return_forest_coordinates(forest, x, y, z, bind, unpack)


def scale_forest_along_pca(
    forest: _Forest,
    scale: float | int | tuple | ndarray,
    robust: bool = True,
    recenter: bool = True,
    scale_radii: bool = False,
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Scale node coordinates in a forest along global PCA axes.

    Decomposes each pooled node coordinate into its projections onto the
    first three principal components of the forest, then rescales those
    components independently. Uses :func:`scale_along_basis`.

    Parameters
    ----------
    forest : _Forest
        Forest to scale.
    scale : float | int | tuple | ndarray
        Scale factor(s) along PC1, PC2, and PC3. A scalar is applied
        uniformly to all three directions. Otherwise supply
        ``(s1, s2, s3)``.
    robust : bool, optional
        Use robust PCA when computing the basis. By default True.
    recenter : bool, optional
        If True, translate coordinates so their global centroid lies at the
        origin before computing PCA and scaling. By default True.
    scale_radii : bool, optional
        If True, multiply node radii by the mean of the three scale factors
        when the ``radius`` vertex property is present. By default False
        because anisotropic scaling has no single natural radius scale factor.
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write scaled coordinates to forest graphs and return None.
        By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled scaled ``(x, y, z)`` arrays when bind=False; None when
        bind=True.

    Notes
    -----
    The PCA basis is computed from all node coordinates pooled across the
    forest. For uniform scaling with automatic radius rescaling, prefer
    :func:`scale_forest`.
    """
    sx, sy, sz = _resolve_scale_factors(scale)

    x, y, z = _get_pooled_coordinates(forest)
    x, y, z = _maybe_recenter_at_centroid(x, y, z, recenter)

    _, evecs = eig_decomp(x, y, z, robust=robust)
    b1 = evecs[:, 0]
    b2 = evecs[:, 1]
    b3 = evecs[:, 2]

    x, y, z = scale_along_basis(x, y, z, (sx, sy, sz), b1, b2, b3)
    out = _return_forest_coordinates(forest, x, y, z, bind, unpack)
    if bind and scale_radii:
        _scale_forest_radii(forest, (sx + sy + sz) / 3.0)
    return out


def apply_rotation_steps_to_forest(
    forest: _Forest,
    step1: tuple,
    step2: tuple,
    recenter: bool = False,
    unpack: bool = False,
    bind: bool = True,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Apply two precomputed axis-angle rotations to all forest node coordinates.

    Applies the rotations in the order *step1* followed by *step2* to pooled
    node coordinates using :func:`apply_rotation_steps`. This is the same
    two-step rotation machinery used internally by :func:`align_forest`.

    Parameters
    ----------
    forest : _Forest
        Forest to rotate.
    step1 : tuple
        First rotation specified as ``(axis, angle)``, where ``axis`` is a
        three-component unit vector and ``angle`` is in radians.
    step2 : tuple
        Second rotation specified as ``(axis, angle)``, where ``axis`` is a
        three-component unit vector and ``angle`` is in radians.
    recenter : bool, optional
        If True, translate coordinates so their global centroid lies at the
        origin before applying the rotations. By default False.
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write rotated coordinates to forest graphs and return None.
        By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Pooled rotated ``(x, y, z)`` arrays when bind=False; None when
        bind=True.

    Notes
    -----
    The rotation axes in *step1* and *step2* are assumed to be normalized.
    Node radii are not modified by this operation.

    See Also
    --------
    align_forest : Computes and applies PCA alignment steps automatically.
    compute_alignment_rotation : Computes the two rotation steps for a basis
        alignment.
    """
    x, y, z = _get_pooled_coordinates(forest)
    x, y, z = _maybe_recenter_at_centroid(x, y, z, recenter)
    x, y, z = apply_rotation_steps(x, y, z, step1, step2)
    return _return_forest_coordinates(forest, x, y, z, bind, unpack)


def forest_pca(forest: _Forest, robust: bool = False, norm: bool = True) -> Tuple[ndarray, ndarray]:
    """Perform PCA on pooled node coordinates across a forest.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    robust : bool, optional
        Use robust covariance estimation. By default False.
    norm : bool, optional
        Normalize eigenvalues to sum to 1. By default True.

    Returns
    -------
    tuple[ndarray, ndarray]
        ``(evals, evecs)`` from :func:`eig_decomp`: eigenvalues in descending
        order and corresponding eigenvectors as columns.
    """
    x, y, z = forest.get_node_coordinates(
        SoA=True, merge_axis=1, show_progress=False
    )
    return eig_decomp(x, y, z, robust=robust, norm=norm)


def get_forest_convex_hull(forest: _Forest) -> ConvexHull:
    """Build a convex hull around pooled node coordinates in a forest.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.

    Returns
    -------
    ConvexHull
        SciPy convex hull of all node coordinates.
    """
    return ConvexHull(
        forest.get_node_coordinates(
            SoA=False, merge_axis=0, show_progress=False
        )
    )


def get_forest_convex_hull_volume(forest: _Forest) -> float:
    """Return the volume enclosed by the convex hull of forest node coordinates.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.

    Returns
    -------
    float
        Convex hull volume in cubic tree units.
    """
    return get_forest_convex_hull(forest).volume
