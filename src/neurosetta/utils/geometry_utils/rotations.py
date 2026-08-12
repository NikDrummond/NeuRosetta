""" """

from numba import njit
from numpy import array, asarray, cos, empty, float64, isclose, pi, sin, sqrt
from numpy.linalg import norm

from ._validation import _broadcast_vectors, _check_scalar, _check_vector_broadcast
from .algebra import cross_scalar, dot_scalar, normalize_scalar
from .angles import angle_scalar

_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")


@njit(**_JIT)
def rotate_scalar(
    x,
    y,
    z,
    ax,
    ay,
    az,
    rotation_angle,
    assume_normalized=False,
):
    """
    Rotate a vector about an axis using Rodrigues' rotation formula.

    Angle is in radians.
    """

    if not assume_normalized:
        ax, ay, az = normalize_scalar(ax, ay, az)

    c = cos(rotation_angle)
    s = sin(rotation_angle)

    #
    # cross(axis, vector)
    #

    cx, cy, cz = cross_scalar(
        ax,
        ay,
        az,
        x,
        y,
        z,
    )

    #
    # dot(axis, vector)
    #

    d = dot_scalar(
        ax,
        ay,
        az,
        x,
        y,
        z,
    )

    return (
        c * x + s * cx + (1.0 - c) * d * ax,
        c * y + s * cy + (1.0 - c) * d * ay,
        c * z + s * cz + (1.0 - c) * d * az,
    )


@njit(**_JIT)
def rotate_about_scalar(
    x,
    y,
    z,
    ax,
    ay,
    az,
    angle,
    cx,
    cy,
    cz,
    assume_normalized=False,
):
    """
    Rotate a point about an axis passing through a point.
    """

    # Translate so the rotation centre is the origin.
    rx = x - cx
    ry = y - cy
    rz = z - cz

    # Rotate using the existing rotation kernel.
    rx, ry, rz = rotate_scalar(
        rx,
        ry,
        rz,
        ax,
        ay,
        az,
        angle,
        assume_normalized,
    )

    # Translate back.
    return (
        rx + cx,
        ry + cy,
        rz + cz,
    )


@njit(**_JIT)
def rotate_xyz(
    x,
    y,
    z,
    ax,
    ay,
    az,
    rotation_angle,
    assume_normalized=False,
):

    n = x.size

    xr = empty(n)
    yr = empty(n)
    zr = empty(n)

    for i in range(n):
        xr[i], yr[i], zr[i] = rotate_scalar(
            x[i],
            y[i],
            z[i],
            ax[i],
            ay[i],
            az[i],
            rotation_angle,
            assume_normalized,
        )

    return xr, yr, zr


@njit(**_JIT)
def rotate_about_xyz(
    x,
    y,
    z,
    ax,
    ay,
    az,
    angle,
    cx,
    cy,
    cz,
    assume_normalized=False,
):
    """
    Rotate points about an axis passing through a point.
    """

    n = x.size

    xo = empty(n, dtype=float64)
    yo = empty(n, dtype=float64)
    zo = empty(n, dtype=float64)

    for i in range(n):
        xo[i], yo[i], zo[i] = rotate_about_scalar(
            x[i],
            y[i],
            z[i],
            ax,
            ay,
            az,
            angle,
            cx,
            cy,
            cz,
            assume_normalized,
        )

    return xo, yo, zo


### Python functions


def rotate(x, y, z, ax, ay, az, rotation_angle, assume_normalized=False):
    """
    Rotate a 3D vector about an axis.

    Uses Rodrigues' rotation formula. The rotation angle is in radians.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of the vector to rotate.
    ax, ay, az : float or ndarray
        Components of the rotation axis. A scalar axis may be broadcast
        against an array vector.
    angle : float
        Rotation angle in radians.
    assume_normalized : bool, default False
        If True, treat ``(ax, ay, az)`` as a unit vector.

    Returns
    -------
    xr, yr, zr : float or ndarray
        Components of the rotated vector.
    """

    if _check_vector_broadcast(x, y, z, ax, ay, az):
        x, y, z, ax, ay, az = _broadcast_vectors(x, y, z, ax, ay, az)

    if _check_scalar(x, y, z):
        return rotate_scalar(x, y, z, ax, ay, az, rotation_angle, assume_normalized)

    return rotate_xyz(x, y, z, ax, ay, az, rotation_angle, assume_normalized)


def minimum_rotation_to_align(x1, y1, z1, x2, y2, z2, assume_normalized=False):
    """
    Compute the minimum rotation that aligns one 3D vector with another.

    The returned rotation is the shortest rotation taking vector 1 onto
    vector 2. The rotation angle is in radians and lies in the interval
    [0, pi].

    Parameters
    ----------
    x1, y1, z1 : float
        Components of the vector to rotate.
    x2, y2, z2 : float
        Components of the target vector.
    assume_normalized : bool, default False
        If True, assume both input vectors are already unit vectors.
        Otherwise, both vectors are normalized before computing the
        rotation.

    Returns
    -------
    axis : tuple of float
        Unit rotation axis. If the vectors are already aligned, the axis
        is arbitrary because the required rotation angle is zero.
    angle : float
        Rotation angle in radians in the interval [0, pi].

    Notes
    -----
    When the vectors are antiparallel, there are infinitely many valid
    rotation axes. In this case, an axis perpendicular to the first
    vector is selected deterministically.
    """

    if not assume_normalized:
        x1, y1, z1 = normalize_scalar(x1, y1, z1)
        x2, y2, z2 = normalize_scalar(x2, y2, z2)

    ang = angle_scalar(x1, y1, z1, x2, y2, z2, assume_normalized=True, degrees=False)

    if isclose(ang, 0.0):
        # already aligned; axis is irrelevant since angle is 0
        return (1.0, 0.0, 0.0), 0.0

    if isclose(ang, pi):
        # antiparallel: cross(v1, v2) is ~0, need any axis perpendicular
        # to v1. Pick the standard basis vector least parallel to v1 to
        # avoid a near-zero cross product.
        if abs(x1) <= abs(y1) and abs(x1) <= abs(z1):
            ref = (1.0, 0.0, 0.0)
        elif abs(y1) <= abs(z1):
            ref = (0.0, 1.0, 0.0)
        else:
            ref = (0.0, 0.0, 1.0)
        ax, ay, az = cross_scalar(x1, y1, z1, *ref)
        n = sqrt(ax**2 + ay**2 + az**2)
        return (ax / n, ay / n, az / n), ang

    ax, ay, az = cross_scalar(x1, y1, z1, x2, y2, z2)
    n = sqrt(ax**2 + ay**2 + az**2)
    return (ax / n, ay / n, az / n), ang


def compute_alignment_rotation(
    e1,
    e2,
    e3,
    b1=(0.0, 1.0, 0.0),
    b2=(1.0, 0.0, 0.0),
    b3=(0.0, 0.0, 1.0),
    assume_normalized=False,
):
    """
    Compute two minimum-rotation steps that align one orthonormal basis
    with another.

    The first rotation aligns ``e1`` with ``b1``. The second rotation is
    then computed using the transformed ``e2`` and aligns it with ``b2``.
    Together, the two rotations align the input basis with the target
    basis.

    Parameters
    ----------
    e1, e2, e3 : array_like
        Three components of the input orthonormal basis vectors.
    b1, b2, b3 : array_like, default
        Three components of the target orthonormal basis vectors.
        The default target basis is

        ``b1 = (0, 1, 0)``
        ``b2 = (1, 0, 0)``
        ``b3 = (0, 0, 1)``

    assume_normalized : bool, default False
        If True, assume the input basis vectors are already normalized.
        Otherwise, each input vector is normalized before computing the
        rotations. The target basis vectors are always normalized.

    Returns
    -------
    step1 : tuple
        ``(axis1, angle1)`` describing the first rotation.
    step2 : tuple
        ``(axis2, angle2)`` describing the second rotation.

        Each axis is a tuple ``(x, y, z)`` representing a unit vector,
        and each angle is in radians.

    Notes
    -----
    The handedness of the input basis is adjusted to match the handedness
    of the target basis before computing the rotations. If necessary,
    ``e3`` is negated. This is required because eigenvectors can have
    arbitrary signs.

    The second rotation is computed after applying the first rotation to
    ``e2``. Consequently, the two returned rotations should be applied in
    the order ``step1`` followed by ``step2``.
    """

    e1 = asarray(e1, dtype=float)
    e2 = asarray(e2, dtype=float)
    e3 = asarray(e3, dtype=float)

    if not assume_normalized:
        e1 = e1 / norm(e1)
        e2 = e2 / norm(e2)
        e3 = e3 / norm(e3)

    b1 = tuple(asarray(b1, dtype=float) / norm(b1))
    b2 = tuple(asarray(b2, dtype=float) / norm(b2))
    b3 = tuple(asarray(b3, dtype=float) / norm(b3))

    # --- fix eigenvector handedness to match the target triad's handedness ---
    cx, cy, cz = cross_scalar(*e1, *e2)
    triple_e = dot_scalar(cx, cy, cz, *e3)
    cbx, cby, cbz = cross_scalar(*b1, *b2)
    triple_b = dot_scalar(cbx, cby, cbz, *b3)
    if (triple_e < 0.0) != (triple_b < 0.0):
        e3 = -e3

    # --- step 1: e1 -> b1 ---
    axis1, angle1 = minimum_rotation_to_align(*e1, *b1, assume_normalized=True)

    # carry e2 through step 1's rotation
    e2_rot = array(rotate_scalar(*e2, *axis1, angle1, assume_normalized=True))

    # --- step 2: rotated e2 -> b2 (axis will land on ±b1 automatically) ---
    axis2, angle2 = minimum_rotation_to_align(*e2_rot, *b2, assume_normalized=True)

    return (axis1, angle1), (axis2, angle2)


def apply_rotation_steps(x, y, z, step1, step2):
    """
    Apply two successive axis-angle rotations to a 3D vector or array
    of vectors.

    The rotations are applied in the order ``step1`` followed by
    ``step2``.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of the vector or vectors to rotate.
    step1 : tuple
        First rotation specified as ``(axis, angle)``, where ``axis`` is
        a three-component unit vector and ``angle`` is in radians.
    step2 : tuple
        Second rotation specified as ``(axis, angle)``, where ``axis`` is
        a three-component unit vector and ``angle`` is in radians.

    Returns
    -------
    xr, yr, zr : float or ndarray
        Components of the vector or vectors after both rotations.

    Notes
    -----
    The rotation axes in ``step1`` and ``step2`` are assumed to be
    normalized. The two rotations are applied sequentially, so the result
    is equivalent to first applying ``step1`` and then applying ``step2``.
    """
    (axis1, angle1) = step1
    (axis2, angle2) = step2
    x, y, z = rotate(x, y, z, *axis1, angle1, assume_normalized=True)
    x, y, z = rotate(x, y, z, *axis2, angle2, assume_normalized=True)
    return x, y, z


def rotate_about(
    x,
    y,
    z,
    ax,
    ay,
    az,
    angle,
    cx,
    cy,
    cz,
    assume_normalized=False,
):
    """
    Rotate a 3D point about an axis passing through a point.

    Uses Rodrigues' rotation formula. The rotation angle is in radians.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of the point to rotate.
    ax, ay, az : float or ndarray
        Components of the rotation axis. A scalar axis may be broadcast
        against array points.
    angle : float
        Rotation angle in radians.
    cx, cy, cz : float or ndarray
        Components of a point through which the rotation axis passes.
        Scalar centre coordinates may be broadcast against array points.
    assume_normalized : bool, default False
        If True, treat ``(ax, ay, az)`` as a unit vector.

    Returns
    -------
    xo, yo, zo : float or ndarray
        Components of the rotated point.
    """

    if _check_vector_broadcast(x, y, z, ax, ay, az):
        x, y, z, ax, ay, az = _broadcast_vectors(x, y, z, ax, ay, az)

    if _check_vector_broadcast(x, y, z, cx, cy, cz):
        x, y, z, cx, cy, cz = _broadcast_vectors(x, y, z, cx, cy, cz)

    if _check_scalar(x, y, z):
        return rotate_about_scalar(
            x,
            y,
            z,
            ax,
            ay,
            az,
            angle,
            cx,
            cy,
            cz,
            assume_normalized,
        )

    return rotate_about_xyz(
        x,
        y,
        z,
        ax,
        ay,
        az,
        angle,
        cx,
        cy,
        cz,
        assume_normalized,
    )
