"""    """
from numpy import cos, sin, empty, isclose, pi, sqrt, asarray, array
from numpy.linalg import norm
from numba import njit

from .algebra import normalize_scalar, cross_scalar, dot_scalar
from .angles import angle_scalar
from ._validation import _check_scalar, _check_vector_broadcast, _broadcast_vectors

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
    Axis and angle of the minimum rotation that aligns vector 1 with
    vector 2.

    Parameters
    ----------
    x1, y1, z1 : float
        Components of the vector to rotate.
    x2, y2, z2 : float
        Components of the target vector.
    assume_normalized : bool, default False
        If True, skip normalizing the inputs.

    Returns
    -------
    axis : (float, float, float)
        Unit rotation axis. Arbitrary but valid when angle == 0
        (no rotation needed).
    angle : float
        Rotation angle in radians, in [0, pi].
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
    e1, e2, e3,
    b1=(0.0, 1.0, 0.0),
    b2=(1.0, 0.0, 0.0),
    b3=(0.0, 0.0, 1.0),
    assume_normalized=False,
):
    """
    Compute the two (axis, angle) rotation steps that align an
    orthonormal eigenbasis (e1, e2, e3) to a target orthonormal basis
    (b1, b2, b3), using minimum_rotation_to_align for each step.
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
    (axis1, angle1) = step1
    (axis2, angle2) = step2
    x, y, z = rotate(x, y, z, *axis1, angle1, assume_normalized=True)
    x, y, z = rotate(x, y, z, *axis2, angle2, assume_normalized=True)
    return x, y, z
