# angles.py
import numpy as np
from numba import njit

from .algebra import (
    dot_scalar,
    magnitude_scalar,
    cross_scalar,
    reject_scalar,
    normalize_scalar,
)
from ._validation import _check_vector_broadcast, _broadcast_vectors, _check_scalar

### Numba

_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")


@njit(**_JIT)
def clamp(x, xmin, xmax):
    if x < xmin:
        return xmin
    if x > xmax:
        return xmax
    return x


### Scalar kernels


@njit(**_JIT)
def angle_scalar(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    assume_normalized=False,
    degrees=True,
):
    """
    Unsigned angle between two 3D vectors.
    """

    d = dot_scalar(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
    )

    if assume_normalized:
        c = d
    else:

        m1 = magnitude_scalar(x1, y1, z1)
        m2 = magnitude_scalar(x2, y2, z2)

        if m1 == 0.0 or m2 == 0.0:
            return np.nan

        c = d / (m1 * m2)

    c = clamp(c, -1.0, 1.0)

    a = np.arccos(c)

    if degrees:
        a *= 180.0 / np.pi

    return a


@njit(**_JIT)
def planar_angle_scalar(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    lx,
    ly,
    lz,
    degrees=True,
):
    """
    Angle after projecting both vectors into the plane
    perpendicular to the look vector.
    """

    x1, y1, z1 = reject_scalar(
        x1,
        y1,
        z1,
        lx,
        ly,
        lz,
    )

    x2, y2, z2 = reject_scalar(
        x2,
        y2,
        z2,
        lx,
        ly,
        lz,
    )

    return angle_scalar(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        False,
        degrees,
    )


@njit(**_JIT)
def signed_angle_scalar(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    lx,
    ly,
    lz,
    degrees=True,
):
    """
    Signed angle in the plane perpendicular
    to the supplied look vector.
    """

    #
    # Project into plane
    #

    x1, y1, z1 = reject_scalar(
        x1,
        y1,
        z1,
        lx,
        ly,
        lz,
    )

    x2, y2, z2 = reject_scalar(
        x2,
        y2,
        z2,
        lx,
        ly,
        lz,
    )

    a = angle_scalar(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
        False,
        degrees,
    )

    #
    # Orientation
    #

    cx, cy, cz = cross_scalar(
        x1,
        y1,
        z1,
        x2,
        y2,
        z2,
    )

    orient = dot_scalar(
        cx,
        cy,
        cz,
        lx,
        ly,
        lz,
    )

    if orient < 0.0:
        a = -a

    return a


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

    c = np.cos(rotation_angle)
    s = np.sin(rotation_angle)

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
def align_with_scalar(
    x,
    y,
    z,
    ax,
    ay,
    az,
    reverse=False,
):
    """
    Flip a vector so that it points along (or opposite)
    another vector.
    """

    d = dot_scalar(
        x,
        y,
        z,
        ax,
        ay,
        az,
    )

    if reverse:

        if d > 0.0:
            return -x, -y, -z

    else:

        if d < 0.0:
            return -x, -y, -z

    return x, y, z


### Array Kernels


@njit(**_JIT)
def angle_xyz(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    assume_normalized=False,
    degrees=True,
):

    n = x1.size

    out = np.empty(n, dtype=np.float64)

    for i in range(n):

        out[i] = angle_scalar(
            x1[i],
            y1[i],
            z1[i],
            x2[i],
            y2[i],
            z2[i],
            assume_normalized,
            degrees,
        )

    return out


@njit(**_JIT)
def planar_angle_xyz(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    lx,
    ly,
    lz,
    degrees=True,
):

    n = x1.size

    out = np.empty(n, dtype=np.float64)

    for i in range(n):

        out[i] = planar_angle_scalar(
            x1[i],
            y1[i],
            z1[i],
            x2[i],
            y2[i],
            z2[i],
            lx[i],
            ly[i],
            lz[i],
            degrees,
        )

    return out


@njit(**_JIT)
def signed_angle_xyz(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    lx,
    ly,
    lz,
    degrees=True,
):

    n = x1.size

    out = np.empty(n, dtype=np.float64)

    for i in range(n):

        out[i] = signed_angle_scalar(
            x1[i],
            y1[i],
            z1[i],
            x2[i],
            y2[i],
            z2[i],
            lx[i],
            ly[i],
            lz[i],
            degrees,
        )

    return out


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

    xr = np.empty(n)
    yr = np.empty(n)
    zr = np.empty(n)

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
def align_with_xyz(
    x,
    y,
    z,
    ax,
    ay,
    az,
    reverse=False,
):

    n = x.size

    xo = np.empty(n)
    yo = np.empty(n)
    zo = np.empty(n)

    for i in range(n):

        xo[i], yo[i], zo[i] = align_with_scalar(
            x[i],
            y[i],
            z[i],
            ax[i],
            ay[i],
            az[i],
            reverse,
        )

    return xo, yo, zo


### Angular mean and va


@njit(**_JIT)
def _angular_resultant(angles, weights=None):

    c = 0.0
    s = 0.0

    if weights is None:

        wsum = float(angles.size)

        for i in range(angles.size):
            a = angles[i]
            c += np.cos(a)
            s += np.sin(a)

    else:

        wsum = 0.0

        for i in range(angles.size):
            w = weights[i]
            a = angles[i]

            c += w * np.cos(a)
            s += w * np.sin(a)
            wsum += w

    return c, s, wsum


@njit(cache=True, fastmath=True)
def angular_mean_nb(angles, weights=None):

    c, s, _ = _angular_resultant(angles, weights)
    return np.arctan2(s, c)


@njit(cache=True, fastmath=True)
def angular_variance_nb(angles, weights=None):

    c, s, wsum = _angular_resultant(angles, weights)

    if wsum == 0.0:
        return np.nan

    return 1.0 - np.sqrt(c * c + s * s) / wsum


### Python wrappers


def angle(x1, y1, z1, x2, y2, z2, assume_normalized=False, degrees=False):
    """
    Unsigned angle between two 3D vectors.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the first vector.
    x2, y2, z2 : float or ndarray
        Components of the second vector. A scalar vector may be broadcast
        against an array vector.
    assume_normalized : bool, default False
        If True, skip magnitude normalization before taking the arccosine.
    degrees : bool, default False
        If True, return the angle in degrees; otherwise radians.

    Returns
    -------
    float or ndarray
        Angle in ``[0, pi]`` (radians) or ``[0, 180]`` (degrees). Returns
        ``nan`` for zero-length vectors when ``assume_normalized`` is False.
    """

    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
        x1, y1, z1, x2, y2, z2 = _broadcast_vectors(x1, y1, z1, x2, y2, z2)

    if _check_scalar(x1, y1, z1):
        return angle_scalar(x1, y1, z1, x2, y2, z2, assume_normalized, degrees)

    return angle_xyz(x1, y1, z1, x2, y2, z2, assume_normalized, degrees)


def planar_angle(x1, y1, z1, x2, y2, z2, l1, l2, l3, degrees=False):
    """
    Angle between two vectors after projecting into a plane.

    Both vectors are rejected from the look vector ``(l1, l2, l3)`` before
    computing the unsigned angle.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the first vector.
    x2, y2, z2 : float or ndarray
        Components of the second vector.
    l1, l2, l3 : float or ndarray
        Components of the look vector defining the plane normal. Scalar
        vectors may be broadcast against array vectors.
    degrees : bool, default False
        If True, return the angle in degrees; otherwise radians.

    Returns
    -------
    float or ndarray
        Planar angle after projection.
    """

    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2, l1, l2, l3):
        x1, y1, z1, x2, y2, z2, l1, l2, l3 = _broadcast_vectors(
            x1, y1, z1, x2, y2, z2, l1, l2, l3
        )

    if _check_scalar(x1, y1, z1):
        return planar_angle_scalar(x1, y1, z1, x2, y2, z2, l1, l2, l3, degrees)

    return planar_angle_xyz(x1, y1, z1, x2, y2, z2, l1, l2, l3, degrees)


def signed_angle(x1, y1, z1, x2, y2, z2, l1, l2, l3, degrees=False):
    """
    Signed angle between two vectors in a plane.

    The sign is determined by the orientation of ``v1 x v2`` relative to the
    look vector ``(l1, l2, l3)``.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the first vector.
    x2, y2, z2 : float or ndarray
        Components of the second vector.
    l1, l2, l3 : float or ndarray
        Components of the look vector defining the plane normal. Scalar
        vectors may be broadcast against array vectors.
    degrees : bool, default False
        If True, return the angle in degrees; otherwise radians.

    Returns
    -------
    float or ndarray
        Signed planar angle. Negative when ``(v1 x v2) · look < 0``.
    """

    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2, l1, l2, l3):
        x1, y1, z1, x2, y2, z2, l1, l2, l3 = _broadcast_vectors(
            x1, y1, z1, x2, y2, z2, l1, l2, l3
        )

    if _check_scalar(x1, y1, z1):
        return signed_angle_scalar(x1, y1, z1, x2, y2, z2, l1, l2, l3, degrees)

    return signed_angle_xyz(x1, y1, z1, x2, y2, z2, l1, l2, l3, degrees)


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


def align_with(x, y, z, ax, ay, az, reverse=False):
    """
    Flip a vector so it points along (or opposite) a reference direction.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of the vector to align.
    ax, ay, az : float or ndarray
        Components of the reference direction. A scalar direction may be
        broadcast against an array vector.
    reverse : bool, default False
        If False, ensure ``v · axis >= 0``. If True, ensure ``v · axis <= 0``.

    Returns
    -------
    xo, yo, zo : float or ndarray
        Original vector, or its negation if a flip was required.
    """

    if _check_vector_broadcast(x, y, z, ax, ay, az):
        x, y, z, ax, ay, az = _broadcast_vectors(x, y, z, ax, ay, az)
    if _check_scalar(x, y, z):
        return align_with_scalar(x, y, z, ax, ay, az, reverse)
    return align_with_xyz(x, y, z, ax, ay, az, reverse)


def angular_mean(angles, weights=None):
    """
    Returns the angular mean of an array of angles IN RADIAN

    Parameters
    ----------
    angles : ndarrau
        1D array of angler. MUST BE IN RADIANS
    weights : ndarray, optional
        Array of weights for weighted mean, by default None

    Returns
    -------
    float
        Mean angle of angles in radians.
    """
    return angular_mean_nb(angles, weights)


def angular_var(angles, weights=None):
    """
    Returns the angular variance of an array of angles IN RADIAN

    Parameters
    ----------
    angles : ndarrau
        1D array of angler. MUST BE IN RADIANS
    weights : ndarray, optional
        Array of weights for weighted variance, by default None

    Returns
    -------
    float
        Anglular variance of angles in radians.
    """
    return angular_variance_nb(angles, weights)
