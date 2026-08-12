from numba import njit
from numpy import empty, float64

from ._validation import _check_scalar

_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")


# Scalar ops
@njit(**_JIT)
def translate_scalar(
    x,
    y,
    z,
    dx,
    dy,
    dz,
):
    return (
        x + dx,
        y + dy,
        z + dz,
    )


@njit(**_JIT)
def recenter_scalar(
    x,
    y,
    z,
    cx,
    cy,
    cz,
):
    return (
        x - cx,
        y - cy,
        z - cz,
    )


@njit(**_JIT)
def scale_about_scalar(
    x,
    y,
    z,
    sx,
    sy,
    sz,
    cx,
    cy,
    cz,
):
    return (
        cx + sx * (x - cx),
        cy + sy * (y - cy),
        cz + sz * (z - cz),
    )


# array ops
@njit(**_JIT)
def translate_xyz(
    x,
    y,
    z,
    dx,
    dy,
    dz,
):
    n = x.size

    xo = empty(n, dtype=float64)
    yo = empty(n, dtype=float64)
    zo = empty(n, dtype=float64)

    for i in range(n):
        xo[i], yo[i], zo[i] = translate_scalar(
            x[i],
            y[i],
            z[i],
            dx,
            dy,
            dz,
        )

    return xo, yo, zo


@njit(**_JIT)
def recenter_xyz(
    x,
    y,
    z,
    cx,
    cy,
    cz,
):
    n = x.size

    xo = empty(n, dtype=float64)
    yo = empty(n, dtype=float64)
    zo = empty(n, dtype=float64)

    for i in range(n):
        xo[i], yo[i], zo[i] = recenter_scalar(
            x[i],
            y[i],
            z[i],
            cx,
            cy,
            cz,
        )

    return xo, yo, zo


@njit(**_JIT)
def scale_about_xyz(
    x,
    y,
    z,
    sx,
    sy,
    sz,
    cx,
    cy,
    cz,
):
    n = x.size

    xo = empty(n, dtype=float64)
    yo = empty(n, dtype=float64)
    zo = empty(n, dtype=float64)

    for i in range(n):
        xo[i], yo[i], zo[i] = scale_about_scalar(
            x[i],
            y[i],
            z[i],
            sx,
            sy,
            sz,
            cx,
            cy,
            cz,
        )

    return xo, yo, zo


@njit(**_JIT)
def center_at_centroid_xyz(
    x,
    y,
    z,
):
    n = x.size

    cx = 0.0
    cy = 0.0
    cz = 0.0

    for i in range(n):
        cx += x[i]
        cy += y[i]
        cz += z[i]

    inv_n = 1.0 / n

    cx *= inv_n
    cy *= inv_n
    cz *= inv_n

    xo, yo, zo = recenter_xyz(
        x,
        y,
        z,
        cx,
        cy,
        cz,
    )

    return xo, yo, zo, cx, cy, cz


def translate(x, y, z, dx, dy, dz):
    """
    Translate a 3D vector or array of 3D vectors by a displacement.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of the vector or vectors to translate.
    dx, dy, dz : float
        Components of the translation displacement.

    Returns
    -------
    xo, yo, zo : float or ndarray
        Components of the translated vector or vectors.
    """

    if _check_scalar(x, y, z):
        return translate_scalar(
            x,
            y,
            z,
            dx,
            dy,
            dz,
        )

    return translate_xyz(
        x,
        y,
        z,
        dx,
        dy,
        dz,
    )


def recenter(x, y, z, cx, cy, cz):
    """
    Recenter a 3D vector or array of 3D vectors relative to a point.

    The supplied centre is subtracted from each vector component. This is
    equivalent to translating the coordinates by ``(-cx, -cy, -cz)``.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of the vector or vectors to recenter.
    cx, cy, cz : float
        Components of the point to use as the new centre.

    Returns
    -------
    xo, yo, zo : float or ndarray
        Components relative to the supplied centre.
    """

    if _check_scalar(x, y, z):
        return recenter_scalar(
            x,
            y,
            z,
            cx,
            cy,
            cz,
        )

    return recenter_xyz(
        x,
        y,
        z,
        cx,
        cy,
        cz,
    )


def scale_about(
    x,
    y,
    z,
    sx,
    sy,
    sz,
    cx,
    cy,
    cz,
):
    """
    Scale a 3D vector or array of 3D vectors about a specified point.

    Scaling is performed independently along each coordinate axis. The
    supplied centre remains fixed during the transformation.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of the vector or vectors to scale.
    sx, sy, sz : float
        Scale factors along the x, y, and z axes, respectively.
    cx, cy, cz : float
        Components of the point about which the scaling is performed.

    Returns
    -------
    xo, yo, zo : float or ndarray
        Components of the scaled vector or vectors.

    Notes
    -----
    A scale factor of ``1`` leaves the corresponding coordinate unchanged.
    A scale factor of ``0`` collapses the corresponding coordinate onto
    the centre. Negative scale factors reflect the corresponding
    coordinate through the centre while scaling its magnitude.
    """

    if _check_scalar(x, y, z):
        return scale_about_scalar(
            x,
            y,
            z,
            sx,
            sy,
            sz,
            cx,
            cy,
            cz,
        )

    return scale_about_xyz(
        x,
        y,
        z,
        sx,
        sy,
        sz,
        cx,
        cy,
        cz,
    )


def center_at_centroid(x, y, z):
    """
    Translate an array of 3D points so that its centroid is at the origin.

    The centroid is computed independently for each coordinate as the
    arithmetic mean of all points. The input points are then translated
    by the negative centroid.

    Parameters
    ----------
    x, y, z : ndarray
        Components of the points to center. All three arrays must have
        the same length.

    Returns
    -------
    xo, yo, zo : ndarray
        Components of the centered points.
    cx, cy, cz : float
        Components of the centroid that was subtracted from the input
        points.

    Notes
    -----
    The centroid is

    ``(mean(x), mean(y), mean(z))``.

    The returned centroid can be added back to ``(xo, yo, zo)`` to
    recover the original coordinates, subject to floating-point
    round-off.
    """

    return center_at_centroid_xyz(
        x,
        y,
        z,
    )
