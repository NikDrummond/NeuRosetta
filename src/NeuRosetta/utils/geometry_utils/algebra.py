import numpy as np
from numba import njit

_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")

###
# NUMBA
###


## scalars
@njit(**_JIT)
def magnitude_scalar(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


@njit(**_JIT)
def normalize_scalar(x, y, z):

    mag = np.sqrt(x * x + y * y + z * z)

    if mag > 0.0:
        inv = 1.0 / mag
        return x * inv, y * inv, z * inv

    return 0.0, 0.0, 0.0


@njit(**_JIT)
def dot_scalar(x1, y1, z1, x2, y2, z2):

    return x1 * x2 + y1 * y2 + z1 * z2


@njit(**_JIT)
def cross_scalar(x1, y1, z1, x2, y2, z2):

    return (y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2)


@njit(**_JIT)
def scalar_projection_scalar(vx, vy, vz, ax, ay, az):
    """
    Scalar projection of vector onto another vector.
    """

    mag = np.sqrt(ax * ax + ay * ay + az * az)

    if mag == 0.0:
        return np.nan

    inv = 1.0 / mag

    return vx * ax * inv + vy * ay * inv + vz * az * inv


@njit(**_JIT)
def project_scalar(vx, vy, vz, ax, ay, az):

    mag = np.sqrt(ax * ax + ay * ay + az * az)

    if mag == 0.0:
        return 0.0, 0.0, 0.0

    inv = 1.0 / mag

    scale = vx * ax * inv + vy * ay * inv + vz * az * inv

    return (scale * ax * inv, scale * ay * inv, scale * az * inv)


@njit(**_JIT)
def reject_scalar(vx, vy, vz, ax, ay, az):

    px, py, pz = project_scalar(vx, vy, vz, ax, ay, az)

    return (vx - px, vy - py, vz - pz)


# @njit(cache=True)
# def reject_axis_scalar(x, y, z, basis):

#     if basis == 0:
#         return 0.0, y, z
#     elif basis == 1:
#         return x, 0.0, z
#     elif basis == 2:
#         return x, y, 0.0

#     return np.nan, np.nan, np.nan


@njit(**_JIT)
def perpendicular_scalar(x1, y1, z1, x2, y2, z2, normalized=True):

    cx = y1 * z2 - z1 * y2
    cy = z1 * x2 - x1 * z2
    cz = x1 * y2 - y1 * x2

    if normalized:

        mag = np.sqrt(cx * cx + cy * cy + cz * cz)

        if mag > 0.0:
            inv = 1.0 / mag
            cx *= inv
            cy *= inv
            cz *= inv

    return cx, cy, cz


@njit(**_JIT)
def scale_factor_scalar(x1, y1, z1, x2, y2, z2):

    denom = x1 * x1 + y1 * y1 + z1 * z1

    if denom == 0.0:
        return np.nan

    return (x1 * x2 + y1 * y2 + z1 * z2) / denom


### arrays
@njit(**_JIT)
def magnitude_xyz(x, y, z):

    n = x.size
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        out[i] = np.sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i])

    return out


@njit(**_JIT)
def normalize_xyz(x, y, z):

    n = x.size

    xn = np.empty(n, dtype=np.float64)
    yn = np.empty(n, dtype=np.float64)
    zn = np.empty(n, dtype=np.float64)

    for i in range(n):

        mag = np.sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i])

        if mag > 0.0:
            inv = 1.0 / mag

            xn[i] = x[i] * inv
            yn[i] = y[i] * inv
            zn[i] = z[i] * inv

        else:
            xn[i] = 0.0
            yn[i] = 0.0
            zn[i] = 0.0

    return xn, yn, zn


@njit(**_JIT)
def dot_xyz(x1, y1, z1, x2, y2, z2):

    n = x1.size
    out = np.empty(n, dtype=np.float64)

    for i in range(n):

        out[i] = x1[i] * x2[i] + y1[i] * y2[i] + z1[i] * z2[i]

    return out


@njit(**_JIT)
def cross_xyz(x1, y1, z1, x2, y2, z2):

    n = x1.size

    cx = np.empty(n, dtype=np.float64)
    cy = np.empty(n, dtype=np.float64)
    cz = np.empty(n, dtype=np.float64)

    for i in range(n):

        cx[i] = y1[i] * z2[i] - z1[i] * y2[i]
        cy[i] = z1[i] * x2[i] - x1[i] * z2[i]
        cz[i] = x1[i] * y2[i] - y1[i] * x2[i]

    return cx, cy, cz


@njit(**_JIT)
def scalar_projection_xyz(vx, vy, vz, ax, ay, az):
    """
    Array version of scalar projection.
    """

    n = vx.size

    out = np.empty(n, dtype=np.float64)

    for i in range(n):

        mag = np.sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i])

        if mag == 0.0:
            out[i] = np.nan
        else:
            inv = 1.0 / mag

            out[i] = vx[i] * ax[i] * inv + vy[i] * ay[i] * inv + vz[i] * az[i] * inv

    return out


@njit(**_JIT)
def project_xyz(vx, vy, vz, ax, ay, az):

    n = vx.size

    px = np.empty(n)
    py = np.empty(n)
    pz = np.empty(n)

    for i in range(n):

        mag = np.sqrt(ax[i] * ax[i] + ay[i] * ay[i] + az[i] * az[i])

        if mag == 0.0:

            px[i] = 0.0
            py[i] = 0.0
            pz[i] = 0.0

        else:

            inv = 1.0 / mag

            scale = vx[i] * ax[i] * inv + vy[i] * ay[i] * inv + vz[i] * az[i] * inv

            px[i] = scale * ax[i] * inv
            py[i] = scale * ay[i] * inv
            pz[i] = scale * az[i] * inv

    return px, py, pz


@njit(**_JIT)
def reject_xyz(vx, vy, vz, ax, ay, az):

    px, py, pz = project_xyz(vx, vy, vz, ax, ay, az)

    return (vx - px, vy - py, vz - pz)


@njit(**_JIT)
def perpendicular_xyz(x1, y1, z1, x2, y2, z2, normalized=True):
    """ """

    n = x1.size

    px = np.empty(n, dtype=np.float64)
    py = np.empty(n, dtype=np.float64)
    pz = np.empty(n, dtype=np.float64)

    for i in range(n):

        # Cross product
        cx = y1[i] * z2[i] - z1[i] * y2[i]
        cy = z1[i] * x2[i] - x1[i] * z2[i]
        cz = x1[i] * y2[i] - y1[i] * x2[i]

        if normalized:

            mag = np.sqrt(cx * cx + cy * cy + cz * cz)

            if mag > 0.0:
                inv_mag = 1.0 / mag
                cx *= inv_mag
                cy *= inv_mag
                cz *= inv_mag

        px[i] = cx
        py[i] = cy
        pz[i] = cz

    return px, py, pz


@njit(**_JIT)
def scale_factor_xyz(x1, y1, z1, x2, y2, z2):

    n = x1.size

    out = np.empty(n)

    for i in range(n):

        denom = x1[i] * x1[i] + y1[i] * y1[i] + z1[i] * z1[i]

        if denom == 0.0:
            out[i] = np.nan
        else:
            out[i] = (x1[i] * x2[i] + y1[i] * y2[i] + z1[i] * z2[i]) / denom

    return out


###
# Numpy
###


### Check
def _check_scalar(x, y, z):
    """
    Check whether three vector components represent a scalar 3D vector.

    Parameters
    ----------
    x, y, z : scalar or array-like
        Components of a 3D vector.

    Returns
    -------
    bool
        True if all components are scalar (shape ``()``).

    Raises
    ------
    ValueError
        If the components do not share the same shape.
    """
    v_shapes = [np.shape(v) for v in (x, y, z)]

    if not (v_shapes[0] == v_shapes[1] == v_shapes[2]):
        raise ValueError(f"vector components have inconsistent shapes: {v_shapes}")

    return v_shapes[0] == v_shapes[1] == v_shapes[2] == ()


def _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
    """
    Check whether two 3D vectors require broadcasting.

    Returns
    -------
    bool
        True if one vector is scalar and the other is array-like.

    Raises
    ------
    ValueError
        If components within a vector have inconsistent shapes.
        If both vectors are arrays with incompatible shapes.
    """

    v1_shapes = [np.shape(v) for v in (x1, y1, z1)]
    v2_shapes = [np.shape(v) for v in (x2, y2, z2)]

    # Check internal consistency of each vector
    if not (v1_shapes[0] == v1_shapes[1] == v1_shapes[2]):
        raise ValueError(
            f"First vector components have inconsistent shapes: {v1_shapes}"
        )

    if not (v2_shapes[0] == v2_shapes[1] == v2_shapes[2]):
        raise ValueError(
            f"Second vector components have inconsistent shapes: {v2_shapes}"
        )

    shape1 = v1_shapes[0]
    shape2 = v2_shapes[0]

    scalar1 = shape1 == ()
    scalar2 = shape2 == ()

    # Both arrays
    if not scalar1 and not scalar2:
        if shape1 != shape2:
            raise ValueError(f"Vector shapes do not match: {shape1} vs {shape2}")

        return False

    # One scalar, one array
    return scalar1 != scalar2


def _broadcast_vectors(x1, y1, z1, x2, y2, z2):
    """
    Broadcast two 3D vectors represented as separate components.

    Parameters
    ----------
    x1, y1, z1 : scalar or ndarray
        First vector components.
    x2, y2, z2 : scalar or ndarray
        Second vector components.

    Returns
    -------
    x1, y1, z1, x2, y2, z2
        Broadcast-compatible components.
    """

    v1_array = any(np.ndim(v) > 0 for v in (x1, y1, z1))
    v2_array = any(np.ndim(v) > 0 for v in (x2, y2, z2))

    # Determine target shape
    if v1_array and v2_array:

        # Ensure both vectors have matching shapes
        shape1 = np.shape(x1)
        shape2 = np.shape(x2)

        if shape1 != shape2:
            raise ValueError(f"Vector shapes do not match: {shape1} and {shape2}")

        return x1, y1, z1, x2, y2, z2

    elif v1_array:

        shape = np.shape(x1)

        x2 = np.full(shape, x2)
        y2 = np.full(shape, y2)
        z2 = np.full(shape, z2)

    elif v2_array:

        shape = np.shape(x2)

        x1 = np.full(shape, x1)
        y1 = np.full(shape, y1)
        z1 = np.full(shape, z1)

    return x1, y1, z1, x2, y2, z2


def magnitude(x, y, z):
    """
    Euclidean magnitude of one or more 3D vectors.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of a 3D vector. For array inputs, all components must have
        the same shape.

    Returns
    -------
    float or ndarray
        ``sqrt(x**2 + y**2 + z**2)`` for each vector.
    """

    if _check_scalar(x, y, z):
        return magnitude_scalar(x, y, z)

    return magnitude_xyz(x, y, z)


def normalize(x, y, z):
    """
    Normalize 3D vector(s) to unit length.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of a 3D vector. For array inputs, all components must have
        the same shape.

    Returns
    -------
    tuple of float or tuple of ndarray
        Unit vector components. Zero-length vectors are mapped to ``(0, 0, 0)``.
    """

    if _check_scalar(x, y, z):
        return normalize_scalar(x, y, z)

    return normalize_xyz(x, y, z)


def dot(x1, y1, z1, x2, y2, z2):
    """
    Dot product of two 3D vectors.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the first vector.
    x2, y2, z2 : float or ndarray
        Components of the second vector. A scalar vector may be broadcast
        against an array vector.

    Returns
    -------
    float or ndarray
        ``x1 * x2 + y1 * y2 + z1 * z2`` for each pair of vectors.
    """

    # broadcast check
    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
        x1, y1, z1, x2, y2, z2 = _broadcast_vectors(x1, y1, z1, x2, y2, z2)

    if _check_scalar(x1, y1, z1):
        return dot_scalar(x1, y1, z1, x2, y2, z2)

    return dot_xyz(x1, y1, z1, x2, y2, z2)


def cross(x1, y1, z1, x2, y2, z2):
    """
    Cross product of two 3D vectors.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the first vector.
    x2, y2, z2 : float or ndarray
        Components of the second vector. A scalar vector may be broadcast
        against an array vector.

    Returns
    -------
    cx, cy, cz : float or ndarray
        Components of ``v1 x v2``.
    """

    # broadcast check
    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
        x1, y1, z1, x2, y2, z2 = _broadcast_vectors(x1, y1, z1, x2, y2, z2)

    if _check_scalar(x1, y1, z1):
        return cross_scalar(x1, y1, z1, x2, y2, z2)

    return cross_xyz(x1, y1, z1, x2, y2, z2)


def scalar_projection(x1, y1, z1, x2, y2, z2):
    """
    Scalar projection of one 3D vector onto another.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the vector being projected.
    x2, y2, z2 : float or ndarray
        Components of the axis vector. A scalar vector may be broadcast
        against an array vector.

    Returns
    -------
    float or ndarray
        Signed length of the projection of ``v1`` onto ``v2``. Returns ``nan``
        when ``v2`` has zero length.
    """

    # broadcast check
    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
        x1, y1, z1, x2, y2, z2 = _broadcast_vectors(x1, y1, z1, x2, y2, z2)

    if _check_scalar(x1, y1, z1):
        return scalar_projection_scalar(x1, y1, z1, x2, y2, z2)

    return scalar_projection_xyz(x1, y1, z1, x2, y2, z2)


def project(x1, y1, z1, x2, y2, z2):
    """
    Vector projection of one 3D vector onto another.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the vector being projected.
    x2, y2, z2 : float or ndarray
        Components of the axis vector. A scalar vector may be broadcast
        against an array vector.

    Returns
    -------
    px, py, pz : float or ndarray
        Components of the projection of ``v1`` onto ``v2``. Returns
        ``(0, 0, 0)`` when ``v2`` has zero length.
    """

    # broadcast check
    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
        x1, y1, z1, x2, y2, z2 = _broadcast_vectors(x1, y1, z1, x2, y2, z2)

    if _check_scalar(x1, y1, z1):
        return project_scalar(x1, y1, z1, x2, y2, z2)

    return project_xyz(x1, y1, z1, x2, y2, z2)


def reject(x1, y1, z1, x2, y2, z2):
    """
    Vector rejection of one 3D vector from another.

    Returns the component of ``v1`` orthogonal to ``v2``:
    ``v1 - project(v1, v2)``.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the vector being rejected.
    x2, y2, z2 : float or ndarray
        Components of the axis vector. A scalar vector may be broadcast
        against an array vector.

    Returns
    -------
    rx, ry, rz : float or ndarray
        Components of the rejected vector.
    """
    # broadcast check
    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
        x1, y1, z1, x2, y2, z2 = _broadcast_vectors(x1, y1, z1, x2, y2, z2)

    if _check_scalar(x1, y1, z1):
        return reject_scalar(x1, y1, z1, x2, y2, z2)

    return reject_xyz(x1, y1, z1, x2, y2, z2)


def perpendicular(x1, y1, z1, x2, y2, z2, normalized=True):
    """
    Cross product of two 3D vectors, optionally normalized.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the first vector.
    x2, y2, z2 : float or ndarray
        Components of the second vector. A scalar vector may be broadcast
        against an array vector.
    normalized : bool, default True
        If True, return a unit vector when the cross product is non-zero.

    Returns
    -------
    px, py, pz : float or ndarray
        Components of ``v1 x v2``, optionally normalized to unit length.
    """
    # broadcast check
    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
        x1, y1, z1, x2, y2, z2 = _broadcast_vectors(x1, y1, z1, x2, y2, z2)

    if _check_scalar(x1, y1, z1):
        return perpendicular_scalar(x1, y1, z1, x2, y2, z2, normalized)

    return perpendicular_xyz(x1, y1, z1, x2, y2, z2, normalized)


def scale_factor(x1, y1, z1, x2, y2, z2):
    """
    Scalar factor relating two collinear 3D vectors.

    Computes ``(v1 · v2) / |v1|^2``, i.e. the scalar ``s`` such that
    ``v2 ≈ s * v1`` when the vectors are collinear.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the reference vector.
    x2, y2, z2 : float or ndarray
        Components of the target vector. A scalar vector may be broadcast
        against an array vector.

    Returns
    -------
    float or ndarray
        Scale factor for each vector pair. Returns ``nan`` when ``v1`` has
        zero length.
    """
    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
        x1, y1, z1, x2, y2, z2 = _broadcast_vectors(x1, y1, z1, x2, y2, z2)

    if _check_scalar(x1, y1, z1):
        return scale_factor_scalar(x1, y1, z1, x2, y2, z2)

    return scale_factor_xyz(x1, y1, z1, x2, y2, z2)
