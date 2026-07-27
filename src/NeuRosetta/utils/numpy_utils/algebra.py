# algebra.py

import numpy as np

from ._validation import check, raise_dimension_error, check_value_any


def broadcast(v1, v2):
    """
    Validate shapes and return v1, v2 broadcast to matching (k, 3) arrays.

    Uses np.broadcast_to (a zero-copy view) instead of np.tile (which
    allocates and copies), since every caller here only reads from the
    result.
    """
    if v1.ndim == 1 and v2.ndim == 2:
        check(locals(), "v1", (3,))
        k = check(locals(), "v2", (-1, 3))
        return np.broadcast_to(v1, (k, 3)), v2
    elif v1.ndim == 2 and v2.ndim == 1:
        k = check(locals(), "v1", (-1, 3))
        check(locals(), "v2", (3,))
        return v1, np.broadcast_to(v2, (k, 3))
    elif v1.ndim == 2 and v2.ndim == 2:
        k = check(locals(), "v1", (-1, 3))
        check(locals(), "v2", (k, 3))
        return v1, v2
    else:
        raise_dimension_error(v1, v2)


def normalize(vector):
    if vector.ndim == 1:
        return vector / np.linalg.norm(vector)
    elif vector.ndim == 2:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]
    else:
        raise_dimension_error(vector)


def magnitude(vector):
    if vector.ndim == 1:
        return np.linalg.norm(vector)
    elif vector.ndim == 2:
        return np.linalg.norm(vector, axis=1)
    else:
        raise_dimension_error(vector)


def dot(v1, v2):
    """
    For the batch (n, 3) path, this uses explicit component multiply-and-sum
    rather than np.einsum. For a fixed inner dimension of 3, this is
    typically faster than einsum's more general dispatch path -- benchmark
    against your real n before relying on the exact margin, but it's a
    reliable win for large batches.
    """
    if v1.ndim == 1 and v2.ndim == 1:
        check(locals(), "v1", (3,))
        check(locals(), "v2", (3,))
        return np.dot(v1, v2)
    else:
        v1b, v2b = broadcast(v1, v2)
        return v1b[:, 0] * v2b[:, 0] + v1b[:, 1] * v2b[:, 1] + v1b[:, 2] * v2b[:, 2]


def cross(v1, v2):
    """
    np.cross already broadcasts (3,) against (n, 3) natively, so the old
    manual tile + newaxis + slice-back dance wasn't needed -- np.cross(a, b)
    for two (n, 3) arrays already returns (n, 3) directly.
    """
    if v1.ndim == 1 and v2.ndim == 1:
        check(locals(), "v1", (3,))
        check(locals(), "v2", (3,))
        return np.cross(v1, v2)
    else:
        v1b, v2b = broadcast(v1, v2)
        return np.cross(v1b, v2b)


def scalar_projection(vector, onto):
    if vector.ndim == 1:
        check(locals(), "vector", (3,))
        check(locals(), "onto", (3,))
    else:
        k = check(locals(), "vector", (-1, 3))
        if onto.ndim == 1:
            check(locals(), "onto", (3,))
        else:
            check(locals(), "onto", (k, 3))

    return dot(vector, normalize(onto))


def project(vector, onto):
    """
    Normalizes `onto` exactly once and reuses it, rather than the original
    which normalized it once inside scalar_projection() and again to scale
    the result -- a full extra norm + divide over the whole array.
    Shape validation still happens implicitly via dot()/normalize()/broadcast().
    """
    onto_normalized = normalize(onto)
    projection_scalar = dot(vector, onto_normalized)

    if vector.ndim == 1:
        return projection_scalar * onto_normalized
    elif vector.ndim == 2:
        return projection_scalar[:, np.newaxis] * onto_normalized
    else:
        raise_dimension_error(vector)


def reject(vector, from_v):
    return vector - project(vector, onto=from_v)


def reject_axis(vector, axis, squash=False):
    if squash:
        dims_to_keep = [0, 1, 2]
        try:
            dims_to_keep.remove(axis)
        except ValueError:
            raise ValueError("axis should be 0, 1, or 2")

        if vector.ndim == 1:
            return vector[dims_to_keep]
        elif vector.ndim == 2:
            return vector[:, dims_to_keep]
        else:
            raise_dimension_error(vector)
    else:
        if axis not in [0, 1, 2]:
            raise ValueError("axis should be 0, 1, or 2")
        result = vector.copy()
        if vector.ndim == 1:
            result[axis] = 0.0
        elif vector.ndim == 2:
            result[:, axis] = 0.0
        else:
            raise_dimension_error(vector)
        return result


def perpendicular(v1, v2, normalized=True):
    result = cross(v1, v2)
    return normalize(result) if normalized else result

def scale_factor(v1, v2):
    # Fixed: the original passed name="v1" for BOTH checks, so a shape
    # error on v2 was misreported as a v1 error. Also now shares the single
    # merged check_value_any instead of the near-duplicate private version.
    k = check_value_any(v1, (3,), (-1, 3), name="v1")
    check_value_any(v2, (3,), (-1 if k is None else k, 3), name="v2")

    v1_dot_v2 = dot(v1, v2)
    v1_dot_v1 = dot(v1, v1)

    if np.isscalar(v1_dot_v1) and v1_dot_v1 == 0:
        v1_dot_v1 = np.nan
    elif not np.isscalar(v1_dot_v1):
        v1_dot_v1[v1_dot_v1 == 0] = np.nan

    return v1_dot_v2 / v1_dot_v1