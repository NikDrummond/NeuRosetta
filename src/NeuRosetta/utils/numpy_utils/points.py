# points.py

import numbers
import numpy as np

from ._validation import check, check_value_any
from .algebra import magnitude, dot


def euclidean_distance(v1, v2):
    k = check_value_any(v1, (3,), (-1, 3), name="v1")
    check_value_any(v2, (3,), (-1 if k is None else k, 3), name="v2")

    if v1.ndim == 1 and v2.ndim == 1:
        return np.sqrt(np.sum(np.square(v2 - v1)))
    else:
        return np.sqrt(np.sum(np.square(v2 - v1), axis=1))


def argapex(points, along):
    k = check(locals(), "points", (-1, 3))
    if k == 0:
        raise ValueError("At least one point is required")
    check(locals(), "along", (3,))
    coords_on_axis = points.dot(along)
    return np.argmax(coords_on_axis)


def apex(points, along):
    return points[argapex(points=points, along=along)].copy()


def apex_and_opposite(points, along):
    k = check(locals(), "points", (-1, 3))
    if k == 0:
        raise ValueError("At least one point is required")
    check(locals(), "along", (3,))
    coords_on_axis = points.dot(along)
    apex_idx = np.argmax(coords_on_axis)
    opposite_idx = np.argmin(coords_on_axis)
    return points[[apex_idx, opposite_idx]].copy()


def nearest(from_points, to_point, ret_index=False):
    check(locals(), "from_points", (-1, 3))
    check(locals(), "to_point", (3,))

    absolute_distances = magnitude(from_points - to_point)

    index_of_nearest_point = np.argmin(absolute_distances)
    nearest_point = from_points[index_of_nearest_point]

    if ret_index:
        return nearest_point, index_of_nearest_point
    else:
        return nearest_point


def farthest(from_points, to_point, ret_index=False):
    # Fixed: now routed through the shared check() helper instead of a
    # hand-rolled shape check with its own (inconsistent) error message.
    check(locals(), "from_points", (-1, 3))
    check(locals(), "to_point", (3,))

    absolute_distances = magnitude(from_points - to_point)

    index_of_farthest_point = np.argmax(absolute_distances)
    farthest_point = from_points[index_of_farthest_point]

    if ret_index:
        return farthest_point, index_of_farthest_point
    else:
        return farthest_point


def within(points, radius, of_point, atol=1e-08, ret_indices=False):
    # Fixed: isinstance(radius, float) rejected plain ints and numpy scalar
    # types (e.g. np.float64), which is what most numpy pipelines actually
    # produce. numbers.Real covers int, float, and numpy numeric scalars.
    check(locals(), "points", (-1, 3))
    if not isinstance(radius, numbers.Real):
        raise ValueError("radius should be a real number")
    check(locals(), "of_point", (3,))

    absolute_distances = magnitude(points - of_point)
    (indices_within_radius,) = (absolute_distances < radius + atol).nonzero()
    points_within_radius = points[indices_within_radius]
    if ret_indices:
        return points_within_radius, indices_within_radius
    else:
        return points_within_radius


def average(values, weights=None, ret_sum_of_weights=False):
    # Fixed: calling with ret_sum_of_weights=True and weights=None used to
    # crash on np.sum(None). The sum of k implicit unit weights is just k.
    k = check(locals(), "values", (-1, 3))
    if weights is not None:
        weights = np.array(weights)
        check(locals(), "weights", (k,))
    result = np.average(values, axis=0, weights=weights)
    if ret_sum_of_weights:
        sum_of_weights = np.sum(weights) if weights is not None else float(k)
        return result, sum_of_weights
    else:
        return result