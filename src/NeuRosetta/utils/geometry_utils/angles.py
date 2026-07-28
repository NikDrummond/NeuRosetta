# angles.py

import math
import numpy as np

from ._validation import check, check_value_any
from .algebra import magnitude, reject, scale_factor


def angle(v1, v2, look=None, assume_normalized=False, units="deg"):
    if units not in ["deg", "rad"]:
        raise ValueError(f"Unrecognized units {units}; expected deg or rad")

    # Previously unvalidated -- mismatched shapes fell straight through to a
    # raw einsum/broadcasting error instead of a clear ValueError.
    k = check_value_any(v1, (3,), (-1, 3), name="v1")
    check_value_any(v2, (3,), (-1 if k is None else k, 3), name="v2")
    if look is not None:
        check_value_any(look, (3,), (-1 if k is None else k, 3), name="look")
        v1, v2 = [reject(v, from_v=look) for v in (v1, v2)]

    dot_products = np.einsum("ij,ij->i", v1.reshape(-1, 3), v2.reshape(-1, 3))

    if assume_normalized:
        cosines = dot_products
    else:
        cosines = dot_products / magnitude(v1) / magnitude(v2)

    # Clip, because the dot product can slip past 1 or -1 due to rounding and
    # we can't compute arccos(-1.00001).
    angles = np.arccos(np.clip(cosines, -1.0, 1.0))
    if units == "deg":
        angles = np.degrees(angles)

    return angles[0] if v1.ndim == 1 and v2.ndim == 1 else angles


def signed_angle(v1, v2, look, units="deg"):
    # The sign of (A x B) dot look gives the sign of the angle.
    # > 0 means clockwise, < 0 is counterclockwise.
    sign = np.array(np.sign(np.cross(v1, v2).dot(look)))

    # 0 means collinear: 0 or 180. Let's call that clockwise.
    sign[sign == 0] = 1

    return sign * angle(v1, v2, look, units=units)


def rotate(vector, around_axis, angle, units="deg", assume_normalized=False):
    # Previously unvalidated, same rationale as angle() above.
    check_value_any(vector, (3,), (-1, 3), name="vector")
    check(locals(), "around_axis", (3,))

    if units == "deg":
        angle = math.radians(angle)
    elif units != "rad":
        raise ValueError(f'Unknown units "{units}"; expected "deg" or "rad"')

    cosine = math.cos(angle)
    sine = math.sin(angle)

    if not assume_normalized:
        from .algebra import normalize
        around_axis = normalize(around_axis)

    if vector.ndim == 1:
        dot_products = np.inner(around_axis, vector)
    elif vector.ndim == 2:
        dot_products = np.inner(around_axis, vector)[:, np.newaxis]
    else:
        from ._validation import raise_dimension_error
        raise_dimension_error(vector)

    # Rodrigues' rotation formula.
    return (
        cosine * vector
        + sine * np.cross(around_axis, vector)
        + (1 - cosine) * dot_products * around_axis
    )


def aligned_with(vector, along, reverse=False):
    from .algebra import project

    check(locals(), "vector", (3,))
    check(locals(), "along", (3,))

    projected = project(vector, onto=along)
    computed_scale_factor = scale_factor(projected, along)
    if not reverse and computed_scale_factor < 0:
        return -vector
    elif reverse and computed_scale_factor > 0:
        return -vector
    else:
        return vector