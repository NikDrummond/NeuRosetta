from .basis import basis
from .algebra import (
    normalize,
    magnitude,
    dot,
    cross,
    scalar_projection,
    project,
    reject,
    perpendicular,
    scale_factor,
)
from .angles import angle, planar_angle, signed_angle, align_with
from .rotations import (
    rotate,
    minimum_rotation_to_align,
    compute_alignment_rotation,
    apply_rotation_steps,
)
from .tolerance import almost_zero, almost_unit_length, almost_collinear, almost_equal
from .points import (
    euclidean_distance,
    argapex,
    apex,
    apex_and_opposite,
    nearest,
    farthest,
    within_radius,
    average,
)
from .pca import eig_decomp
from ._validation import (
    check_value,
    check_value_any,
    check,
    columnize,
    raise_dimension_error,
)

__all__ = [
    "basis",
    "normalize",
    "magnitude",
    "dot",
    "cross",
    "scalar_projection",
    "project",
    "reject",
    "perpendicular",
    "scale_factor",
    "angle",
    "planar_angle",
    "signed_angle",
    "rotate",
    "minimum_rotation_to_align",
    "compute_alignment_rotation",
    "apply_rotation_steps",
    "align_with",
    "almost_zero",
    "almost_unit_length",
    "almost_collinear",
    "almost_equal",
    "euclidean_distance",
    "argapex",
    "apex",
    "apex_and_opposite",
    "nearest",
    "farthest",
    "within_radius",
    "average",
    "eig_decomp",
    "check_value",
    "check_value_any",
    "check",
    "columnize",
    "raise_dimension_error",
]
