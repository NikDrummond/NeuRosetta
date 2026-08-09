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
from .projections import (
    bisector,
    reflect_plane,
    mirror_axis,
)
from .angles import (
    angle,
    planar_angle,
    signed_angle,
    align_with,
    angular_mean,
    angular_var,
)
from .rotations import (
    rotate,
    rotate_about,
    minimum_rotation_to_align,
    compute_alignment_rotation,
    apply_rotation_steps,
)
from .transforms import (
    translate,
    recenter,
    scale_about,
    center_at_centroid,
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

from .scaling import scale_along_basis

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
    "bisector",
    "reflect_plane",
    "mirror_axis",
    "angle",
    "planar_angle",
    "signed_angle",
    "angular_mean",
    "angular_var",
    "rotate",
    "rotate_about",
    "minimum_rotation_to_align",
    "compute_alignment_rotation",
    "apply_rotation_steps",
    "align_with",
    "translate",
    "recenter",
    "scale_about",
    "center_at_centroid",
    "scale_along_basis",
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
