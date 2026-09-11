"""Anatomical frame: external spatial context for embedding metrics.

Topology and intrinsic geometry describe a neuron itself. Embedding metrics
describe that neuron relative to an external anatomical context. An
:class:`AnatomicalFrame` is a lightweight container for that context — meshes,
optional surface pairs, and named axes — without replacing existing geometry
classes or algorithms.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Hashable

import numpy as np
from numpy.linalg import norm

from ..core import _Mesh
from ..ops.neuropils.distances import (
    distance_from_neuropil_surface,
    neuropil_point_depth,
)
from ..utils.geometry_utils.angles import angle
from ..utils.units import units_are_equal


class FrameRequirementError(ValueError):
    """Raised when an AnatomicalFrame lacks a required component."""

    def __init__(
        self,
        message: str,
        *,
        metric: str | None = None,
        missing: Sequence[str] | None = None,
        frame: AnatomicalFrame | None = None,
    ) -> None:
        super().__init__(message)
        self.metric = metric
        self.missing = tuple(missing) if missing is not None else ()
        self.frame = frame


# Metric → alternatives of required component sets (OR of ANDs).
# Components: reference_mesh | inner_surface | outer_surface | axis
METRIC_FRAME_REQUIREMENTS: dict[str, tuple[tuple[str, ...], ...]] = {
    "distance_from_neuropil_surface": (("reference_mesh",),),
    "neuropil_point_depth": (
        ("reference_mesh",),
        ("inner_surface", "outer_surface"),
    ),
    "get_edge_angles": (("axis",),),
    "get_mean_edge_angle": (("axis",),),
    "get_edge_angle_variance": (("axis",),),
    "coordinate_mean_along_axis": (("axis",),),
    "coordinate_variance_along_axis": (("axis",),),
    "coordinate_std_along_axis": (("axis",),),
    "coordinate_minmax_along_axis": (("axis",),),
    "coordinate_extent_along_axis": (("axis",),),
    "coordinate_rms_along_axis": (("axis",),),
    "coordinate_mean_absolute_along_axis": (("axis",),),
    "coordinate_projection_moments": (("axis",),),
}


def _is_mesh_like(obj: Any) -> bool:
    return obj is not None and hasattr(obj, "mesh") and not isinstance(
        obj, (list, tuple, np.ndarray, str, dict)
    )


def _normalize_axis(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"axis {name!r} must contain three finite values, got {arr}")
    mag = float(norm(arr))
    if mag == 0.0:
        raise ValueError(f"axis {name!r} has zero magnitude")
    return arr / mag


class AnatomicalFrame:
    """External anatomical / spatial context for embedding analysis.

    Parameters
    ----------
    name : hashable, optional
        Human-readable identifier (e.g. neuropil or layer name).
    reference_mesh : Neuropil or _Mesh, optional
        Mesh used for surface distance and for single-mesh normalized depth
        (inner/outer faces classified from normals).
    inner_surface, outer_surface : Neuropil or _Mesh, optional
        Explicit surface pair for normalized depth. Must be supplied together.
        When both are set, depth uses distances to each surface (same
        normalisation as the low-level single-mesh path).
    axes : mapping of str → length-3 vector, optional
        Named anatomical axes. Stored as unit vectors.
    default_axis : str, optional
        Axis name used by embedding metrics that need a single direction.
        Defaults to ``"depth"`` if present, else the first axis key.
    metadata : dict, optional
        Optional reproducibility fields (``units``, ``source``, ``version``,
        ``coordinate_space``, …). Geometry is not serialized here.
    depth_t : float, optional
        Dot-product threshold forwarded to single-mesh depth. By default 0.0.
    depth_surface : {"inner", "outer"}, optional
        Which surface depth is measured from. By default ``"inner"``.
    depth_norm : bool, optional
        Normalize depth by inner+outer distances. By default True.

    Notes
    -----
    Low-level APIs that take an explicit mesh or axis remain supported.
    Prefer :class:`AnatomicalFrame` for high-level workflows such as
    ``nr.describe(..., domains="embedding", reference_frame=frame)``.
    """

    __slots__ = (
        "name",
        "reference_mesh",
        "inner_surface",
        "outer_surface",
        "axes",
        "default_axis",
        "metadata",
        "depth_t",
        "depth_surface",
        "depth_norm",
    )

    def __init__(
        self,
        name: Hashable | None = None,
        *,
        reference_mesh: _Mesh | None = None,
        inner_surface: _Mesh | None = None,
        outer_surface: _Mesh | None = None,
        axes: Mapping[str, Any] | None = None,
        default_axis: str | None = None,
        metadata: dict | None = None,
        depth_t: float = 0.0,
        depth_surface: str = "inner",
        depth_norm: bool = True,
    ) -> None:
        if (inner_surface is None) ^ (outer_surface is None):
            raise ValueError(
                "inner_surface and outer_surface must be supplied together "
                "(or neither)"
            )
        for label, obj in (
            ("reference_mesh", reference_mesh),
            ("inner_surface", inner_surface),
            ("outer_surface", outer_surface),
        ):
            if obj is not None and not _is_mesh_like(obj):
                raise TypeError(
                    f"{label} must be a Neuropil/_Mesh-like object with a .mesh "
                    f"attribute, got {type(obj)!r}"
                )
        if depth_surface not in {"inner", "outer"}:
            raise ValueError("depth_surface must be 'inner' or 'outer'")

        normalized_axes: dict[str, np.ndarray] = {}
        if axes is not None:
            for key, value in axes.items():
                normalized_axes[str(key)] = _normalize_axis(value, name=str(key))

        if default_axis is not None:
            if default_axis not in normalized_axes:
                raise ValueError(
                    f"default_axis {default_axis!r} not in axes "
                    f"{sorted(normalized_axes)}"
                )
        elif "depth" in normalized_axes:
            default_axis = "depth"
        elif normalized_axes:
            default_axis = next(iter(normalized_axes))

        self.name = name
        self.reference_mesh = reference_mesh
        self.inner_surface = inner_surface
        self.outer_surface = outer_surface
        self.axes = normalized_axes
        self.default_axis = default_axis
        self.metadata = dict(metadata) if metadata is not None else {}
        self.depth_t = float(depth_t)
        self.depth_surface = depth_surface
        self.depth_norm = bool(depth_norm)

    # ------------------------------------------------------------------
    # Presence / validation
    # ------------------------------------------------------------------

    def has(self, component: str) -> bool:
        """Return True if *component* is available on this frame."""
        if component == "reference_mesh":
            return self.reference_mesh is not None
        if component == "inner_surface":
            return self.inner_surface is not None
        if component == "outer_surface":
            return self.outer_surface is not None
        if component == "axis":
            return bool(self.axes)
        if component.startswith("axis:"):
            return component.split(":", 1)[1] in self.axes
        raise ValueError(f"unknown frame component {component!r}")

    def available_components(self) -> tuple[str, ...]:
        """Return names of components present on this frame."""
        names: list[str] = []
        for key in ("reference_mesh", "inner_surface", "outer_surface"):
            if self.has(key):
                names.append(key)
        if self.axes:
            names.append("axis")
            names.extend(f"axis:{k}" for k in self.axes)
        return tuple(names)

    def require(
        self,
        *components: str,
        metric: str | None = None,
    ) -> None:
        """Raise :class:`FrameRequirementError` if any *components* are missing."""
        missing = [c for c in components if not self.has(c)]
        if not missing:
            return
        metric_bit = f" for metric {metric!r}" if metric else ""
        raise FrameRequirementError(
            f"AnatomicalFrame {self.name!r} missing {missing}{metric_bit}; "
            f"available={list(self.available_components())}",
            metric=metric,
            missing=missing,
            frame=self,
        )

    def require_axis(self, name: str | None = None, *, metric: str | None = None) -> np.ndarray:
        """Return a named (or default) unit axis, or raise."""
        axis_name = name if name is not None else self.default_axis
        if axis_name is None:
            raise FrameRequirementError(
                f"AnatomicalFrame {self.name!r} has no axes"
                + (f" (required by metric {metric!r})" if metric else ""),
                metric=metric,
                missing=("axis",),
                frame=self,
            )
        if axis_name not in self.axes:
            raise FrameRequirementError(
                f"AnatomicalFrame {self.name!r} has no axis {axis_name!r}"
                + (f" (required by metric {metric!r})" if metric else "")
                + f"; available axes={sorted(self.axes)}",
                metric=metric,
                missing=(f"axis:{axis_name}",),
                frame=self,
            )
        return self.axes[axis_name]

    def satisfies(self, alternatives: Sequence[Sequence[str]]) -> bool:
        """True if any alternative requirement set is fully present."""
        return any(all(self.has(c) for c in group) for group in alternatives)

    def check_metric(self, metric_name: str) -> bool:
        """True if this frame can support *metric_name*."""
        reqs = METRIC_FRAME_REQUIREMENTS.get(metric_name)
        if reqs is None:
            return True
        return self.satisfies(reqs)

    def require_for_metric(self, metric_name: str) -> None:
        """Validate frame components required by a registered embedding metric."""
        reqs = METRIC_FRAME_REQUIREMENTS.get(metric_name)
        if reqs is None:
            return
        if self.satisfies(reqs):
            return
        alts = " or ".join("(" + ", ".join(g) + ")" for g in reqs)
        raise FrameRequirementError(
            f"AnatomicalFrame {self.name!r} cannot satisfy metric {metric_name!r}; "
            f"need one of {alts}; available={list(self.available_components())}",
            metric=metric_name,
            missing=tuple(dict.fromkeys(c for g in reqs for c in g)),
            frame=self,
        )

    # ------------------------------------------------------------------
    # Geometry (delegate to existing ops)
    # ------------------------------------------------------------------

    def surface_distance(self, points: np.ndarray) -> np.ndarray:
        """Distance from *points* to :attr:`reference_mesh`."""
        self.require("reference_mesh")
        return distance_from_neuropil_surface(self.reference_mesh, points)

    def normalized_depth(
        self,
        points: np.ndarray,
        *,
        t: float | None = None,
        surface: str | None = None,
        norm: bool | None = None,
    ) -> np.ndarray:
        """Normalized anatomical depth for *points*.

        Uses :attr:`reference_mesh` via :func:`~neurosetta.neuropil_point_depth`
        when available. Otherwise uses distances to
        :attr:`inner_surface` / :attr:`outer_surface` with the same
        normalisation ``d / (d_inner + d_outer)``.
        """
        t_val = self.depth_t if t is None else t
        surface_val = self.depth_surface if surface is None else surface
        norm_val = self.depth_norm if norm is None else norm

        if self.reference_mesh is not None:
            return neuropil_point_depth(
                self.reference_mesh,
                points,
                t=t_val,
                surface=surface_val,
                norm=norm_val,
            )

        self.require("inner_surface", "outer_surface")
        in_dists = distance_from_neuropil_surface(self.inner_surface, points)
        out_dists = distance_from_neuropil_surface(self.outer_surface, points)
        if surface_val == "inner":
            dists = in_dists
        elif surface_val == "outer":
            dists = out_dists
        else:
            raise ValueError("surface must be 'inner' or 'outer'")
        if norm_val:
            denom = in_dists + out_dists
            with np.errstate(divide="ignore", invalid="ignore"):
                dists = np.where(denom > 0, dists / denom, np.nan)
        return dists

    def angle_to_axis(
        self,
        vector: Any,
        axis: str | None = None,
        *,
        degrees: bool = False,
    ) -> float:
        """Angle between *vector* and a named anatomical axis.

        Uses :func:`~neurosetta.utils.geometry_utils.angles.angle`.
        """
        axis_vec = self.require_axis(axis)
        v = np.asarray(vector, dtype=float).reshape(3)
        if not np.all(np.isfinite(v)) or float(norm(v)) == 0.0:
            raise ValueError("vector must be a finite non-zero length-3 array")
        return float(
            angle(v[0], v[1], v[2], axis_vec[0], axis_vec[1], axis_vec[2], degrees=degrees)
        )

    def units(self) -> str | None:
        """Optional spatial units from metadata, if present."""
        value = self.metadata.get("units")
        return None if value is None else str(value)

    def check_units_compatible(self, other_units: str | None) -> None:
        """Raise if frame and *other_units* are both set and disagree."""
        frame_units = self.units()
        if frame_units is None or other_units is None:
            return
        if not units_are_equal(frame_units, other_units):
            raise ValueError(
                f"AnatomicalFrame units {frame_units!r} incompatible with "
                f"{other_units!r}"
            )

    def describe(self, obj, **kwargs):
        """Convenience wrapper for ``nr.describe(obj, reference_frame=self, ...)``."""
        from ..utils.metrics.descriptors import describe as _describe

        kwargs.setdefault("domains", "embedding")
        return _describe(obj, reference_frame=self, **kwargs)

    def __repr__(self) -> str:
        axis_keys = list(self.axes.keys())
        return (
            f"AnatomicalFrame("
            f"name={self.name!r}, "
            f"reference_mesh={self.reference_mesh is not None}, "
            f"inner_surface={self.inner_surface is not None}, "
            f"outer_surface={self.outer_surface is not None}, "
            f"axes={axis_keys}"
            f")"
        )


def frame_requirements_for(metric_name: str) -> tuple[tuple[str, ...], ...] | None:
    """Return requirement alternatives for *metric_name*, or None if unconstrained."""
    return METRIC_FRAME_REQUIREMENTS.get(metric_name)


__all__ = [
    "AnatomicalFrame",
    "FrameRequirementError",
    "METRIC_FRAME_REQUIREMENTS",
    "frame_requirements_for",
]
