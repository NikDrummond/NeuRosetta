"""Helpers for synapse overlays in 2D/3D plots."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy import asarray, ndarray

from ...core import _Tree
from ...core.synapses import TYPE_ALIASES, canonicalize_synapse_type
from ...utils.geometry_utils.pca import eig_decomp
from ...utils.geometry_utils.rotations import apply_rotation_steps, compute_alignment_rotation
from ..tree_graphs.tree_coordinates import get_node_coordinates

SynapseOverlay = Literal["pre", "post", "both"]


def align_points_like_tree(
    tree: _Tree,
    points: ndarray,
    *,
    robust: bool = False,
    b1: tuple = (0.0, 1.0, 0.0),
    b2: tuple = (1.0, 0.0, 0.0),
    b3: tuple = (0.0, 0.0, 0.1),
) -> ndarray:
    """Apply the same PCA alignment used by ``align_coordinates(bind=False)``.

    Used so synapse markers share the 2D ``force_perspective`` frame.
    """
    pts = asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)

    coords = get_node_coordinates(tree, SoA=True)
    mean = coords.mean(axis=1)
    x = coords[0] - mean[0]
    y = coords[1] - mean[1]
    z = coords[2] - mean[2]
    _, evecs = eig_decomp(x, y, z, robust=robust)
    step1, step2 = compute_alignment_rotation(evecs[:, 0], evecs[:, 1], evecs[:, 2], b1, b2, b3)

    sx = pts[:, 0] - mean[0]
    sy = pts[:, 1] - mean[1]
    sz = pts[:, 2] - mean[2]
    xr, yr, zr = apply_rotation_steps(sx, sy, sz, step1, step2)
    return asarray([xr, yr, zr]).T


def _coerce_overlay_mode(value: bool | str | None, *, name: str) -> SynapseOverlay | None:
    """Normalise a display flag to ``None`` / ``pre`` / ``post`` / ``both``."""
    if value is None or value is False:
        return None
    if value is True:
        return "both"
    if not isinstance(value, str):
        raise TypeError(f"{name} must be bool, str, or None; got {type(value)!r}")
    key = value.strip().lower()
    if key in ("both", "all"):
        return "both"
    if key in TYPE_ALIASES:
        return canonicalize_synapse_type(key)  # type: ignore[return-value]
    if key in ("pre", "post"):
        return key  # type: ignore[return-value]
    raise ValueError(
        f"{name}={value!r} invalid; expected True/False/None/'pre'/'post'/'both' "
        f"(or input/output aliases)"
    )


def resolve_synapse_overlay(
    synapses: bool | str | None = None,
    show_synapses: bool | str | None = None,
) -> SynapseOverlay | None:
    """Resolve ``synapses=`` / ``show_synapses=`` into a single overlay mode.

    ``True`` means ``\"both\"``. ``False`` / ``None`` mean do not overlay.
    If both arguments resolve to different non-``None`` modes, raise.
    """
    a = _coerce_overlay_mode(synapses, name="synapses")
    b = _coerce_overlay_mode(show_synapses, name="show_synapses")
    if a is None:
        return b
    if b is None:
        return a
    if a != b:
        raise ValueError(
            f"Conflicting synapse overlay modes: synapses={synapses!r}, "
            f"show_synapses={show_synapses!r}"
        )
    return a


def type_mask(types: ndarray, mode: SynapseOverlay) -> ndarray:
    """Boolean mask selecting synapse rows for *mode*."""
    if mode == "both":
        return np.ones(len(types), dtype=bool)
    if mode == "pre":
        return types == "pre"
    if mode == "post":
        return types == "post"
    raise ValueError(f"Invalid synapse overlay mode {mode!r}")


def categorical_rgb(
    labels: ndarray,
    *,
    cmap: str = "tab10",
) -> tuple[ndarray, dict[Any, tuple[float, float, float]]]:
    """Map categorical labels to RGB colours in ``[0, 1]``.

    Returns
    -------
    colours : ndarray, shape (N, 3)
    legend : dict
        Unique label → RGB triple (stable order = sorted by ``str``).
    """
    import matplotlib.pyplot as plt

    labels = np.asarray(labels, dtype=object)
    # Stable category order for reproducible colours across calls.
    uniques = sorted(set(labels.tolist()), key=lambda x: str(x))
    cmap_obj = plt.colormaps[cmap]
    n = max(len(uniques), 1)
    legend: dict[Any, tuple[float, float, float]] = {}
    for i, lab in enumerate(uniques):
        rgba = cmap_obj(i / max(n - 1, 1) if n > 1 else 0.0)
        legend[lab] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
    colours = np.array([legend[lab] for lab in labels], dtype=float)
    return colours, legend
