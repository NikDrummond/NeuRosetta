"""Persistence-diagram conversion from TMD results."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def validate_diagram(diagram: Any) -> NDArray[np.float64]:
    """Validate and coerce a persistence diagram to shape ``(N, 2)``.

    Checks numpy-compatibility, two columns, and finite values.
    """
    arr = np.asarray(diagram, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Persistence diagram must have shape (N, 2); got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Persistence diagram contains non-finite values")
    return arr


def persistence_diagram(
    tmd_result: dict[str, Any],
    normalize: bool = True,
) -> NDArray[np.float64]:
    """Convert a TMD result dict to a persistence diagram of shape ``(N, 2)``.

    NeuRosetta preserves original TMD pair indices on the TMD result. This
    helper converts filtration values into diagram coordinates. Generic GUDHI
    operations should use ``normalize=True`` so that
    ``diagram[:, 0] <= diagram[:, 1]``.

    Parameters
    ----------
    tmd_result : dict
        Output of :func:`~neurosetta.utils.topology.tmd.tmd`.
    normalize : bool, optional
        If True, order endpoints as ``(low, high)``. If False, preserve raw
        TMD ``(birth, death)`` order (branch/root value, leaf value).
        By default True.

    Returns
    -------
    ndarray
        Shape ``(n_pairs, 2)``. Empty input yields shape ``(0, 2)``.
    """
    birth = np.asarray(tmd_result["birth"], dtype=np.float64).ravel()
    death = np.asarray(tmd_result["death"], dtype=np.float64).ravel()

    if birth.size == 0 and death.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if birth.shape != death.shape:
        raise ValueError(f"TMD birth/death length mismatch: {birth.shape} vs {death.shape}")

    if normalize:
        low = np.minimum(birth, death)
        high = np.maximum(birth, death)
        return np.column_stack((low, high))

    return np.column_stack((birth, death))
