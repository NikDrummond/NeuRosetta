"""Thin GUDHI plotting wrappers for persistence diagrams / barcodes."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from ._gudhi import require_gudhi
from .diagrams import validate_diagram


def plot_persistence_diagram(
    diagram: np.ndarray,
    axes: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot a persistence diagram via GUDHI; return the matplotlib Axes."""
    gudhi = require_gudhi()
    d = validate_diagram(diagram)
    if axes is None:
        _, axes = plt.subplots()
    return gudhi.plot_persistence_diagram(d, axes=axes, **kwargs)


def plot_barcode(
    diagram: np.ndarray,
    axes: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot a persistence barcode via GUDHI; return the matplotlib Axes."""
    gudhi = require_gudhi()
    d = validate_diagram(diagram)
    if axes is None:
        _, axes = plt.subplots()
    return gudhi.plot_persistence_barcode(d, axes=axes, **kwargs)
