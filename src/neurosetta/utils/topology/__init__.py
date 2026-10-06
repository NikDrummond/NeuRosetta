"""Topological data analysis utilities (TMD + persistence).

Operates on ``graph_tool.Graph`` / property maps / numpy arrays only.
Must not depend on ``_Tree`` or ``Forest``.
"""

from .diagrams import persistence_diagram, validate_diagram
from .distances import (
    barcode_distance,
    bottleneck_distance,
    pairwise_distances,
    wasserstein_distance,
)
from .plotting import plot_barcode, plot_persistence_diagram
from .representations import (
    fit_persistence_image,
    persistence_images,
    reshape_persistence_image,
)
from .tmd import TMD_GP, get_bound_tmd, normalize_tmd_payload, tmd
from .visitors import TMDVisitor

__all__ = [
    "TMDVisitor",
    "TMD_GP",
    "tmd",
    "get_bound_tmd",
    "normalize_tmd_payload",
    "persistence_diagram",
    "validate_diagram",
    "barcode_distance",
    "bottleneck_distance",
    "wasserstein_distance",
    "pairwise_distances",
    "fit_persistence_image",
    "persistence_images",
    "reshape_persistence_image",
    "plot_persistence_diagram",
    "plot_barcode",
]
