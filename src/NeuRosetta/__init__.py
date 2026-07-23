"""
NeuRosetta: A Python package for morphological analysis and connectomics.

This package provides tools for working with tree-like structures,
spatial coordinates, and graph-based morphological data.
"""

from .api import Tree, Forest, Tree_mesh, Neuropil, Forest_mesh, Neuropils
from .io import import_swc, export_swc, load, save, import_mesh, export_mesh
from .ops.plotting import Viewer
from .gui import start_GUI
from .analysis import reconstruct_neuropil_surface


def _check_conda_deps():
    """Check if required conda packages are installed."""
    try:
        import graph_tool  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Missing required package: graph-tool. "
            "Install with: conda install -c conda-forge graph-tool"
        ) from exc


_check_conda_deps()

from graph_tool.all import openmp_set_num_threads

openmp_set_num_threads(1)

# Public API
__all__ = [
    "Tree",
    "Forest",
    "Tree_mesh",
    "Forest_mesh",
    "Neuropil",
    "Neuropils",
    "import_swc",
    "export_swc",
    "import_mesh",
    "export_mesh",
    "load",
    "save",
    "Viewer",
    "start_GUI",
    "reconstruct_neuropil_surface",
]

# Version
__version__ = "0.1.0"
