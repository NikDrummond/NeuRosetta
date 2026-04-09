"""
NeuRosetta: A Python package for morphological analysis and connectomics.

This package provides tools for working with tree-like structures,
spatial coordinates, and graph-based morphological data.
"""
from .api import Tree, Forest
from .io import import_swc, load
from .analysis import reduce_tree
from .ops.plotting import Viewer
from .gui import start_GUI

# Check for required conda dependencies
def _check_conda_deps():
    """Check if required conda packages are installed."""
    missing = []
    try:
        import graph_tool
    except ImportError:
        missing.append("graph-tool")

    try:
        import gudhi
    except ImportError:
        missing.append("gudhi")

    if missing:
        raise ImportError(
            f"Missing required packages: {missing}. "
            f"Install with: conda install -c conda-forge {' '.join(missing)}"
        )


_check_conda_deps()

from graph_tool.all import openmp_set_num_threads

openmp_set_num_threads(1)

# Public API
__all__ = ["Tree", "import_swc", "load", "Viewer",'reduce_tree','start_GUI']

# Version
__version__ = "0.1.0"
