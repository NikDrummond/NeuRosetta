"""
NeuRosetta: A Python package for morphological analysis and connectomics.

This package provides tools for working with tree-like structures,
spatial coordinates, and graph-based morphological data.
"""
from .classes import Tree, Forest
from .io_utils import import_swc, load
from .tree_surgery import reduce_tree
from .plotting import Viewer

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
__all__ = ["Tree", "import_swc", "load", "Viewer",'reduce_tree']

# Version
__version__ = "0.1.0"
