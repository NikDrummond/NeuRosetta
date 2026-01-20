"""
NeuRosetta: A Python package for morphological analysis and connectomics.

This package provides tools for working with tree-like structures,
spatial coordinates, and graph-based morphological data.
"""

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

from .classes import Tree
from .io_utils import import_swc, load
from .plotting import Viewer

# Public API
__all__ = [
    "Tree",
    "import_swc",
    "load",
    "Vi"
]

# Version
__version__ = "0.1.0"