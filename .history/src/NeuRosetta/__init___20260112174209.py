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

from .classes import Tree_graph
from .tree_graphs import vertex_coordinates
from .io_utils import import_swc, write_swc

# Public API
__all__ = [
    # core class
    "Tree_graph",
    ### tree graphs
    "import_swc",

    # Add other public functions/classes as needed
]

# Version
__version__ = "0.1.0"