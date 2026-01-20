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

from . import core
from . import io_utils
from . import erros



# from .tree_functions import vertex_coordinates
# from .tree_class import Tree_graph
# from .io_utils import read_swc, write_swc#, tree_to_SWCtable

# Public API
__all__ = [
    "Tree_graph",
    "vertex_coordinates",
    "read_swc",
    "write_swc",
    # Add other public functions/classes as needed
]

# Version
__version__ = "0.1.0"