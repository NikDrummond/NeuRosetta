### various checks for tree graphs
from ..core import _Tree
from .counting import count_transitive_nodes

def check_reduced(tree: _Tree):
    """Check if the given graph, g, has no nodes with  """
    return count_transitive_nodes(tree) > 0