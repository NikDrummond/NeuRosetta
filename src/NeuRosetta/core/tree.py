""" base underlying _Tree class """
from graph_tool.all import Graph

from .stone import _Stone

class _Tree(_Stone):
    """Underlying tree graph class"""

    # constructor
    def __init__(self, ID: int, metadata: dict, graph: Graph) -> None:
        super().__init__(ID, metadata)
        self.graph = graph
