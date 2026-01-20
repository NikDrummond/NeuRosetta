from graph_tool.all import Graph
from .core import _Tree
from .tree_functions import vertex_coordinates

class Tree_graph(_Tree):

    def __init__(self, ID:int, units:str, meta:dict, graph:Graph) -> None:
        super().__init__(ID = ID, units = units, meta = meta, graph = graph)

    def get_coordinates(self, subset = None):
        return vertex_coordinates(self, subset)