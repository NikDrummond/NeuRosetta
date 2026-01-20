from graph_tool.all import Graph
from numpy import ndarray

from ..core import _Tree
from ..tree_graphs.tree_functions import vertex_coordinates


class Tree_graph(_Tree):

    def __init__(self, ID:int, units:str, meta:dict, graph:Graph) -> None:
        super().__init__(ID = ID, units = units, meta = meta, graph = graph)

    def get_coordinates(self, subset = None) -> ndarray:
        return vertex_coordinates(self, subset)

    def write_swc(self, fpath: str, header = None):
        """"""
        from ..io_utils.swc_utils import write_swc
        write_swc(self, fpath, header)