

class Tree_graph(_Tree):

    def __init__(self, ID:int, units:str, meta:dict, graph:Graph) -> None:
        super().__init__(ID = ID, units = units, meta = meta, graph = graph)
    # @staticmethod
    def get_coordinates(self, subset = None):
        return _vertex_coordinates(self, subset)