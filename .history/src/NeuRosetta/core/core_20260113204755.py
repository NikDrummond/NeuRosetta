from graph_tool.all import Graph

class _Stone(object):
    """Core single item class"""
    
    # constructor
    def __init__(self, name:int, units:str, meta:dict) -> None:
        self.ame = name
        self.units = units
        self.meta = meta

class _Tree(_Stone):
    """Underlying tree graph class"""
    
    # constructor
    def __init__(self, ID:int, units:str, meta:dict, graph:Graph) -> None:
        super().__init__(ID,units,meta)
        self.graph = graph