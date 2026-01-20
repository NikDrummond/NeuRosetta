from graph_tool.all import Graph

class _Stone(object):
    """Core single item class"""
    
    # constructor
    def __init__(self, ID:int, metadata:dict) -> None:
        self.ID = ID
        self.metadata = metadata

class _Tree(_Stone):
    """Underlying tree graph class"""
    
    # constructor
    def __init__(self, ID:int, metadata:dict, graph:Graph) -> None:
        super().__init__(ID,metadata)
        self.graph = graph