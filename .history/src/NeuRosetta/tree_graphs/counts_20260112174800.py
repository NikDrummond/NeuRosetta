### boring functions for counting things
from numpy import 

def count_roots(tree: _Tree) -> int:
    return len(where(tree.graph.degree_property_map("in").a == 0))