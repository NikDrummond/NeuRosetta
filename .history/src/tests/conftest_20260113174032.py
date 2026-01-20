import pytest
from graph_tool.all import Graph
from NeuRosetta.core import _Tree, _Stone
from numpy import array

@pytest.fixture
def simple_tree():
    """create a simple test tree graph"""
    edges = array([[ 0,  1],
       [ 0,  3],
       [ 1,  2],
       [ 3,  4],
       [ 3, 14],
       [ 4,  5],
       [ 5,  6],
       [ 6,  7],
       [ 6,  9],
       [ 7,  8],
       [ 9, 10],
       [10, 11],
       [10, 12],
       [10, 13],
       [14, 15],
       [15, 16]])
    g = Graph(edges, directed = True, hashed = True)
    return g

@pytest.fixture
def test_tree(simple_tree):
    """test _Tree instance"""
    return 