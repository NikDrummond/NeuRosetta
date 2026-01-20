import pytest
from graph_tool.all import Graph
from NeuRosetta.core import _Tree, _Stone
from numpy import array

@pytest.fixture
def simple_tree():
    """create a simple test tree graph"""
    edges = array([[ 0,  1],
       [ 1,  2],
       [ 0,  3],
       [ 3,  4],
       [ 4,  5],
       [ 5,  6],
       [ 6,  7],
       [ 7,  8],
       [ 6,  9],
       [ 9, 10],
       [10, 11],
       [10, 12],
       [10, 13],
       [ 3, 14],
       [14, 15],
       [15, 16]])