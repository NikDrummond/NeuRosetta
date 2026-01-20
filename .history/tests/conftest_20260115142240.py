import pytest
from graph_tool.all import Graph
from NeuRosetta.core import _Tree, _Stone
from numpy import array, ones_like


@pytest.fixture
def simple_tree() -> Graph:
    """create a simple test tree graph"""

    # core graph attributes (hardcoded)

    # edges
    edges = array(
        [
            [0, 1],
            [0, 3],
            [1, 2],
            [3, 4],
            [3, 14],
            [4, 5],
            [5, 6],
            [6, 7],
            [6, 9],
            [7, 8],
            [9, 10],
            [10, 11],
            [10, 12],
            [10, 13],
            [14, 15],
            [15, 16],
        ]
    )
    # node coordinates
    coords = array([[0,0,0],
            [0,1,0],
            [1,0,0],
            [1,1,0],
            [2,0,0],
            [1,-2,0],
            [2,1,0],
            [3,1,0],
            [3,2,0],
            [5,1,0],
            [4,2,0],
            [4,2,0],
            [5,0,0],
            [3.5,0,0],
            [5,-1,0],
            [7,0,0],
            [2,-2,0],
            [5,-2,0]])
    n_types = array([-1,0,5,6,0,0,0,5,0,0,6,5,6,6,6,0,6])
    radius = ones_like(n_types)

    # generate graph
    g = Graph(edges, directed=True, hashed=True)
    
    # add basic properties
    c_vp = g.new_vp("vector<double>")
    c_vp.set_2d_array(coords.T)
    g.vp['coordinates'] = c_vp
    g.vp['radius'] = g.new_vp('double', radius)
    g.vp['node_type'] = g.new_vp('int', n_types)
    
    return g


@pytest.fixture
def test_tree(simple_tree) -> _Tree:
    """test _Tree instance"""
    return _Tree(ID=1, meta={}, graph=simple_tree)
