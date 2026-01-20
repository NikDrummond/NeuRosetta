# tests/test_core.py
import pytest
from graph_tool.all import Graph
from NeuRosetta.core import _Stone, _Tree

def test_stone_initialization():
    stone = _Stone(ID=1, units="um", meta={"key": "value"})
    assert stone.ID == 1
    assert stone.units == "um"
    assert stone.meta == {"key": "value"}

def test_tree_initialization(simple_graph):
    tree = _Tree(ID=1, units="um", meta={}, graph=simple_tree)
    assert tree.ID == 1
    assert isinstance(tree.graph, Graph)
    assert tree.graph.num_vertices() == 17

def test_tree_inherits_from_stone(simple_graph):
    tree = _Tree(ID=1, units="um", meta={"test": 123}, graph=simple_tree)
    assert isinstance(tree, _Stone)
    assert tree.meta == {"test": 123}