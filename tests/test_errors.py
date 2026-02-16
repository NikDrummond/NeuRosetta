# tests/test_errors.py
import pytest
from graph_tool.all import Graph
from NeuRosetta.errors import _InternalPropertyMissingError, _raise_internal_property

def test_internal_property_missing_error():
    error = _InternalPropertyMissingError("test_prop")
    assert str(error) == "Internal Property Missing: test_prop"
    assert error.missing_propertie == "test_prop"

def test_check_internal_property_missing():
    g = Graph(directed=True)
    g.add_vertex(1)
    
    with pytest.raises(_InternalPropertyMissingError) as exc_info:
        _raise_internal_property(g, "nonexistent_prop")
    assert "nonexistent_prop" in str(exc_info.value)

def test_check_internal_property_exists(simple_tree):
    # Assuming simple_graph has "ids" property
    _raise_internal_property(simple_tree, "ids")  # Should not raise