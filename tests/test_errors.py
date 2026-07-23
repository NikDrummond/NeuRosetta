# tests/test_errors.py
import pytest
from graph_tool.all import Graph
from NeuRosetta.utils.graph_utils.gt_properties import (
    _InternalPropertyMissingError,
    raise_internal_property_missing,
)


def test_internal_property_missing_error():
    error = _InternalPropertyMissingError("test_prop", "v")
    assert "test_prop" in str(error)
    assert error.missing_property == "test_prop"


def test_check_internal_property_missing():
    g = Graph(directed=True)
    g.add_vertex(1)

    with pytest.raises(_InternalPropertyMissingError) as exc_info:
        raise_internal_property_missing(g, "nonexistent_prop")
    assert "nonexistent_prop" in str(exc_info.value)
