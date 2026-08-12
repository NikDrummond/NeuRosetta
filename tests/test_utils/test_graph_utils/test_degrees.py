"""Tests for graph_utils.degrees."""

import pytest

from neurosetta.utils.graph_utils.degrees import degree_distribution, get_vertex_degrees


def test_get_vertex_degrees_out(simple_tree):
    out_deg = get_vertex_degrees(simple_tree, deg="out")
    assert out_deg.sum() == 16


def test_get_vertex_degrees_in(simple_tree):
    in_deg = get_vertex_degrees(simple_tree, deg="in")
    assert in_deg.sum() == 16


def test_get_vertex_degrees_total(simple_tree):
    total = get_vertex_degrees(simple_tree, deg="total")
    assert len(total) == 17


def test_degree_distribution_normalized(simple_tree):
    degrees, counts = degree_distribution(simple_tree, mass=True)
    assert counts.sum() == pytest.approx(1.0)


def test_degree_distribution_raw_counts(simple_tree):
    degrees, counts = degree_distribution(simple_tree, mass=False)
    assert counts.sum() == 17


def test_invalid_degree_type_raises(simple_tree):
    with pytest.raises(ValueError, match="deg must be"):
        get_vertex_degrees(simple_tree, deg="sideways")
