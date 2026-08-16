"""Infer SWC node types from directed graph topology."""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import int32, ndarray, zeros_like


def infer_node_types(g: Graph) -> ndarray:
    """Infer node types from directed topology.

    Returns root (-1), branch (5), terminal (6), or transitory (0).
    """
    out_deg = g.get_out_degrees(g.get_vertices())
    in_deg = g.get_in_degrees(g.get_vertices())
    node_types = zeros_like(g.get_vertices(), dtype=int32)
    node_types[out_deg == 0] = 6
    node_types[out_deg > 1] = 5
    node_types[in_deg == 0] = -1
    return node_types
