"""Low-level graph utility helpers."""

from .gt_properties import (
    g_has_property,
    raise_internal_property_missing,
    bind_vertex_property,
    bind_edge_property,
    bind_graph_property,
)

from .traversals import(
    bfsearch,
    bf_iterator,
    dfsearch,
    df_iterator,
    TreeDepthVisitor,
    PostOrderVisitor,
    ReduceVisitor,
    SubtreeMaskVisitor,
)

from .vertex_inds import (
    root_index,
    leaf_indices,
    branch_indices,
    core_indices,
    edge_indices,
    subtree_indices
)
from .counting import (
    count_roots,
    count_vertices,
    count_edges,
    count_leaves,
    count_branches,
    count_transitive_vertices,
)

from .coordinates import (
    vertex_coordinates,
    vertex_coordinates_subtree,
    edge_coordinates,
    edge_coordinates_subtree
)

from .graph_editing import (
    reduce_graph,
    reroot_graph
)

from .subgraphs import(
    subgraph_score,
    max_subgraph_ind,
)

from .degrees import (
    get_vertex_degrees,
    degree_distribution
)

__all__ = [
    "g_has_property",
    "raise_internal_property_missing",
    "bind_vertex_property",
    "bind_edge_property",
    "bind_graph_property",
    "root_index",
    "leaf_indices",
    "branch_indices",
    "core_indices",
    "edge_indices",
    "subtree_indices",
    "count_roots",
    "count_vertices",
    "count_edges",
    "count_leaves",
    "count_branches",
    "count_transitive_vertices",
    "bfsearch",
    "bf_iterator",
    "dfsearch",
    "df_iterator",
    "TreeDepthVisitor",
    "PostOrderVisitor",
    "ReduceVisitor",
    "SubtreeMaskVisitor",
    "vertex_coordinates",
    "vertex_coordinates_subtree",
    "edge_coordinates",
    "edge_coordinates_subtree",
    "reduce_graph",
    "reroot_graph",
    "subgraph_score",
    "max_subgraph_ind",
    "get_vertex_degrees",
    "degree_distribution",
]
