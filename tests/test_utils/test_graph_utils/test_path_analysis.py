"""Path retrieval, distances, ancestry, and geometry on a known tree."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tree_builders import make_directed_tree

from neurosetta.utils.graph_utils import root_distance as root_from_package
from neurosetta.utils.graph_utils.paths import (
    ancestor_indices,
    branch_path,
    is_ancestor,
    lowest_common_ancestor,
    pairwise_path_lengths,
    parent_index,
    path_branch_count,
    path_coordinates,
    path_cumulative_length,
    path_edge_count,
    path_edge_values,
    path_edges,
    path_euclidean_distance,
    path_exists,
    path_length,
    path_lengths_to_vertices,
    path_section_count,
    path_tortuosity,
    path_vertex_values,
    path_vertices,
    root_distance,
    source_distances,
)

#           0
#          / \
#         1   2
#        / \   \
#       3   4   5
#          / \
#         6   7
_EDGES = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (4, 6), (4, 7)]
_PATH = [1.0, 2.0, 1.0, 1.0, 5.0, 6.0, 7.0]
_EUCL = [10.0, 3.0, 10.0, 4.0, 8.0, 7.0, 9.0]
_DELAY = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
_COORDS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (2.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
    (1.0, -1.0, 0.0),
]


def _tree(**kwargs):
    return make_directed_tree(
        _EDGES,
        coords=_COORDS,
        path_length=_PATH,
        euclidean=_EUCL,
        **kwargs,
    )


def _with_delay():
    g = _tree()
    g.ep["delay"] = g.new_ep("double", _DELAY)
    return g


def test_imports_stay_bound():
    assert root_from_package is root_distance


def test_directed_and_undirected_retrieval():
    g = _tree()
    assert_array_equal(path_vertices(g, 0, 6), [0, 1, 4, 6])
    assert_array_equal(path_edges(g, 0, 1), [[0, 1]])
    assert path_vertices(g, 3, 4) is None
    assert path_exists(g, 3, 4) is False
    assert path_exists(g, 1, 6) is True
    assert_array_equal(path_vertices(g, 3, 4, directed=False), [3, 1, 4])
    assert_array_equal(path_edges(g, 3, 4, directed=False), [[3, 1], [1, 4]])
    assert_array_equal(path_vertices(g, 3, 6, directed=False), [3, 1, 4, 6])
    assert_array_equal(path_edges(g, 6, 7, directed=False), [[6, 4], [4, 7]])
    assert path_exists(g, 6, 3, directed=False) is True
    assert_array_equal(path_vertices(g, 4, 4), [4])
    assert path_edges(g, 4, 4).shape == (0, 2)
    assert path_edge_count(g, 4, 4) == 0
    assert path_edge_count(g, 0, 6) == 3
    assert path_edge_count(g, 6, 0) is None


def test_invalid_indices_and_filter():
    g = _tree()
    with pytest.raises(ValueError):
        path_vertices(g, 0, 99)
    with pytest.raises(TypeError):
        path_vertices(g, True, 1)
    mask = g.new_vp("bool")
    mask.a = True
    g.set_vertex_filter(mask)
    with pytest.raises(ValueError, match="filter"):
        path_vertices(g, 0, 1)


def test_lengths_properties_and_cumulative():
    g = _with_delay()
    before = np.array(g.ep["Path_length"].a, copy=True)
    names = set(g.vp.keys())
    assert path_length(g, 0, 6) == 8.0
    assert path_length(g, 0, 6, ep="Euclidean_length") == 21.0
    assert path_length(g, 0, 6, ep="delay") == 11.0
    assert path_length(g, 0, 6, ep=g.ep["delay"]) == 11.0
    assert path_length(g, 3, 6, directed=False) == 8.0
    assert path_length(g, 6, 1) is None
    assert path_length(g, 5, 5) == 0.0
    cumulative = path_cumulative_length(g, 3, 6, directed=False)
    assert_allclose(cumulative, [0.0, 1.0, 2.0, 8.0])
    assert cumulative[-1] == path_length(g, 3, 6, directed=False)
    assert_array_equal(g.ep["Path_length"].a, before)
    assert set(g.vp.keys()) == names

    only_eucl = _tree()
    del only_eucl.ep["Path_length"]
    assert path_length(only_eucl, 0, 1) == 10.0
    with pytest.raises(ValueError, match="not on the graph"):
        path_length(g, 0, 1, ep="missing")


def test_weight_validation():
    g = _tree()
    g.ep["bad"] = g.new_ep("double", [np.nan, *([1.0] * 6)])
    with pytest.raises(ValueError, match="finite"):
        path_length(g, 0, 1, ep="bad")
    g.ep["signed"] = g.new_ep("double", [-1.0, *([1.0] * 6)])
    with pytest.raises(ValueError, match="nonnegative"):
        path_length(g, 0, 1, ep="signed")
    g.ep["zero"] = g.new_ep("double", np.zeros(7))
    assert path_length(g, 0, 6, ep="zero") == 0.0


def test_source_and_batch_distances():
    g = _with_delay()
    dist = source_distances(g, 1)
    values = np.array(dist.a, copy=True)
    assert values[1] == 0.0
    assert_allclose(values[6], 7.0)
    assert np.isinf(values[0])
    assert np.isinf(values[5])
    assert "Root_distance" not in g.vp
    undirected = np.array(source_distances(g, 1, directed=False).a, copy=True)
    assert np.isfinite(undirected).all()
    assert_allclose(undirected[0], 1.0)

    targets = path_lengths_to_vertices(g, 1, [3, 6, 2])
    assert_allclose(targets, [1.0, 7.0, np.inf])
    assert path_lengths_to_vertices(g, 1, []).shape == (0,)

    pairs = [[1, 3], [1, 6], [3, 6], [6, 1], [1, 3]]
    directed = pairwise_path_lengths(g, pairs)
    assert_allclose(directed, [1.0, 7.0, np.inf, np.inf, 1.0])
    undirected_pairs = pairwise_path_lengths(g, [[6, 7], [3, 6]], directed=False)
    assert_allclose(undirected_pairs, [13.0, 8.0])
    assert pairwise_path_lengths(g, []).shape == (0,)
    custom = pairwise_path_lengths(g, [[0, 6]], ep="delay")
    assert_allclose(custom, [11.0])
    with pytest.raises(ValueError, match="shape"):
        pairwise_path_lengths(g, [[0, 1, 2]])


def test_root_distance_keeps_explicit_property():
    g = _tree()
    dist = root_distance(g)
    assert float(np.array(dist.a, copy=True)[0]) == 0.0
    assert_allclose(np.array(dist.a, copy=True)[3], 2.0)
    with pytest.raises(Exception, match="Path_length"):
        root_distance(_tree_without_path(), ep="Path_length")


def _tree_without_path():
    return make_directed_tree(_EDGES, coords=_COORDS, euclidean=_EUCL)


def test_ancestry_and_lca():
    g = _tree()
    assert parent_index(g, 0) is None
    assert parent_index(g, 6) == 4
    assert_array_equal(ancestor_indices(g, 6), [4, 1, 0])
    assert_array_equal(ancestor_indices(g, 6, include_self=True), [6, 4, 1, 0])
    assert_array_equal(ancestor_indices(g, 0), [])
    assert is_ancestor(g, 6, 6) is True
    assert is_ancestor(g, 1, 6) is True
    assert is_ancestor(g, 6, 1) is False

    assert lowest_common_ancestor(g, 6, 7) == 4
    assert lowest_common_ancestor(g, 3, 6, 7) == 1
    assert lowest_common_ancestor(g, [3, 4, 6, 7]) == 1
    assert lowest_common_ancestor(g, 3, 5, 7) == 0
    assert lowest_common_ancestor(g, [6, 7, 4]) == 4
    assert lowest_common_ancestor(g, [6]) == 6
    assert lowest_common_ancestor(g, [6, 6, 6]) == 6
    assert lowest_common_ancestor(g, [0, 6, 7]) == 0
    assert lowest_common_ancestor(g, (6, 7)) == 4
    assert lowest_common_ancestor(g, np.array([3, 5, 7])) == 0
    assert lowest_common_ancestor(g, (v for v in (6, 7))) == 4
    with pytest.raises(ValueError):
        lowest_common_ancestor(g, [])
    with pytest.raises(ValueError):
        lowest_common_ancestor(g)
    with pytest.raises(ValueError):
        lowest_common_ancestor(g, 0, 99)

    assert_array_equal(branch_path(g, 3, 7, directed=False), [1, 4])
    assert path_branch_count(g, 0, 6) == 2
    assert path_branch_count(g, 0, 6, include_endpoints=True) == 3
    assert branch_path(g, 6, 0) is None


def test_lca_multifurcation_transitive_and_nonzero_root():
    multi = make_directed_tree([(0, 1), (0, 2), (0, 3)])
    assert lowest_common_ancestor(multi, 1, 2, 3) == 0
    chain = make_directed_tree([(0, 1), (1, 2), (2, 3)])
    assert lowest_common_ancestor(chain, 3, 2) == 2
    assert_array_equal(ancestor_indices(chain, 3), [2, 1, 0])
    shifted = make_directed_tree([(2, 0), (2, 1), (0, 3)])
    assert parent_index(shifted, 2) is None
    assert lowest_common_ancestor(shifted, 3, 1) == 2
    single = make_directed_tree([], n=1)
    assert parent_index(single, 0) is None
    assert lowest_common_ancestor(single, 0) == 0
    assert_array_equal(path_vertices(single, 0, 0), [0])
    assert path_length(single, 0, 0, ep=single.new_ep("double")) == 0.0


def test_geometry_tortuosity_and_profiles():
    g = _with_delay()
    coords = path_coordinates(g, 0, 3)
    assert_allclose(coords, [_COORDS[0], _COORDS[1], _COORDS[3]])
    assert path_coordinates(g, 3, 0) is None
    straight = path_euclidean_distance(g, 0, 3)
    assert_allclose(straight, 2.0)
    assert path_euclidean_distance(g, 3, 0) == straight
    assert_allclose(path_tortuosity(g, 0, 3), 1.0)
    assert_allclose(path_tortuosity(g, 0, 4), np.sqrt(2.0))
    assert path_tortuosity(g, 0, 4) > path_tortuosity(g, 0, 3)
    assert np.isnan(path_tortuosity(g, 4, 4))
    assert path_tortuosity(g, 6, 0) is None
    bare = _tree()
    del bare.ep["Path_length"]
    with pytest.raises(ValueError, match="Path_length"):
        path_tortuosity(bare, 0, 3)
    # Explicit Euclidean_length is a different ratio: 20 / 2.
    assert_allclose(path_tortuosity(g, 0, 3, ep="Euclidean_length"), 10.0)

    assert_allclose(path_vertex_values(g, 0, 3, "x"), [0.0, 1.0, 2.0])
    assert path_vertex_values(g, 3, 6, "x") is None
    assert_allclose(path_edge_values(g, 3, 6, "Path_length", directed=False), [1.0, 1.0, 6.0])
    assert path_edge_values(g, 2, 2, "delay").shape == (0,)
    g.vp["label"] = g.new_vp("string", [f"n{i}" for i in range(8)])
    assert list(path_vertex_values(g, 0, 4, "label")) == ["n0", "n1", "n4"]


def test_section_count_partial_sections():
    g = _tree()
    # 6-4-1-0-2-5 crosses four landmark-to-landmark pieces.
    assert path_section_count(g, 6, 5, directed=False) == 4
    # 2 is transitive inside the section that ends at 5.
    assert path_section_count(g, 2, 5) == 1
    assert path_section_count(g, 4, 4) == 0
    assert path_section_count(g, 6, 0) is None


def test_ancestors_after_reroot(reduced_graph_fixture):
    from neurosetta.utils.graph_utils.graph_editing import reroot_graph
    from neurosetta.utils.graph_utils.vertex_inds import leaf_indices

    leaf = int(leaf_indices(reduced_graph_fixture)[0])
    rerooted = reroot_graph(reduced_graph_fixture, leaf)
    assert parent_index(rerooted, 0) is None
    other = 1 if rerooted.num_vertices() > 1 else 0
    if other != 0:
        assert parent_index(rerooted, other) is not None
        assert is_ancestor(rerooted, 0, other) is True
