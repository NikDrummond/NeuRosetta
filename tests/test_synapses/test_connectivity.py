"""Forest connectivity table / graph tests."""

from __future__ import annotations

import pandas as pd
from graph_tool.all import Graph

from neurosetta.api import Forest, Tree


def _blank_tree(tree_id: int) -> Tree:
    g = Graph(directed=True)
    g.add_vertex()
    g.vp["x"] = g.new_vp("double", [0.0])
    g.vp["y"] = g.new_vp("double", [0.0])
    g.vp["z"] = g.new_vp("double", [0.0])
    g.vp["radius"] = g.new_vp("double", [1.0])
    g.vp["node_type"] = g.new_vp("int", [-1])
    g.gp["ID"] = g.new_gp("long", int(tree_id))
    g.gp["metadata"] = g.new_gp("object", {"isReduced": False, "Flag": False, "Neuron_type": "T"})
    return Tree.from_graph(g)


def _syn(sid, typ, partner, cleft, x=0, y=0, z=0):
    return {
        "synapse_id": sid,
        "type": typ,
        "x": x,
        "y": y,
        "z": z,
        "partner_id": partner,
        "cleft_id": cleft,
    }


def _attach(tree: Tree, rows: list[dict]) -> None:
    tree.set_synapses(pd.DataFrame(rows))


def test_connectivity_direction_and_weights():
    a, b, c = _blank_tree(1), _blank_tree(2), _blank_tree(3)
    # A -> B : 3 pre on A
    _attach(
        a,
        [_syn(i, "pre", 2, f"ab{i}") for i in range(3)]
        + [_syn(10 + i, "pre", 3, f"ac{i}") for i in range(2)],
    )
    # B -> A : 1 pre on B
    _attach(b, [_syn(100, "pre", 1, "ba0")])
    _attach(c, [_syn(200 + i, "pre", 2, f"cb{i}") for i in range(4)])
    forest = Forest([a, b, c])
    table = forest.get_connectivity_table(deduplicate="auto")
    counts = {(r.source_id, r.target_id): r.synapse_count for r in table.itertuples()}
    assert counts[(1, 2)] == 3
    assert counts[(1, 3)] == 2
    assert counts[(2, 1)] == 1
    assert counts[(3, 2)] == 4

    g = forest.get_connectivity_graph(deduplicate="auto")
    assert g.is_directed()
    assert g.num_vertices() == 3
    assert g.num_edges() == 4
    assert set(g.vp["tree_id"]) == {1, 2, 3}
    assert all(g.vp["in_forest"])

    assert forest.get_out_degree()[1] == 2
    assert forest.get_in_degree()[2] == 2
    assert forest.get_out_strength()[1] == 5
    assert forest.get_in_strength()[2] == 7


def test_deduplicate_shared_cleft():
    a, b = _blank_tree(1), _blank_tree(2)
    # Same physical synapse recorded on both neurons.
    _attach(a, [_syn(1, "pre", 2, "shared")])
    _attach(b, [_syn(99, "post", 1, "shared")])
    forest = Forest([a, b])
    table_auto = forest.get_connectivity_table(deduplicate="auto")
    assert len(table_auto) == 1
    assert int(table_auto.iloc[0]["synapse_count"]) == 1

    table_none = forest.get_connectivity_table(deduplicate="none")
    assert int(table_none.iloc[0]["synapse_count"]) == 2


def test_include_external_and_subset():
    a, b = _blank_tree(1), _blank_tree(2)
    _attach(a, [_syn(1, "pre", 2, "a"), _syn(2, "pre", 999, "b")])
    forest = Forest([a, b])
    g_int = forest.get_connectivity_graph(include_external=False)
    assert g_int.num_vertices() == 2
    g_ext = forest.get_connectivity_graph(include_external=True)
    assert g_ext.num_vertices() == 3
    assert sum(1 for v in g_ext.vertices() if not g_ext.vp["in_forest"][v]) == 1

    subset = Forest([a])
    g_sub = subset.get_connectivity_graph(include_external=False)
    assert g_sub.num_edges() == 0  # partner 2 not in subset
    g_sub_ext = subset.get_connectivity_graph(include_external=True)
    assert g_sub_ext.num_edges() == 2


def test_connectivity_invariant_to_reduction():
    from neurosetta.testing import make_synthetic_tree

    trees = []
    for i, partner in enumerate([2, 1], start=1):
        t = make_synthetic_tree(20, tree_id=i, seed=i)
        coords = t.get_node_coordinates()
        mid = coords[1]
        t.set_synapses(
            pd.DataFrame(
                {
                    "synapse_id": [0, 1],
                    "type": ["pre", "post"],
                    "x": [mid[0], mid[0]],
                    "y": [mid[1], mid[1]],
                    "z": [mid[2], mid[2]],
                    "partner_id": [partner, partner],
                    "cleft_id": [f"c{i}a", f"c{i}b"],
                }
            )
        )
        t.map_synapses()
        trees.append(t)
    forest = Forest(trees)
    before = forest.get_connectivity_table(deduplicate="auto")
    for t in forest:
        t.get_reduced_tree(inplace=True)
    after = forest.get_connectivity_table(deduplicate="auto")
    pd.testing.assert_frame_equal(
        before.sort_values(["source_id", "target_id"]).reset_index(drop=True),
        after.sort_values(["source_id", "target_id"]).reset_index(drop=True),
    )


def test_post_on_a_means_b_to_a():
    a = _blank_tree(1)
    _attach(a, [_syn(1, "post", 5, "x")])
    forest = Forest([a])
    table = forest.get_connectivity_table(include_external=True)
    assert list(table.itertuples(index=False))[0][:2] == (5, 1)
