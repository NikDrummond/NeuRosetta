# Subtrees

Subtree masks, scores, extraction, partition asymmetry, per-node subtree
measurements, and Strahler order.

The subtree of a node is that node plus every descendant. The edge that
connects the node to its parent is **outside** the subtree. Consequences:

* a leaf has 1 vertex, 1 leaf, 0 edges, 0 cable, depth 0, and height 0
* on a valid tree, `vertex_count = edge_count + 1` at every node
* the root's subtree is the whole tree

## Upstream depth vs downstream depth

{meth}`~neurosetta.Tree.get_node_depth` counts edges from the root **down to**
the node. {meth}`~neurosetta.Tree.get_subtree_depth` counts edges from the node
**down to** the farthest descendant leaf. At a leaf those numbers are different:
node depth can be large, subtree depth is 0.

{meth}`~neurosetta.Tree.get_subtree_branch_depth` counts branching nodes
(out-degree ≥ 2) strictly downstream of the start. The starting node is not
included, so a branch whose children are all leaves has branch depth 0.

## Strahler order vs branch order from the root

Strahler order is a label of the hierarchy, read from the leaves upward:

* leaves are 1
* one child: inherit that child's order
* several children: if at least two share the maximum child order `m`, the
  node is `m + 1`; otherwise it is `m`

It does not use coordinates or edge lengths. A chain of transitive nodes
inherits the downstream order, so inserting those nodes does not change the
order at the original branch and leaf nodes. It is not the number of branches
between the root and a node, and it is not subtree branch depth.

## Path length vs Euclidean length

`Path_length` is physical cable, including lengths that were preserved when
transitive nodes were collapsed. `Euclidean_length` is the straight-line
length of the edge as stored. Subtree cable, height, and mean leaf distance
default to `Path_length` and do not silently switch to Euclidean chords.
A reduced tree that has lost `Path_length` raises. An unreduced tree without
`Path_length` uses the straight-line length of each uncollapsed segment.

The historical subgraph score still prefers `Path_length` and falls back to
`Euclidean_length`. Partition asymmetry's cable modes require `Path_length`.

```{eval-rst}
.. automodule:: neurosetta.ops.tree_graphs.tree_subtree_selection
   :members:
.. automodule:: neurosetta.ops.tree_graphs.tree_subtree_scores
   :members:
.. automodule:: neurosetta.ops.tree_graphs.tree_partition_asymmetry
   :members:
.. automodule:: neurosetta.ops.tree_graphs.tree_subtree_counts
   :members:
.. automodule:: neurosetta.ops.tree_graphs.tree_subtree_lengths
   :members:
.. automodule:: neurosetta.ops.tree_graphs.tree_subtree_depth
   :members:
.. automodule:: neurosetta.ops.tree_graphs.tree_subtree_hierarchy
   :members:
```

```python
import neurosetta as nr

tree = nr.load_example_data(nr.example_ids[0])
orders = tree.get_strahler_order()
print(tree.get_max_strahler_order(), orders[tree.get_root_index()])
print(tree.get_subtree_path_length()[tree.get_root_index()])
```
