# Tree

Main user-facing class for a single neuron morphology.

Most methods are bound aliases of functions in {doc}`tree_ops/index`. Use the
sections below as a method index; full parameter documentation lives with the
underlying op.

```{seealso}
Method groups map to the tree-ops reference:

| Group | Ops page |
|-------|----------|
| Node indices | {doc}`tree_ops/vertex_indices` |
| Counting | {doc}`tree_ops/counting` |
| Coordinates | {doc}`tree_ops/coordinates` |
| Projection moments | {doc}`tree_ops/coordinate_moments` |
| Cable length | {doc}`tree_ops/path_lengths` |
| Degrees | {doc}`tree_ops/degrees` |
| Traversals | {doc}`tree_ops/traversals` |
| Angles & bifurcations | {doc}`tree_ops/tree_geometry` |
| Shape fitting | {doc}`tree_ops/shape_fitting` |
| Subtrees | {doc}`tree_ops/subtrees` |
| Editing | {doc}`tree_ops/tree_editing` |
| Checks | {doc}`tree_ops/tree_checks` |
| Transforms | {doc}`tree_ops/transformations` |
| I/O | {doc}`io` |
| Plotting | {doc}`plotting` |
```

```{eval-rst}
.. autoclass:: neurosetta.api.tree_class.Tree
   :members:
   :inherited-members:
   :show-inheritance:
```