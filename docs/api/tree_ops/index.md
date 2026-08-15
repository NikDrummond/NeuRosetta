# Tree graph operations

Functional API for single-neuron graph analysis. These functions accept a
{class}`~neurosetta.api.tree_class.Tree` (or any `_Tree` subclass) as their
first argument.

Most are also bound as methods on {class}`~neurosetta.api.tree_class.Tree` and,
where noted in each function's **See Also** section, as batch methods on
{class}`~neurosetta.api.forest_class.Forest`.

```{toctree}
:maxdepth: 1

vertex_indices
counting
coordinates
coordinate_moments
path_lengths
degrees
traversals
tree_structure
tree_geometry
shape_fitting
subtrees
tree_editing
tree_checks
transformations
tree_summary
```

## Choosing an entry point

| You have… | Prefer… |
|-----------|---------|
| One tree, interactive work | {class}`~neurosetta.api.tree_class.Tree` methods — {doc}`../tree` |
| Many trees, same operation | {class}`~neurosetta.api.forest_class.Forest` batch methods — {doc}`../forest` |
| Custom pipeline / scripting | Functions below (pass a `Tree` explicitly) |
