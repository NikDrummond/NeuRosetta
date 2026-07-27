# Tree surgery and subtrees

NeuRosetta supports reducing morphologies, rerooting, and extracting
subtrees — operations also exposed interactively in the GUI.

```{important}
Several editing helpers take ``inplace`` / ``bind`` flags.

- ``get_reduced_tree(inplace=False)`` and ``get_rerooted_tree(..., inplace=False)``
  return a **graph-tool Graph**, not a ``Tree``.
- ``get_subtree()`` edits the tree **in place** and returns ``None``.
- Scoring helpers default to ``bind=True`` (attach a graph property, return
  ``None``). Pass ``bind=False`` to get arrays back.
```

## Reduce a tree

Reduction collapses transitive (chain) vertices while preserving branch
points, leaves, and the root. This removes any nodes in the neuron with out degree = in degree = 1, reducing your graph representation to only the core elements of the neurons morphoplogy - the root, branch nodes and terminal nodes. 

```python
from pathlib import Path
import NeuRosetta as nr

tree = nr.import_swc(Path("docs/data/1.swc"))
print("before:", tree.count_nodes(), "reduced?", tree.is_reduced())

# In-place reduction
tree.get_reduced_tree(inplace=True)
print("after:", tree.count_nodes(), "reduced?", tree.is_reduced())
```

To keep the original tree, reduce a copy by rebuilding from the returned graph:

```python
tree = nr.import_swc(Path("docs/data/1.swc"))
g = tree.get_reduced_tree(inplace=False)
reduced = nr.Tree(ID=tree.ID, metadata=dict(tree.metadata), graph=g)
print(reduced.count_nodes())
```

This damatically reduces the memory footprint of single neurons, at the expence of the curvature in the neuronal sections. NeuRosetta keeps track of if a neuron is reduces of not within its metadata under the `isReduced` key. You can check if a neuron has been reduced:

```python
tree.is_reduced()
```
Which returns True if the neuron has no transitive nodes. NeuRosetta will assume that a neuron has not been reduced on import, unless the metadata says otherwise. If you have a neuron which has been reduced and, for whetever reason, the metadata says it has not, you can use:

```python
tree.update_reduced()
```
Which will update the `isReduced` key in the neurons metadata if, and only if, the neuron does infact have no transitive nodes. 

## Reroot

Pick a new root using a **graph-tool vertex index** (not the SWC node id):

```python
tree = nr.import_swc(Path("docs/data/1.swc"))
leaves = tree.get_leaf_indices()
new_root = int(leaves[0])
tree.get_rerooted_tree(new_root, inplace=True)
print(tree.get_root_index())
```

## Subtree masks, scores, and extraction

```python
tree = nr.import_swc(Path("docs/data/1.swc"))

# Score all vertices as potential subtree roots (return array)
scores = tree.get_subtree_scores(bind=False)
best = tree.get_max_subtree_node()
print("best subtree root index:", best, "score:", scores[best])

# Mask from a chosen branch point, then extract in place
branches = tree.get_branch_indices()
if len(branches):
    v = int(branches[0])
    tree.subtree_mask_from_root(v)   # bind=True by default
    tree.get_subtree()        # edits tree in place
    print("subtree nodes:", tree.count_nodes())
```

Partition asymmetry at branch points:

```python
tree = nr.import_swc(Path("docs/data/1.swc"))
asym = tree.get_partition_asymmetry(bind=False)
print(asym)
```

```{warning}
Extracting a subtree can currently break subsequent plotting until the tree
is saved and reloaded. If ``show_2d`` / ``show_3d`` misbehave after
``get_subtree``, round-trip via ``.nr`` or SWC as a workaround.
```

## Typical workflow

```python
tree = nr.import_swc(Path("docs/data/1.swc"))
tree.get_reduced_tree(inplace=True)
# inspect with tree.show_3d() or the GUI, then reroot / extract as needed
```

Next: {doc}`05_plotting`.
