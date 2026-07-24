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
points, leaves, and the root:

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
