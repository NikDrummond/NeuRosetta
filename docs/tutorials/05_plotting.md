# Plotting

NeuRosetta plotting spans matplotlib (2D / dendrogram) and vedo (3D /
{class}`~neurosetta.ops.plotting.Viewer`).

## 2D projection

```python
from pathlib import Path
import neurosetta as nr

tree = nr.import_swc(Path("docs/data/1.swc"))
ax = tree.show_2d()
# ax is a matplotlib Axes — customise or save as usual
```

## Dendrogram

```python
ax = tree.show_dendrogram()
```

```{note}
Dendrogram aspect/scaling is still being refined. Treat layouts as
exploratory for now.
```

## Single-tree 3D

```python
tree.show_3d()
```

This builds a {class}`~neurosetta.ops.plotting.Viewer`, shows the scene, and
closes the viewer when finished.

## Custom Viewer

For more control (multiple neurons, screenshots, reuse):

```python
from neurosetta import Viewer

tree_a = nr.import_swc(Path("docs/data/1.swc"))
tree_b = nr.import_swc(Path("docs/data/2.swc"))

with Viewer(interactive=False) as viewer:
    viewer.add_neuron(tree_a)
    viewer.add_neuron(tree_b)
    viewer.screenshot("two_neurons.png")
```

Set `interactive=True` (default in many contexts) for an on-screen vedo
window.

## Forest 3D

```python
forest = nr.import_swc(Path("docs/data"))
forest.show_3d()
```

Or build a viewer yourself:

```python
viewer = Viewer()
viewer.add_forest(forest)
viewer.show()
```

```{note}
vedo is not thread-safe and graph-tool is not multiprocessing-safe, so forest
3D construction parallelises only the thread-safe coordinate preparation
where possible. Large forests may still take time to build.
```

## Subtree highlighting

After creating subtree masks (`subtree_mask_from_root`), you can build a
coloured 3D subtree representation with
{func}`~neurosetta.ops.plotting.build_3d_subtree`.

Next: {doc}`06_forests`.
