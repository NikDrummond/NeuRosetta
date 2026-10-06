# API reference

NeuRosetta's public surface is organised as:

- **Tree** — single-neuron container and method index
- **Forest** — multi-tree batch operations
- **Workspace** — persistent analysis session (``.nrw``)
- **Tree ops** — lower-level functions grouped by operation type
- **Synapses** — synapse table, tree mapping ops, forest connectivity
- **I/O** — SWC, native ``.nr``, Workspace ``.nrw``, mesh, and synapse tables
- **Plotting** — 2D/3D/dendrogram and `Viewer`
- **Analysis** — neuropil surface reconstruction
- **Mesh & neuropil classes** — mesh containers

```{note}
Most {class}`~neurosetta.api.tree_class.Tree` methods are bound aliases of
functions in {doc}`tree_ops/index`. The Tree page lists method names; parameter
documentation lives with the underlying ops.
```

```{toctree}
:maxdepth: 2

package
tree
forest
workspace
tree_ops/index
synapses
io
plotting
analysis
api_classes
neuropils
gui
```
