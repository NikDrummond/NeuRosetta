# API reference

NeuRosetta's public surface is organised as:

- **API classes** — `Tree`, `Forest`, mesh/neuropil containers
- **I/O** — SWC, native `.nr`, and mesh files
- **Plotting** — 2D/3D/dendrogram and `Viewer`
- **Analysis** — neuropil surface reconstruction
- **Functional ops** — lower-level functions bound onto `Tree` / `Forest`

```{note}
Many `Tree` methods are bound aliases of functions in
{mod}`NeuRosetta.ops.tree_graphs`. The class page lists the method names;
full parameter documentation lives with the underlying ops functions.
```

```{toctree}
:maxdepth: 2

package
api_classes
io
plotting
analysis
tree_graphs
neuropils
gui
```
