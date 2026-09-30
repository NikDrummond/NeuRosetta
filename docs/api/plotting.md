# Plotting

Every tree carries a {class}`~neurosetta.ops.plotting.utils.TreePlot3D` as
``tree.plot3d``. It exists from the moment the tree does, but holds no vedo actors
until they are needed, so style can be set up front:

```python
tree.plot3d.colour = "red"
tree.plot3d.lw = 2
tree.show_3d()                           # uses the stored style
tree.plot3d.colour_by("Path_length")     # or colour edges by a graph property
tree.plot3d.rebuild()                    # after editing the tree, style preserved
```

Meshes use the same `show_3d` name via {func}`~neurosetta.ops.plotting.plot_mesh.plot_mesh`.
Attached neuron meshes overlay on morphology with ``show_mesh=True``:

```python
tree.show_3d(show_mesh=True, mesh_kwargs={"alpha": 0.35})

neuropil.show_3d(c="lightblue", alpha=0.4)
viewer = Viewer()
viewer.add_neuron(tree, show_mesh=True)
viewer.add_mesh(neuropil, alpha=0.3)
viewer.show()
```

```{eval-rst}
.. autoclass:: neurosetta.ops.plotting.utils.TreePlot3D
   :members:

.. autoclass:: neurosetta.ops.plotting.style.Plot3DStyle
   :members:

.. automodule:: neurosetta.ops.plotting.plot_2d
   :members:

.. automodule:: neurosetta.ops.plotting.plot_3d
   :members:

.. automodule:: neurosetta.ops.plotting.plot_mesh
   :members:

.. automodule:: neurosetta.ops.plotting.plot_dendrogram
   :members:

.. automodule:: neurosetta.ops.plotting.plot_subtree
   :members:

.. autoclass:: neurosetta.ops.plotting.viewer.Viewer
   :members:
```
