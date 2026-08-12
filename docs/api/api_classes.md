# API classes

## Tree

Main user-facing class for a single neuron morphology. Most methods are aliases
of functions in :mod:`neurosetta.ops.tree_graphs` and :mod:`neurosetta.ops.plotting`.

```{eval-rst}
.. autoclass:: neurosetta.api.tree_class.Tree
   :members:
   :inherited-members:
   :show-inheritance:
```

## Forest

Container for multiple :class:`~neurosetta.api.tree_class.Tree` objects with
batch operations via :meth:`~neurosetta.api.forest_class.Forest.apply`.

```{eval-rst}
.. autoclass:: neurosetta.api.forest_class.Forest
   :members:
   :inherited-members:
   :show-inheritance:
```

## Tree_mesh

```{eval-rst}
.. autoclass:: neurosetta.api.tree_mesh_class.Tree_mesh
   :members:
   :inherited-members:
   :show-inheritance:
```

## Forest_mesh

```{eval-rst}
.. autoclass:: neurosetta.api.forest_mesh_class.Forest_mesh
   :members:
   :inherited-members:
   :show-inheritance:
```

## Neuropil

```{eval-rst}
.. autoclass:: neurosetta.api.neuropil_class.Neuropil
   :members:
   :inherited-members:
   :show-inheritance:
```

## Neuropils

```{eval-rst}
.. autoclass:: neurosetta.api.neuropils_class.Neuropils
   :members:
   :inherited-members:
   :show-inheritance:
```
