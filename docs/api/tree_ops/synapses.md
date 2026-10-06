# Synapses

Attach, map, count, and analyse synapses on a single tree. Also bound as
methods on {class}`~neurosetta.api.tree_class.Tree` (and batch methods on
{class}`~neurosetta.api.forest_class.Forest` where noted).

Forest batch attach by ID: {func}`~neurosetta.ops.forest_ops.forest_synapses.set_synapses`
(bound as ``Forest.set_synapses``). Typical pipeline::

    syns = extract_synapses(connectivity_df, forest.ids())
    forest.set_synapses(syns)

```{seealso}
Container type: {class}`~neurosetta.core.synapses.Synapses` — {doc}`../synapses`
· I/O: {func}`~neurosetta.io.synapse_io.extract_synapses`
```

```{eval-rst}
.. automodule:: neurosetta.ops.tree_graphs.tree_synapses
   :members:
```

## Forest batch attach

```{eval-rst}
.. automodule:: neurosetta.ops.forest_ops.forest_synapses
   :members:
```
