# Synapses

Synapses are **observations attached to a morphology**, not morphology vertices.
The table type is {class}`~neurosetta.core.synapses.Synapses`; tree-level ops live
in {doc}`tree_ops/synapses`, and forest-level connectivity in the section below.

```{seealso}
Tutorial: {doc}`../tutorials/synapses`
```

## Container

```{eval-rst}
.. autoclass:: neurosetta.core.synapses.Synapses
   :members:
   :show-inheritance:
```

## Helpers

```{eval-rst}
.. autofunction:: neurosetta.core.synapses.synapses_from_arrays
.. autofunction:: neurosetta.core.synapses.canonicalize_synapse_type
.. autofunction:: neurosetta.core.synapses.resolve_synapse_type_filter
```

## Forest connectivity

Directed network graph among trees (and optional external partners), derived
from synapse tables — never merged into the morphology graph.

```{eval-rst}
.. automodule:: neurosetta.ops.forest_ops.forest_connectivity
   :members:
```
