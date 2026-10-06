# Synapses

Synapses are **observations attached to a morphology**, not morphology vertices.
The table type is {class}`~neurosetta.core.synapses.Synapses`; tree-level ops live
in {doc}`tree_ops/synapses`, and forest-level connectivity in the section below.

Connectivity-table → ``Synapses`` conversion (FlyWire / navis-style pre/post
frames) lives in I/O: {func}`~neurosetta.io.synapse_io.extract_synapses`.
Already per-neuron NeuRosetta-schema tables use
{func}`~neurosetta.io.synapse_io.import_synapses` — see {doc}`io`.

Batch-attach to a Forest by ID with
{func}`~neurosetta.ops.forest_ops.forest_synapses.set_synapses`
(``forest.set_synapses(...)``) — see {doc}`tree_ops/synapses`.

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
