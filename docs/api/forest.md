# Forest

Container for multiple {class}`~neurosetta.api.tree_class.Tree` objects. Batch
methods mirror the Tree API and delegate to the same underlying ops (see
{doc}`tree_ops/index`), with optional parallel execution via
{meth}`~neurosetta.api.forest_class.Forest.apply`.

```{note}
When ``global_=True``, several coordinate and shape-fitting methods use
forest-wide implementations instead of per-tree results.
```

```{eval-rst}
.. autoclass:: neurosetta.api.forest_class.Forest
   :members:
   :inherited-members:
   :show-inheritance:
```