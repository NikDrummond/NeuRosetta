# Forest

Container for multiple {class}`~neurosetta.api.tree_class.Tree` objects. Batch
methods mirror the Tree API and delegate to the same underlying ops (see
{doc}`tree_ops/index`), with optional parallel execution via
{meth}`~neurosetta.api.forest_class.Forest.apply`.

Container helpers inherited from {class}`~neurosetta.core.forest._Forest` are
included below (``apply``, ``filter``, ``copy``, ``append``, metadata batch
helpers, etc.). Batch op methods accept the same ``apply`` execution kwargs
documented on each method: ``parallel``, ``max_workers``, ``show_progress``,
``bind``, and ``merge_axis`` (when ``global_=False``).

```{note}
When ``global_=True``, several coordinate and shape-fitting methods use
forest-wide implementations instead of per-tree results.
```

```{eval-rst}
.. autoclass:: neurosetta.api.forest_class.Forest
   :members:
   :inherited-members: _Forest
   :special-members: __contains__
   :exclude-members: count, index
   :show-inheritance:
```