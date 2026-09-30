# Mesh, neuropil & anatomical frame

```{important}
**Preferred neuron-surface workflow:** attach with ``tree.set_mesh`` /
``forest.set_meshes`` and work from :class:`~neurosetta.api.Tree` /
:class:`~neurosetta.api.Forest` (see {doc}`../tutorials/meshes`).
:class:`~neurosetta.api.Tree_mesh` / :class:`~neurosetta.api.Forest_mesh`
remain the I/O and facet payload types — not a parallel mesh-only neuron API.
```

Three mesh-related roles — do not mix them:

| Type | Role | ``ID`` | ``mesh_kind`` |
|------|------|--------|---------------|
| {class}`~neurosetta.api.Tree_mesh` | Neuron surface payload (``tree.mesh``) | Match owning {class}`~neurosetta.api.Tree` ``ID`` | ``neuron`` |
| {class}`~neurosetta.api.Forest_mesh` | Batch import of ``Tree_mesh`` (attach via ``set_meshes``) | — | — |
| {class}`~neurosetta.api.Neuropil` | Compartment / region boundary | Region label (e.g. ``\"AL\"``) | ``neuropil`` |
| {class}`~neurosetta.api.Neuropils` | Collection of ``Neuropil`` | — | — |
| {class}`~neurosetta.api.AnatomicalFrame` | Analysis context (meshes + axes) — **not** a mesh | Optional ``name`` | — |

{class}`~neurosetta.api.AnatomicalFrame` accepts ``Neuropil`` (or generic
``_Mesh``) as ``reference_mesh`` / surfaces. Passing a ``Tree_mesh`` raises
``TypeError``.

Helpers: {func}`~neurosetta.core.mesh.mesh_kind_of`,
{func}`~neurosetta.core.mesh.check_neuron_mesh_owner_id`.

Attach a neuron mesh (same pattern as ``tree.synapses``)::

    tree.set_mesh("7.ply", set_units="um")
    # or tree.set_mesh(Tree_mesh(...)) / tree.mesh = ...
    tree.show_3d(show_mesh=True)   # skeleton + mesh overlay

Batch-attach to a forest by ID::

    forest.set_meshes("meshes/", set_units="um")
    forest.show_3d(show_mesh=True)

Tutorial: {doc}`../tutorials/meshes` · ops: {doc}`tree_ops/mesh`.

```{eval-rst}
.. autoclass:: neurosetta.api.tree_mesh_class.Tree_mesh
   :members:
   :inherited-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: neurosetta.api.forest_mesh_class.Forest_mesh
   :members:
   :inherited-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: neurosetta.api.neuropil_class.Neuropil
   :members:
   :inherited-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: neurosetta.api.neuropils_class.Neuropils
   :members:
   :inherited-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: neurosetta.api.anatomical_frame.AnatomicalFrame
   :members:
   :show-inheritance:
```
