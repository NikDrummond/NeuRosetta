# Path analysis

Paths on a NeuRosetta tree follow edges that point from parent to child.
``directed=True`` is the default, so a path exists only from a node to one
of its descendants. ``directed=False`` walks the unique undirected path
through the lowest common ancestor and does not rewrite the graph.

Three lengths are easy to mix up:

| Quantity | What it is |
| --- | --- |
| ``Path_length`` | Physical cable on an edge, including a collapsed section on a reduced tree |
| ``Euclidean_length`` | Straight line between the edge endpoints |
| Endpoint distance | Straight line between two node coordinates, ignoring the route |

``ep=None`` on a path length selects ``Path_length``, then ``Euclidean_length``.
An explicit name is used as given. A missing explicit property is an error.
Low-level path utilities do not invent lengths from coordinates. Reduced
trees must keep the ``Path_length`` that was stored for the original section.
Radial or chord geometry of a reduced edge is not that cable.

``get_root_distance`` is unchanged: its default is ``Path_length`` with no
fallback, and binding still writes ``Root_distance`` or
``Root_euclidean_distance``. Distances from an arbitrary node use
``get_source_distances`` and a separate property name when bound.

A lowest common ancestor is the deepest node, in hop count from the root,
that is ancestral to every requested node. It accepts one vertex, several
positional indices, or any nonempty list, tuple, or array. Edge lengths are
not used.

Tortuosity is path cable divided by endpoint displacement. The same node,
or two distinct nodes at the same coordinate, returns ``nan``. A missing
directed path returns ``None``. Batch distances use ``inf`` for unreachable
pairs.

Vertex indices passed to a Forest method are interpreted inside each tree.
They are not split into one index per tree.

```{eval-rst}
.. automodule:: neurosetta.ops.tree_graphs.tree_paths
   :members:
.. automodule:: neurosetta.ops.tree_graphs.tree_path_ancestry
   :members:
.. automodule:: neurosetta.ops.tree_graphs.tree_path_geometry
   :members:
.. automodule:: neurosetta.ops.tree_graphs.tree_path_profiles
   :members:
.. automodule:: neurosetta.ops.tree_graphs.tree_path_morphology
   :members:
```
