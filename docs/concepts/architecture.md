# Architecture

NeuRosetta is organised in strict layers. **User code should import from the
top** (`neurosetta`, `neurosetta.io`, `neurosetta.gui`) and avoid reaching into
`core/` or `utils/` unless you are extending the library.

## Layer diagram

```
┌─────────────────────────────────────────────────────────┐
│  gui/           layer 6 — PySide6 + vedo viewer         │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│  analysis/      layer 5 — high-level analysis entry     │
│                 (e.g. reconstruct_neuropil_surface)      │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│  io/            layer 4 — import_swc, save, import_mesh  │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│  api/           layer 3 — Tree, Forest, mesh containers │
│                 methods delegate to ops/                 │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐  - - - - - - - - -
│  ops/           layer 2 — functions on Tree / Forest    │    utils/ (parallel)
└───────────────────────────┬─────────────────────────────┘  graph, geometry,
                            │                                units, vedo helpers
┌───────────────────────────▼─────────────────────────────┐  - - - - - - - - -
│  core/          layer 1 — _Stone, _Tree, _Forest, mesh  │
│                 graph-tool Graph wrappers               │
└─────────────────────────────────────────────────────────┘
```

Data flows **down** on every operation: an `api.Tree` method calls an `ops/`
function, which pulls what it needs from `utils/` and mutates or reads the
`core._Tree.graph` object.

## Layer 1: `core/`

Internal graph containers. Not part of the public import surface, but important
to understand.

| Module | Role |
|--------|------|
| `core/stone.py` | `_Stone` — shared `ID` + `metadata` base (slotted) |
| `core/tree.py` | `_Tree` — single directed tree; `ID` / `metadata` are **graph properties** on `graph.gp` |
| `core/forest.py` | `_Forest` — ordered list of `_Tree` objects + batch helpers |
| `core/mesh.py` | Base for vedo mesh wrappers |

`_Tree` is the object every `ops/` function actually receives (the public
{class}`~neurosetta.api.Tree` subclasses it and adds bound methods).

## Layer 2: `ops/` vs `utils/`

This is the main split to internalise.

### `ops/` — operations on NeuRosetta objects

Functions take a `Tree`, `Forest`, or mesh container as their first argument.
They know about metadata, units, and NeuRosetta conventions (root at 0,
`isReduced`, etc.).

| Subpackage | Purpose |
|------------|---------|
| `ops/tree_graphs/` | Counting, indices, coordinates, traversals, editing, subtrees, path lengths, shape fitting |
| `ops/forest_ops/` | **Forest-wide** geometry — align/rotate/scale all trees together, forest PCA, convex hull |
| `ops/plotting/` | `show_2d`, `show_3d`, dendrogram, {class}`~neurosetta.ops.plotting.Viewer` |
| `ops/neuropils/` | Point-to-surface distances on neuropil meshes |
| `ops/units/` | `set_units`, `convert_units`, forest harmonisation — updates tree metadata **and** coordinates |

Most `Tree` methods are binds of `ops/tree_graphs` functions. A smaller set
(forest alignment, etc.) comes from `ops/forest_ops`.

### `utils/` — stateless helpers on graphs and arrays

Functions work on raw `graph_tool.Graph` objects, numpy arrays, or Pint units.
They have **no** knowledge of the `Tree` class.

| Subpackage | Purpose |
|------------|---------|
| `utils/graph_utils/` | Property bind/get/set, traversals, vertex indices, counting, subgraph editing |
| `utils/geometry_utils/` | Linear algebra, rotations, PCA, projections, tolerances |
| `utils/units/` | Pint registry, alias normalisation, voxel specs, coordinate rescaling |
| `utils/vedo_utils/` | Mesh distance / shape helpers used by plotting and neuropil ops |

**Example call chain** for `tree.count_nodes()`:

```
Tree.count_nodes()
  → ops.tree_graphs.count_nodes(tree)
    → utils.graph_utils.count_vertices(tree.graph)
```

`ops/` adds the `_Tree` type hint, docstring, and any NeuRosetta-specific
pre/post processing. `utils/` does the graph-tool work.

```{note}
When adding a new tree metric: implement the graph logic in `utils/graph_utils/`
(or `utils/geometry_utils/` if purely numeric), wrap it in `ops/tree_graphs/`,
then bind it onto `api/tree_class.Tree` and optionally re-export from
`neurosetta.__init__`.
```

### `ops/forest_ops/` vs `Forest.apply`

Easy to confuse — they solve different problems:

| Mechanism | What it does |
|-----------|-------------|
| `Forest.apply(fn)` | Run a **per-tree** function on every member (optionally parallel). `forest.count_nodes()` is implemented this way. |
| `ops/forest_ops/*` | Operate on the **entire forest as one geometric object** — e.g. rotate all trees around a shared axis, PCA of pooled coordinates. |

## Layer 3: `api/`

Thin user-facing classes:

- {class}`~neurosetta.api.Tree` — binds ~all tree ops + plotting + units ({doc}`../api/tree`, {doc}`../api/tree_ops/index`)
- {class}`~neurosetta.api.Forest` — per-tree batch binds + `filter`, `apply`, parallel I/O ({doc}`../api/forest`)
- Mesh classes — `Tree_mesh`, `Forest_mesh`, `Neuropil`, `Neuropils`

The binding machinery in `forest_class.py` introspects function signatures so
batch methods inherit `parallel`, `max_workers`, `progress`, and `bind` kwargs
from the underlying op.

Import as:

```python
import neurosetta as nr
tree = nr.Tree(...)   # or nr.import_swc(...)
```

## Layer 4: `io/`

File ↔ object conversion. Returns `api/` instances, never raw graphs.

| Module | Exports |
|--------|---------|
| `io/swc_utils.py` | `import_swc`, `export_swc` |
| `io/nr_utils.py` | `save`, `load` (graph-tool `.gt` wrapper) |
| `io/mesh_utils.py` | `import_mesh`, `export_mesh` |

Single file → `Tree` / mesh object. Directory → `Forest` / mesh collection.

## Layer 5: `analysis/`

High-level routines composed from `ops/` + `utils/`. Kept separate so the core
tree API stays general-purpose.

Example: {func}`~neurosetta.reconstruct_neuropil_surface` pools forest
coordinates, voxelises, runs marching cubes, returns a `Neuropil`.

## Layer 6: `gui/`

Desktop app (`run_neuro_GUI` / `nr.start_GUI()`). Subpackages:

| Subpackage | Role |
|------------|------|
| `gui/core/` | Application lifecycle |
| `gui/file_io/` | Load/save dialogs wrapping `io/` |
| `gui/rendering/` | vedo/VTK scene + point picking |
| `gui/tools/` | Reroot, subtree extraction — calls same ops as the library |
| `gui/ui/` | Qt widgets, scale bar overlay |

The GUI never bypasses `ops/`; it is a front-end over the same stack.

## Public import surface

What `import neurosetta as nr` re-exports (see {doc}`../api/package`):

| Category | Examples |
|----------|----------|
| Classes | `Tree`, `Forest`, mesh containers |
| I/O | `import_swc`, `save`, `import_mesh`, … |
| Tree ops (functional) | `count_nodes`, `reduce_tree`, `get_node_coordinates`, … |
| Plotting | `Viewer`, `plot_2d`, `plot_3d` |
| Units | `set_units`, `convert_units`, `harmonize_forest_units` |
| Analysis | `reconstruct_neuropil_surface` |
| GUI | `start_GUI` |

Functional tree ops are re-exported for scripting convenience; methods on
`Tree` / `Forest` are equivalent.

## Dependency direction (rules)

```
gui, analysis  →  io, api  →  ops  →  utils  →  core  →  graph-tool
```

- **`utils/` must not import from `ops/` or `api/`**
- **`ops/` may import `core/` and `utils/`**
- **`api/` may import `core/` and `ops/`**
- **`io/` builds `api/` objects via `core/` constructors**

These rules keep the graph layer testable without pulling in plotting or Qt.

## Related reading

- {doc}`overview` — objects, file formats, workflows
- {doc}`../tutorials/03_tree_basics` — graph properties in practice
- {doc}`../api/index` — generated API reference
