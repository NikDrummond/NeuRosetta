# Overview

NeuRosetta is a Python library for working with **EM neuron morphologies** as
directed tree graphs: load reconstructions, measure topology and geometry, edit
trees, plot them, and (optionally) relate them to neuropil surfaces.

It targets connectomics-style workflows — FlyWire, CATMAID exports, lab SWC
pipelines — where you need reliable graph operations on large arbor data, not
just coordinate arrays.

## Core objects

| Object | What it is | Typical input |
|--------|------------|---------------|
| {class}`~neurosetta.api.Tree` | One neuron morphology | single `.swc` / `.nr` |
| {class}`~neurosetta.api.Forest` | Ordered collection of trees | directory of SWC/NR files |
| `Tree_mesh` / `Forest_mesh` | Neuron surface meshes (vedo) | `.ply`, etc. |
| `Neuropil` / `Neuropils` | Brain-region surface meshes | `.ply`, or built via {func}`~neurosetta.reconstruct_neuropil_surface` |

Everything user-facing hangs off **ID + metadata + graph** (for trees) or
**ID + metadata + mesh** (for mesh containers).

## Mental model: the graph is the source of truth

A `Tree` wraps a [graph-tool](https://graph-tool.skewed.de/)
`Graph` representing a **directed tree** (one root, edges parent → child).

```{important}
`tree.ID` and `tree.metadata` are **graph properties** (`graph.gp`), not
ordinary Python attributes. The graph object is canonical — edit metadata or
bound properties there and they persist in `.nr` files.
```

Each node (vertex) carries morphology data as **bound properties**, usually:

- `coordinates` (vertex, 3D)
- `radius` (vertex)
- `labels` (vertex, SWC label codes)

Edges carry derived geometry (`Path_length`, `Euclidean_length`, …) when you
compute them. Custom analysis outputs can be attached the same way — see
{doc}`../tutorials/tree_basics`.

**Root index is always 0** after import. Node indices align across all vertex
properties, so `tree.get_branch_indices()` subsets `tree.get_property('radius')`
directly.

## File formats: when to use what

| Format | Role | Metadata & custom properties |
|--------|------|------------------------------|
| **SWC** | Interchange with other tools | Lost on export (currently) |
| **NR** | Native NeuRosetta / graph-tool format | Fully preserved |
| **Mesh** | Surfaces for rendering & distance ops | Partial (via vedo + metadata) |

Rule of thumb:

- **SWC in, SWC out** — sharing with navis, neuromorpho, etc.
- **NR for everything else** — editing sessions, custom graph properties, units,
  reduced-tree flags.

Details: {doc}`io`.

## Reduced trees

Full SWC reconstructions contain many **transitive** nodes (degree-2 chain
points). **Reduction** collapses those while keeping root, branch, and leaf
nodes — smaller graphs, less memory, faster ops.

NeuRosetta tracks reduction in `metadata["isReduced"]`. Cable-length semantics
change after reduction: compute `Path_length` **before** reducing if you need
path length along the original arbor. See {doc}`../tutorials/tree_surgery`.

## Spatial units

Coordinates are meaningless for cross-dataset analysis without units.
NeuRosetta stores scale in `metadata["units"]` (via Pint), supports nm/µm/voxel
specs, and can rescale geometry in place.

See {doc}`../reference/units` and the units section in
{doc}`../tutorials/tree_basics`.

## Two ways to call the same thing

Almost every `tree.some_method()` is a thin bind of a function documented under
{doc}`../api/tree_ops/index`:

```python
import neurosetta as nr

tree = nr.load_example_data(720575940596125868)
assert tree.count_nodes() == nr.count_nodes(tree)
```

| Style | Use when |
|-------|----------|
| **Methods** (`tree.count_nodes()`) | Interactive work, notebooks |
| **Functions** (`nr.count_nodes(tree)`) | Pipelines, `Forest.apply`, custom wrappers |

`Forest` adds batch versions of the same ops plus `filter`, `apply`, and parallel
I/O. See {doc}`../tutorials/forests`.

## Package layout

NeuRosetta is layered (`gui` → `analysis` → `io` → `api` → `ops` → `utils` →
`core`). User code normally stops at `import neurosetta as nr`.

For the full layer diagram, `ops/` vs `utils/` split, call chains, and
contribution rules, see {doc}`../development/architecture`.

## Scripting vs GUI

| Task | Best tool |
|------|-----------|
| Batch import, metrics, figures | Python API |
| Quick inspection, reroot, subtree pick | {doc}`../tutorials/gui` (`run_neuro_GUI`) |
| Publication plots | matplotlib (`show_2d`, dendrogram) + vedo (`show_3d`, `Viewer`) |

The GUI calls the same ops as the library; prefer scripting for reproducible
pipelines.

## Typical workflow

```python
from pathlib import Path
import neurosetta as nr

# 1. Load
forest = nr.import_swc(Path("swc_dir/"), set_units="nm", progress=True)

# 2. Analyse
lengths = forest.get_total_cable_length()

# 3. Edit (optional)
forest.by_id(forest.ids()[0]).get_reduced_tree(inplace=True)

# 4. Visualise
forest.show_3d()

# 5. Save native format
forest.save_forest("out/nr/")
```

## Reading order

1. {doc}`installation` — conda env + editable install
2. {doc}`example_data` — bundled FlyWire morphologies
3. {doc}`io` — SWC vs NR
4. {doc}`../tutorials/tree_basics` — properties, counts, coordinates
5. Pick what you need: surgery, plotting, forests, meshes, surface recon, GUI
6. {doc}`../api/index` — full parameter docs
7. {doc}`../development/architecture` — layer diagram and `ops/` vs `utils/` (contributors)

## Limitations (current)

Worth knowing up front:

- **graph-tool** requires conda-forge; not pip-installable alone.
- **SWC metadata** does not round-trip on export yet.
- **Subtree extraction** can break plotting until save/reload ({doc}`../tutorials/tree_surgery`).
- **Forest 3D** is limited by vedo thread-safety and graph-tool serialisation.
- **Package distribution**: install from source for now; PyPI / conda-forge planned.
