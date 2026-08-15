# Overview

NeuRosetta is a Python library for working with **EM neuron morphologies** as
directed tree graphs: load reconstructions, measure topology and geometry, edit
trees, plot them, and (optionally) relate them to neuropil surfaces.

It targets connectomics-style workflows — FlyWire, CATMAID exports, lab SWC
pipelines — where you need reliable graph operations on large arbor data, not
just coordinate arrays.

We are still in the early days of NeuRosetta, so many more features are in the works. This includes, for release soon, integration of synaptic data and topological analyses of neuron morrphology, for example, as well as fleshing out mesh handling, which is curently short of a few features we have in development.

## Core objects

NeuRosetta is built around a few core objects which handle the majority of things you aer likely to need:

| Object                          | What it is                   | Typical input                                                         |
| ------------------------------- | ---------------------------- | --------------------------------------------------------------------- |
| {class}`~neurosetta.api.Tree`   | One neuron morphology        | single `.swc` / `.nr`                                                 |
| {class}`~neurosetta.api.Forest` | Ordered collection of trees  | directory of SWC/NR files                                             |
| `Tree_mesh` / `Forest_mesh`     | Neuron surface meshes (vedo) | `.ply`, etc.                                                          |
| `Neuropil` / `Neuropils`        | Brain-region surface meshes  | `.ply`, or built via {func}`~neurosetta.reconstruct_neuropil_surface` |

Single objects (single Trees or meshes) have the same 'core' properties: **ID + metadata + graph** for a `Tree`, or **ID + metadata + mesh** for mesh objects (`Tree_mesh`,`Neuropil`). Collections (`Forest`,`Forerst_mesh`,`Neuropils`) are also derived from the same containers with similar `apply` and `filter` functionality, see {doc}`../tutorials/forests` 

## Two ways to call the same thing

Almost every `tree.some_method()` is a thin bind of a function documented under
{doc}`../api/tree_ops/index`:

```python
import neurosetta as nr

tree = nr.load_example_data(720575940596125868)
assert tree.count_nodes() == nr.count_nodes(tree)
```

| Style                                  | Use when                                   |
| -------------------------------------- | ------------------------------------------ |
| **Methods** (`tree.count_nodes()`)     | Interactive work, notebooks                |
| **Functions** (`nr.count_nodes(tree)`) | Pipelines, `Forest.apply`, custom wrappers |

`Forest` adds batch versions of the same ops plus `filter`, `apply`, and parallel
I/O. See {doc}`../tutorials/forests`.

Regardless of the above rercomendation, it is really up to the use preferrence if you want to call things as methods on your object, or using the function based implementation. We aim for both use cases to wok in parrrallel with eachother and be completely interchangeable. 

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

```{note}
### The `bind` argument

Whenever a function has a `bind` kwarg it means that whatever is being computed can be intenalised to the `Graph` representation of the neuron. This can be done with *any* python object in theory, and additionally if the object can be pickled it will be saved along with the `.nr` file for the neuron persistently. This means the output of analyses pipelines will pesist with your dataset and be bound to you dataset. See {doc}`../tutorials/tree_basics` for more about properties.
```

**Root index is always 0** after import. Node indices align across all vertex
properties, so `tree.get_branch_indices()` subsets `tree.get_property('radius')`
directly.

## File formats: when to use what

| Format   | Role                                  | Metadata & custom properties                                                  |
| -------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| **SWC**  | Interchange with other tools          | Lost on export typically (or limited, depending on what goes into the header) |
| **NR**   | Native NeuRosetta / graph-tool format | Fully preserved                                                               |
| **Mesh** | Surfaces for rendering & distance ops | Partial (via vedo + metadata)                                                 |

Rule of thumb:

- **SWC in, SWC out** — sharing with navis, neuromorpho, etc.
- **NR for everything else** — editing sessions, custom graph properties, units,
  reduced-tree flags.

Details: {doc}`io`.

NeuRosetta generally much prefers `.nr` files, and they are geneally significantly smaller/faster to read and write than standad text based `.swc` files. We will include other file formats eventually however.

## Reduced trees

Full SWC reconstructions contain many **transitive** nodes (degree-2 chain
points). **Reduction** collapses those while keeping root, branch, and leaf
nodes — smaller graphs, less memory, faster ops. Additionally, NeuRosetta will do it's best to preserve elements from the 'full' reconstuction across to the 'reduced' version, such as cable length. 

NeuRosetta tracks reduction in `metadata["isReduced"]`. Cable-length semantics
change after reduction: compute `Path_length` **before** reducing if you need
path length along the original arbor. See {doc}`../tutorials/tree_surgery`.

## Spatial units

Coordinates are meaningless for cross-dataset analysis without units.
NeuRosetta stores scale in `metadata["units"]` (via Pint), supports nm/µm/voxel
specs, and can rescale geometry in place.

See {doc}`../reference/units` and the units section in
{doc}`../tutorials/tree_basics`.

## Scripting vs GUI

| Task                                   | Best tool                                                       |
| -------------------------------------- | --------------------------------------------------------------- |
| Batch import, metrics, figures         | Python API                                                      |
| Quick inspection, reroot, subtree pick | {doc}`../tutorials/gui` (`run_neuro_GUI`)                       |
| Publication plots                      | matplotlib (`show_2d`, dendrogram) + vedo (`show_3d`, `Viewer`) |

The GUI calls the same ops as the library; prefer scripting for reproducible
pipelines. However, for opertation such as re-rooting neuron morphologies, or specifying a specific sub-tree from a node, this can be done and verified manually using the GUI (and viewed, saved, or flagged). See {doc}`../tutorials/gui`.

## Typical workflow

```python
from pathlib import Path
import neurosetta as nr

# 1. Load
forest = nr.import_swc(Path("swc_dir/"), set_units="nm", progress=True)

# 2. Analyse - get the total cabloe length of every neuron in the forest
lengths = forest.get_total_cable_length()

# 3. Edit (optional) - reduce the first tree in the forest in place
forest.by_id(forest.ids()[0]).get_reduced_tree(inplace=True)

# 4. Visualise - open a 3D inteactive viewer plotting all neuons in the forest
forest.show_3d()

# 5. Save native format
forest.save_forest("out/nr/")
```
## Making Contributions

Contributions are welcome — bug reports, docs fixes, tests, and new features. Make sure to open an issue on the NeuRosetta GitHub repository for improvements, feature suggestions, critisisms as long as they are amusingly worded, or any other suggestions!

1. **Fork & clone** the repo, then create a conda environment:

   ```bash
   git clone https://github.com/NikDrummond/NeuRosetta.git
   cd NeuRosetta
   mamba env create -f environment.yml
   mamba activate nr
   ```

   Or, if dependencies are already installed:

   ```bash
   python -m pip install -e ".[dev,docs]"
   ```

2. **Make changes** on a feature branch. Keep new tree/forest logic in `ops/`
   (not `utils/` or `core/` directly); see the
   [architecture guide](https://nikdrummond.github.io/NeuRosetta/development/architecture.html)
   for layer rules.

3. **Test & lint** before opening a PR:

   ```bash
   python -m pytest
   ruff check src tests
   ruff format src tests
   ```

4. **Open a pull request** against `main` with a short description of the
   change. Link any related
   [issue](https://github.com/NikDrummond/NeuRosetta/issues) if one exists.

Questions or ideas with no code yet? Open an issue first.

Before diving into adding your own things, it is a good idea to know about the achitectural stucture of NeuRosetta, so have a look at {doc}`../development/architecture`.

Next: {doc}`installation`.