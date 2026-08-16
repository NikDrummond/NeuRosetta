# NeuRosetta Roadmap

NeuRosetta is in **active development**. This page tracks planned and in-progress
work so you can see where the library is headed. It is updated as priorities shift;
timelines are approximate.

For how the codebase is organised today, see {doc}`../development/architecture`.

## Available now

- **Tree / Forest** graph operations — counting, traversals, coordinates, cable
  length, subtrees, rerooting, reduction
- **I/O** — SWC and native `.nr` files; bundled FlyWire example data
- **Plotting** — 2D matplotlib, dendrogram, vedo 3D viewer
- **GUI** — interactive inspection, reroot, subtree extraction
- **Units** — pint-backed spatial units on trees and forests
- **Neuropil surfaces** — reconstruction helpers and distance ops (early API)

See {doc}`overview` for the mental model, then {doc}`installation` and
{doc}`example_data` to get running.

## Near term

| Area                 | Direction                                                            |
| -------------------- | -------------------------------------------------------------------- |
| **Synaptic data**    | Integrate synapse / connectivity data alongside morphology           |
| **Topology**         | Morphology analyses beyond geometry (branching structure, motifs)    |
| **Meshes**           | Round out neuron and neuropil mesh import/export and batch workflows |
| **Metrics**          | Expand the descriptives registry — see {doc}`../reference/metrics`   |
| **Docs & tutorials** | More notebooks, clearer GUI and analysis walkthroughs                |

## Distribution

| Channel         | Status                                                                          |
| --------------- | ------------------------------------------------------------------------------- |
| **Git install** | Supported (`pip install git+https://github.com/NikDrummond/NeuRosetta.git`)     |
| **PyPI**        | Planned                                                                         |
| **conda-forge** | Planned (`graph-tool` + optional GUI deps remain the main packaging constraint) |

Details: {doc}`installation`.

## Longer term

- Richer **Forest** batch analysis and filtering presets
- **Connectomics** workflows tying morphology to connectome segment IDs and atlases
- Performance work on large reconstructions (parallel I/O, reduced-memory paths)
- Stable **public API** surface with semver once PyPI release lands

## Feedback

Open an issue or discussion on
[GitHub](https://github.com/NikDrummond/NeuRosetta) if you need a feature for
your pipeline — it helps set priorities.

Contributors: {doc}`../development/extending_nr` and {doc}`../development/architecture`.
