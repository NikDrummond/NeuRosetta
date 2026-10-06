# Package

Public symbols re-exported from :mod:`neurosetta`:

| Symbol | Description |
|--------|-------------|
| `Tree` | Single neuron — see {doc}`tree` |
| `Forest` | Tree collections — see {doc}`forest` |
| `Workspace` | Analysis session / `.nrw` — see {doc}`workspace` |
| `Synapses` | Synapse table container — see {doc}`synapses` |
| `Tree_mesh`, `Forest_mesh`, `Neuropil`, `Neuropils` | Mesh containers (neuron mesh = attach to Tree) — see {doc}`api_classes` |
| `import_swc`, `export_swc`, `load`, `save` | Morphology I/O — see {doc}`io` |
| `save_workspace`, `load_workspace`, `inspect_workspace` | Workspace I/O — see {doc}`io` |
| `import_mesh`, `export_mesh` | Mesh I/O — see {doc}`io` |
| `set_mesh`, `clear_mesh`, `has_mesh`, `set_meshes` | Neuron mesh facet — see {doc}`tree_ops/mesh` |
| `set_synapses`, `map_synapses`, … | Synapse facet (Tree + Forest batch attach) — see {doc}`tree_ops/synapses` |
| `import_synapses`, `extract_synapses`, `export_synapses` | Synapse table I/O — see {doc}`io` |
| `Viewer` | 3D viewer — see {doc}`plotting` |
| `reconstruct_neuropil_surface` | Surface reconstruction — see {doc}`analysis` |
| `start_GUI` | Desktop GUI — see {doc}`gui` |
| `configure`, `get_settings`, `settings` | Global configuration — see {doc}`../reference/configuration` |
| `openmp_context`, `openmp_enabled`, `sync_vedo_runtime` | OpenMP and vedo runtime helpers — see {doc}`../reference/configuration` |

Version is available as ``neurosetta.__version__``.

Functional tree operations live under {doc}`tree_ops/index` (also bound as
Tree/Forest methods where noted in each function's docstring).
