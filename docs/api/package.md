# Package

Public symbols re-exported from :mod:`neurosetta`:

| Symbol | Description |
|--------|-------------|
| `Tree` | Single neuron — see {doc}`tree` |
| `Forest` | Tree collections — see {doc}`forest` |
| `Tree_mesh`, `Forest_mesh`, `Neuropil`, `Neuropils` | Mesh containers — see {doc}`api_classes` |
| `import_swc`, `export_swc`, `load`, `save` | Morphology I/O — see {doc}`io` |
| `import_mesh`, `export_mesh` | Mesh I/O — see {doc}`io` |
| `Viewer` | 3D viewer — see {doc}`plotting` |
| `reconstruct_neuropil_surface` | Surface reconstruction — see {doc}`analysis` |
| `start_GUI` | Desktop GUI — see {doc}`gui` |
| `configure`, `get_settings`, `settings` | Global configuration — see {doc}`../reference/configuration` |
| `openmp_context`, `openmp_enabled`, `sync_vedo_runtime` | OpenMP and vedo runtime helpers — see {doc}`../reference/configuration` |

Version is available as ``neurosetta.__version__``.

Functional tree operations live under {doc}`tree_ops/index` (also bound as
Tree/Forest methods where noted in each function's docstring).
