# Meshes and neuropils

NeuRosetta wraps vedo meshes in thin container classes:

| Class | Role |
|-------|------|
| `Tree_mesh` | Single neuron surface |
| `Forest_mesh` | Collection of neuron surfaces |
| `Neuropil` | Single neuropil / brain-region surface |
| `Neuropils` | Collection of neuropils |

These classes mainly store `ID`, `metadata`, and a vedo `mesh` handle.
Styling is done on the underlying vedo object until higher-level helpers
land.

## Import / export

```python
import NeuRosetta as nr

# Neuron surface
# neuron = nr.import_mesh("42.ply", mesh_type="Neuron")
# print(neuron.ID, neuron.mesh)

# Neuropil surface
# neuropil = nr.import_mesh("AL.ply", mesh_type="Neuropil")

# Directory → Forest_mesh or Neuropils
# meshes = nr.import_mesh("mesh_dir/", mesh_type="Neuron")

# Export (default binary PLY)
# nr.export_mesh(neuropil, "out/AL.ply")
# nr.export_mesh(meshes, "out_meshes/")
```

## Working with the vedo mesh

```python
# neuropil.mesh.color("lightblue").alpha(0.3)
# neuropil.mesh  # pass to vedo Plotter / NeuRosetta Viewer workflows
```

`Forest_mesh` and `Neuropils` reuse the same collection API as forests
(`ids`, `filter`, `apply`, indexing).

## Where meshes come from

1. External tools / EM pipelines → `import_mesh`
2. NeuRosetta analysis → {func}`~NeuRosetta.reconstruct_neuropil_surface`
   (next tutorial)

Next: {doc}`08_surface_reconstruction`.
