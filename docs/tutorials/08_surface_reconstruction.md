# Surface reconstruction and neuropil distances

{func}`~NeuRosetta.reconstruct_neuropil_surface` builds a neuropil mesh from
the combined point cloud of a {class}`~NeuRosetta.api.Forest` (voxel grid →
marching cubes → optional clean/smooth).

## Units requirement

Every tree in the forest must declare nanometre units:

```python
tree.metadata["units"] = "nm"
```

Coordinates are converted to microns inside the reconstruction routine.

## Minimal example

With a real EM forest in nm:

```python
from pathlib import Path
import NeuRosetta as nr

forest = nr.import_swc(Path("path/to/swc_dir"))
for tree in forest:
    tree.metadata["units"] = "nm"

neuropil = nr.reconstruct_neuropil_surface(
    forest,
    name="example_neuropil",
    voxel_size=3,          # microns
    remove_outliers=True,
    clean=True,
    smooth=True,
)
print(neuropil.ID, neuropil.mesh)
nr.export_mesh(neuropil, "example_neuropil.ply")
```

Useful knobs (see the API docs for full detail):

| Parameter | Meaning |
|-----------|---------|
| `voxel_size` | Voxel size in **microns** |
| `remove_outliers` | kNN / quantile outlier filter |
| `padding` / `dilate` | Grow the occupied voxel mask |
| `largest_component` | Keep only the largest blob |
| `clean` / `smooth` | Mesh post-processing |

```{note}
The bundled ``docs/data`` sample SWCs are single-neuron excerpts with
dimensionless units — not suitable inputs for neuropil reconstruction. Use a
dense EM forest in nm.
```

## Distances and depth

Once you have a neuropil mesh, distance helpers live under
{mod}`NeuRosetta.ops.neuropils.distances`:

```python
from NeuRosetta.ops.neuropils.distances import (
    distance_from_neuropil_surface,
    neuropil_point_depth,
)

points = forest.by_id(forest.ids()[0]).get_node_coordinates()
# d = distance_from_neuropil_surface(neuropil, points)
# depth = neuropil_point_depth(neuropil, points)
```

Next: {doc}`09_gui`.
