# Neuron meshes (`tree.mesh`)

NeuRosetta treats a neuron surface mesh as a **facet attached to a morphology**,
same pattern as synapses — **not** a parallel mesh-only neuron API.

```text
Tree.graph          skeleton morphology (required today)
Tree.synapses       optional synapse table
Tree.mesh           optional neuron surface (Tree_mesh payload)
```

`Tree_mesh` / `Forest_mesh` stay as **I/O and payload types**
(`import_mesh` / `export_mesh` / facet value). Prefer attach + work from
`Tree` / `Forest`. Neuropil / compartment meshes (`Neuropil`) are a different
role — do not attach them with `set_mesh`. See {doc}`../api/api_classes`.

## Minimal workflow

```python
import neurosetta as nr

tree = nr.load("neuron.nr")          # or import_swc(...)

# preferred: path → attach in one step (logical mesh.ID must match tree.ID;
# file stem becomes mesh.name and, by default, mesh.ID)
tree.set_mesh("7.ply", set_units="um")

# differently named files: set logical ID independently
# mesh = nr.import_mesh("neuron_7_surface.ply", ID=7, set_units="um")
# tree.set_mesh(mesh)

tree.has_mesh()
tree.mesh.count_vertices()

# skeleton + attached mesh in one viewer (opt-in)
tree.show_3d(show_mesh=True, mesh_kwargs={"alpha": 0.35, "c": "lightblue"})
```

`mesh=` is an alias of `show_mesh=`. Overlay defaults **off**.

Facet geometry alone (debug / QC) still works via the payload::

    tree.mesh.show_3d(alpha=0.4)

## Units and transforms

On bind, units are stamped / checked like synapses. Spatial ops on the tree
move the mesh with the skeleton:

```python
tree.set_mesh("7.ply", set_units="um")
tree.convert_units("nm")   # scales morphology + mesh; stamps mesh units
tree.translate(10, 0, 0)   # moves both
```

## `.nr` round-trip

Attached neuron meshes freeze into `.nr` as verts/faces (vedo meshes are not
pickle-safe). Trees without a mesh load unchanged.

```python
tree.set_mesh("7.ply", set_units="um")
tree.save_tree("7.nr")
loaded = nr.load("7.nr")
assert loaded.has_mesh()
```

## Forest batch attach

Match meshes to trees by logical `ID` (same int/str coercion as synapse
`owner_id`). File stems become mesh `name` and, by default, mesh `ID`
(`7.ply` → `name="7"`, `ID="7"` matches tree `ID=7`). Use `id_resolver` /
`id_map` when filenames differ from logical IDs.

```python
forest = nr.load("neurons/")                 # Forest of Trees
forest.set_meshes("meshes/", set_units="um") # preferred over Forest_mesh-as-API

forest.show_3d(show_mesh=True)
```

Policies:

| Kwarg | Default | On miss / leftover |
|-------|---------|-------------------|
| `missing` | `"warn"` | tree with no matching mesh |
| `unused` | `"ignore"` | mesh with no matching tree |

Use `missing="error"` / `unused="error"` in pipelines that must be complete.

## What `Tree_mesh` / `Forest_mesh` are for

| Type | Keep using for | Prefer instead for analysis |
|------|----------------|------------------------------|
| `Tree_mesh` | `import_mesh` file, `export_mesh`, `tree.mesh` value | `Tree` + `set_mesh` / `show_3d(show_mesh=True)` |
| `Forest_mesh` | `import_mesh` directory return, batch export | `Forest.set_meshes` + `Forest.show_3d(show_mesh=True)` |

No runtime removal — soft deprecation of the **story**, not the classes.

## Viewer composition

```python
v = nr.Viewer()
v.add_neuron(tree, show_mesh=True, mesh_kwargs={"alpha": 0.3})
v.add_mesh(neuropil, alpha=0.2, c="gray")   # Neuropil — separate add
v.show()
```

## See also

- {doc}`../api/api_classes` — `Tree_mesh` / `Neuropil` taxonomy
- {doc}`../api/tree_ops/mesh` — `set_mesh` / `clear_mesh` / `has_mesh` / `set_meshes`
- {doc}`../api/io` — `import_mesh` / `export_mesh`
- {doc}`../api/plotting` — `plot_mesh`, `Viewer.add_mesh`
