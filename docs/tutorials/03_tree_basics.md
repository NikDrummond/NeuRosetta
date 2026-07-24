# Tree basics

The {class}`~NeuRosetta.api.Tree` class is the primary object for a single
neuron morphology. It wraps a directed graph-tool graph plus an integer `ID`
and a `metadata` dictionary.

## Load a tree

```python
from pathlib import Path
import NeuRosetta as nr

tree = nr.import_swc(Path("docs/data/1.swc"))
```

## Topology queries

```python
print("root:", tree.get_root_index())
print("leaves:", tree.get_leaf_indices())
print("branches:", tree.get_branch_indices())
print("core:", tree.get_core_indices())
print("edges:", tree.get_edge_indices()[:3], "...")
```

## Counts

```python
print(tree.count_nodes())
print(tree.count_edges())
print(tree.count_roots())
print(tree.count_leaves())
print(tree.count_branches())
print(tree.count_transitive_nodes())
```

Transitive nodes are the chain vertices with in-degree = out-degree = 1
(removed by tree reduction; see {doc}`04_tree_surgery`).

## Geometry

```python
coords = tree.get_node_coordinates()
print(coords.shape)  # (n_nodes, 3)

edge_lengths = tree.get_edge_length()
cable = tree.get_total_cable_length()
print(float(cable))
```

## Degrees and depths

```python
degrees = tree.get_degrees()
print(tree.get_degree_distribution())

depths = tree.get_node_depth()
print(depths.min(), depths.max())
```

## Properties and metadata

Trees store analysis caches and flags as graph / metadata properties:

```python
print(tree.list_properties())
print(tree.has_property("coordinates"))
print(tree.is_reduced())
tree.metadata["Neuron_type"] = "example"
print(tree.metadata["Neuron_type"])
```

Set `units` when you know them (important for surface reconstruction):

```python
tree.metadata["units"] = "nm"  # or "µm" / "um" depending on your data
```

```{note}
`reconstruct_neuropil_surface` currently expects forest metadata
`units == "nm"` and converts coordinates to microns internally. See
{doc}`08_surface_reconstruction`.
```

## Functional vs method API

Every tree method is a thin bind of a function in
{mod}`NeuRosetta.ops.tree_graphs`. These are equivalent:

```python
from NeuRosetta.ops.tree_graphs import count_nodes

assert tree.count_nodes() == count_nodes(tree)
```

Use the method API for everyday work; use the functional API when writing
reusable pipelines or applying ops through `Forest.apply`.

Next: {doc}`04_tree_surgery`.
