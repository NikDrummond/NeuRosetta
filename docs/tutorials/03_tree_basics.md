# The Tree Class

The {class}`~NeuRosetta.api.Tree` class, and its big brother, {class}`~NeuRosetta.api.Forest`, is the way to handle and manipulate neurons within NeuRosetta. 

## Basic Tree Poperties

All Trees have three main properties: ID, metadata, and graph. As the Tree class makes use of __slots__, you cannot add arbritray attributes to Trees, but rather have to store them within either metadata, or on the graph. This is easy to do howeve, and when done you can save these elements with the `.nr` file, giving you a persistent state across sessions. You can look at the basis attributes as follows:

```python
from pathlib import Path
import NeuRosetta as nr

tree = nr.import_swc(Path("docs/data/1.swc"))
print(tree)
print(tree.ID)
print(tree.metadata)
print(tree.graph)

```

As explained in {doc}`02_io`, the `ID`of a neuron comes from the file name / identification number within the dataset this neuron comes from. Metadata is a place to keep track of information about this neuron. Finally, graph is the core graph representation of the neuron tree graph and is a `graph_tool.Graph` object. It is pottentially very useful to have a working understanding of how graph_tool works, documentation for which can be found [here](https://graph-tool.skewed.de/), however I will cover some important bits throughout these tutorials.

### Understanding Graphs



### Understanding Metadata

The base metadata dictionary is like so:

| Key | Value |
|--------|-----------|
| 'units'| Assumed to be undefined at import |
| 'file_path'| The location of the `.swc` file|
| 'isReduced'| This is a bool flag, explained {doc}`here <03_tree_basics>`|



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
