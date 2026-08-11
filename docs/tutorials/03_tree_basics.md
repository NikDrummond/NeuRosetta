# The Tree Class

The {class}`~NeuRosetta.api.Tree` class, and its big brother, {class}`~NeuRosetta.api.Forest`, is the way to handle and manipulate neurons within NeuRosetta.

## Tree Basics

All trees have three main properties: ID, metadata, and graph. As the Tree class makes use of __slots__, you cannot add arbitrary attributes to Trees, but rather have to store them within either metadata, or bound to the graph. This is easy to do however, and when done you can save these elements with the `.nr` file, giving you a persistent state across sessions. You can look at the basic attributes as follows:

```python
from pathlib import Path
import NeuRosetta as nr

tree = nr.import_swc(Path("docs/data/1.swc"))
print(tree)
print(tree.ID)
print(tree.metadata)
print(tree.graph)

```

As explained in {doc}`02_io`, the `ID` of a neuron comes from the file name / identification number within the dataset this neuron comes from. Metadata is a place to keep track of information about this neuron. Finally, graph is the core graph representation of the neuron tree graph and is a `graph_tool.Graph` object.

---

### Understanding Graph Properties.

As much as possible, NeuRosetta aims to be as easy to use as possible, but understanding properties will help a little. NeuRosetta tries to handle as much of this under the hood as possible, so in theory you never need to fully understand this. What is important is knowing that a bound property — a property that shows up when you look at the list of properties a neuron has — is savable, and can be persistently attached to the neuron.

Each neuron is treated as a directed tree graph within NeuRosetta. This is done using [graph_tool](https://graph-tool.skewed.de/), which under the hood is a C++ library which gives us some nice speed in some places. There are a few elements which are key to understanding how this works within NeuRosetta, possibly most importantly the idea of properties. Graph_tool gives an overview [here](https://graph-tool.skewed.de/static/docs/stable/quickstart.html#property-maps) which is worth going through. When a property is "bound" to the graph within NeuRosetta, it is an internal property for the `graph_tool.Graph` object. There are three levels of properties, graph properties, vertex properties, and edge properties (vertex is another word for a node in a graph). We use the shorthand ['g','v','e'] for graph, vertex, and edge properties respectively.

To view the properties bound to a tree, you can view them all using:

```python
tree.list_properties()
```

Alternatively, you can use any of ['g','v','e'] to only look at the properties at that specific level:

```python
tree.list_properties('g')
```

You can check if a tree has a specific property if you know the property name as follows:

```python
tree.has_property('coordinates', level = 'all')
```
which will return True/False. You can additionally use the `level` to search specifically within the ['g','v','e'] properties. It is 'all' by default though.

In order to get the values of a property, by default as a numpy array, you should know its level and name, and then can use:

```python
tree.get_property(prop = 'radius',level = 'v', as_array = True)
```

This will give you the numpy array of node radii. Alternatively, you may need the graph_tool property map object itself as documented [here](https://graph-tool.skewed.de/static/docs/stable/graph_tool.html#property-maps), in which case set `as_array = False`. The `level` argument is not needed either. If you don't use it, NeuRosetta will still find your property. If however you have set two properties with identical names at different levels, you will get a dictionary of `{'level':'property values'}` returned.

You can also set or create new properties at any level. When creating a new property, you need data (obviously), as well as knowing the level and datatype. When creating a new property you *need* to specify the datatype. There is a table of usable datatypes [here](https://graph-tool.skewed.de/static/docs/stable/quickstart.html#property-maps). Bound properties can either be 1 or 2 dimensional arrays (at the ['v','e'] levels), or more flexibly any picklable python object (using the `object` dtype) at graph level.

To set a 1-dimensional array, in the below example a value for each node in the neuron, you need your dataset — an array of length equal to the number of nodes, and can set it as follows:

```python
data = np.zeros(tree.count_nodes())
tree.set_property(name = 'zeros', data = data, level = 'v', create = True, dtype = 'int')
tree.get_property('zeros')
```

When setting a 2-dimensional array, for example, the coordinates of points, NeuRosetta will try to make sure the shape is correct, as Graph_tool expects the array in the format dimensions x *n*. Additionally, the datatype needs to be `vector<dtype>`, for example `vector<double>` for an array of floats. Below, we repeat the above operation, but for a 2D array:

```python
data_2d = np.ones((tree.count_nodes(),2))
tree.set_property(name = 'ones', data = data_2d, level = 'v', create = True, dtype = 'vector<int>')
tree.get_property('ones')
```
This creates a vertex(node) property as a 2-dimensional array of ones, with the name 'ones' at the level 'v'.

`tree.set_property()` also works for updating existing property values, which works the same way as above. Lets modify the 'zeros' property:

```python
tree.set_property(name = 'zeros', data = data + 1, level = 'v')
tree.get_property('zeros')
```

Note that you do not need to set the datatype, and you can ignore the `create` argument. Keep in mind however, that once a property is created, the datatype is **fixed**, unless you create a new property with a different name or delete the existing property!

Speaking of, deleting properties is also easy:

```python
tree.del_property('zeros', level = 'v')
tree.list_properties(level = 'v')
```

````{warning}
You cannot use `tree.set_property` to change between 1 and 2 dimensional arrays - `NeuRosetta` will give you a value error. For example, 

```python
data = np.zeros(tree.count_nodes())
tree.set_property(name = 'zeros', data = data, level = 'v', create = True, dtype = 'int')

data_2d = np.ones((tree.count_nodes(),2))
tree.set_property(name = 'zeros', data = data_2d, level = 'v', create = True, dtype = 'vector<int>')
```
Will not work, nor will trying to set a 1-d array to a property which is a 2-d array.

````

---

### Getting Basic Neuron Information.

You can get counts of various things from your neuron like so:

```python
print(f"This neuron has {tree.count_nodes()} nodes")
print(f"This neuron has {tree.count_branches()} branching nodes")
print(f"This neuron has {tree.count_leaves()} terminal nodes")
print(f"This neuron has {tree.count_roots()} root node")
```
The above tells you how many nodes the neuron is made up of, the number of branching nodes (those with an out degree > 1), the number of terminal nodes (or leaves, nodes with an out degree = 0), and the number of root nodes (nodes with an in-degree = 0). Strictly speaking, you should only ever have a single root node. If you have multiple, you have multiple disjoint graphs in your graph representation of your neuron, and something is wrong.

Additionally, we have various functions to get the indices of nodes in the graph representation of your neuron:

```python
print(tree.get_root_index())
print(tree.get_branch_indices())
print(tree.get_leaf_indices())
print(tree.get_edge_indices())
```
NeuRosetta will aim for the root node index to *always* be at the 0 index. You will see that `tree.get_edge_indices()` returns a 2-dimensional array. The first column is the source node index and the second the target node index for each edge.

Having access to the index of specific nodes is important, as the node ordering in all your node level properties is consistent, meaning you can use these indices to subset. For example, if you want to know the radius of nodes at each branch point:

```python
radii = tree.get_property('radius')
print(radii[tree.get_branch_indices()])
```

You can also easily get the coordinates of nodes and edges. To get the coordinates of all nodes as a numpy array:

```python
coord = tree.get_node_coordinates(subset = None)
```
Note here that we used the `subset = None` argument. This is the default behaviour, and gives you the coordinates of all nodes in your neurons. You can pass a subset of node indices, or one of the above index functions, to subset the coordinates you are grabbing:

```python
branch_coordinates = tree.get_node_coordinates(subset = tree.get_branch_indices())
```
You also have a useful helper to just get the coordinate of the root node:

```python
root_coordinate = tree.get_root_coordinate()
```

Much like when looking at the indices of edges, the same thing can be done for coordinates, if you would like that source and target coordinates of each edge within your neuron:

```python
source_c, target_c = tree.get_edge_coordinates()
```
This returns a tuple of coordinate arrays, which we are splitting into source and target above. Currently, subsetting doesn't work here as an argument, although this will likely change in the future! A full list of Tree methods can be found in the API documentation on trees.

---

### Understanding Metadata

The base metadata dictionary is like so:

| Key | Value |
|--------|-----------|
| 'units'| Assumed to be 'dimensionless' at import unless specified|
| 'file_path'| The location of the `.swc` file|
| 'isReduced'| This is a bool flag, explained in {doc}`04_tree_surgery`|

Metadata is a graph-level property, so will be saved alongside any edits made to it when using the `.nr` file format. When we look at the {class}`~NeuRosetta.api.Forest` the metadata is particularly useful, as you can add information on neuron types, for example, and use it to filter your forest for different neurons. It is also how NeuRosetta keeps track of a neuron's spatial units, which is covered below, or the reduced or flagged nature of a neuron, which will be relevant later.

---

### Tree Spatial Units

By default, spatial units of your neurons are dimensionless unless they have been set previously in a neurons metadata, or at import using the `set_units` argument.

You can check if the units of a neuron have been defined using:

```python
tree.check_units_defined()
```
Which will return an ValueError if they haven't been, or nothing. You can also have a look at whatever it says in the neurons metadata. As we use [Pint](https://pint.readthedocs.io/en/stable/) for unit handling, you can set your neurons units using a string:

```python
tree.set_units('nm')
```

Which sets the neuron as being in nanometer units. You can also convert between units using:
```python
tree.convert_units('um')
```
Which converts the neuron's units in place to microns. NeuRosetta can also handle data in isotropic voxels, which can be set and saved within the metadata. To set voxel units:

```python
tree.set_voxel_units(voxel_size = 8, voxel_unit = 'nm')
tree.get_voxel_spec()
```
This tells NeuRosetta that the spatial units of this neuron are in voxel units and each voxel is 8x8x8nm. Using this, you can convert voxel units to nanometers or microns easily using the same `tree.convert_units` functionality above. For specifying spatial units, there are a few aliases and options for strings, some useful ones for connectomics data are as follows:

|Canonical|Aliases|
|-|-|
|`dimensionless`|`undefined`, `Undefined`, `1 dimensionless`|
|`nanometer`|`nm`,`Nanometer`|
|`micron`|`Micron`,`micrometer`,`Micrometer`,`microns`,`Microns`,`um`|
|`voxel`|`voxel`(+`voxel_size`/`voxel_unit` metadata)|

Keep in mind with voxel units, you should use one of the other spatial units, such as 'nm'.

---

## Functional vs method API

Every tree method is a thin bind of a function in
{mod}`NeuRosetta.ops.tree_graphs`. These are equivalent:

```python
import NeuRosetta as nr

assert tree.count_nodes() == nr.count_nodes(tree)
```

Use the method API for everyday work; use the functional API when writing
reusable pipelines or applying ops through `Forest.apply`.

Next: {doc}`04_tree_surgery`.
