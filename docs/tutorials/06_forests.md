# Forests

A {class}`~NeuRosetta.api.Forest` is an ordered collection of
{class}`~NeuRosetta.api.Tree` objects with batch helpers, metadata filtering,
and parallel `apply`.

## Build a forest

From a directory of SWC files:

```python
from pathlib import Path
import NeuRosetta as nr

forest = nr.import_swc(Path("docs/data"), progress=True)
print(len(forest), forest.ids())
```

Or assemble manually:

```python
t1 = nr.import_swc(Path("docs/data/1.swc"))
t2 = nr.import_swc(Path("docs/data/2.swc"))
forest = nr.Forest([t1, t2])
```

Forests behave like sequences:

```python
for tree in forest:
    print(tree.ID, tree.count_nodes())

print(forest.by_id(1).count_leaves())
```

## Metadata filter

Attach metadata, then subset:

```python
forest.by_id(1).metadata["Neuron_type"] = "A"
forest.by_id(2).metadata["Neuron_type"] = "B"

subset = forest.filter(Neuron_type="A")
print(subset.ids())

# membership (OR within a key)
subset = forest.filter(Neuron_type=["A", "B"])
```

Callable predicates are also supported:

```python
subset = forest.filter(lambda m: m.get("Neuron_type") == "A")
```

## Batch descriptives

Forest methods apply tree ops across all members (often with optional
`parallel`, `max_workers`, `progress`, and `bind` keywords — see the method
docstrings):

```python
print(forest.count_nodes())
print(forest.count_leaves())
print(forest.get_total_cable_length())
```

## Custom `apply`

```python
node_counts = forest.apply(lambda t: t.count_nodes(), parallel=True)
print(node_counts)

# Side-effecting updates (sequential is often clearer here)
def _mark(tree):
    tree.metadata["checked"] = True

forest.apply(_mark, parallel=False)
```

## Reduce and export

```python
from pathlib import Path
import tempfile

out = Path(tempfile.mkdtemp())

# reduce each tree in place via the batch wrapper
forest.reduce_tree(inplace=True)

forest.export_forest_to_swc(out / "swc")
forest.save_forest(out / "nr")
```

## Visualise

```python
forest.show_3d()
```

Next: {doc}`07_meshes_neuropils`.
