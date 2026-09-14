# Synapses

NeuRosetta treats synapses as **observations attached to a morphology**, not as
extra graph vertices. The canonical morphology-relative location of a mapped
synapse is the continuous pair:

```text
(edge_index, edge_fraction)
```

where `edge_fraction = 0` is the edge source and `1` is the edge target.
Cable distance along the edge (`distance_along_edge`) is also stored and is the
preferred quantity when transferring mappings through reduction.

## Two graph levels

```text
Tree.graph
    morphology graph — vertices = skeleton nodes, edges = neurites

Forest.get_connectivity_graph()
    network graph — vertices = neurons / Trees, edges = synaptic connections
```

These are never merged. Connectivity is derived from synapse tables attached to
Trees.

## Terminology

| Label | Meaning on this neuron |
|-------|-------------------------|
| `pre` / `output` | Presynaptic site — output **from** this neuron |
| `post` / `input` | Postsynaptic site — input **onto** this neuron |

`partner_id` is the connected neuron ID. Direction therefore depends on type:
a `pre` synapse with partner `B` means this neuron → `B`; a `post` synapse with
partner `B` means `B` → this neuron.

`"both"` always means **select pre and post**, never a third synapse type.

## Minimal workflow

```python
import neurosetta as nr
import pandas as pd

tree = nr.load("neuron.nr")

df = pd.DataFrame({
    "synapse_id": [...],
    "type": ["pre", "post", ...],   # or output / input
    "x": [...], "y": [...], "z": [...],
    "partner_id": [...],            # any hashable ID
    # optional columns freely allowed:
    "confidence": [...],
    "cleft_id": [...],              # recommended for Forest deduplication
})

tree.set_synapses(df)
tree.map_synapses()                 # nearest edge + continuous fraction

tree.synapses.pre
tree.synapses.post
tree.count_synapses(type="post")
tree.synapse_density(type="post")   # synapses / cable length

tree.show_3d(show_synapses=True)
```

## Reduction (map before reduce)

Reduction collapses transitive morphology nodes into section edges. If synapses
are already mapped, their **section identity and cable position** are transferred
deterministically — nearest-edge search is **not** re-run.

```python
tree.set_synapses(synapses)
tree.map_synapses()
before = tree.get_synapse_path_distance()

tree.get_reduced_tree(inplace=True)

after = tree.get_synapse_path_distance()
assert (before == after).all()  # cable position invariant
```

Preserved across reduction:

- synapse count / IDs / partners / types
- raw coordinates
- mapped coordinates and `distance_to_tree`
- root path distance

Changed (by design):

- `edge_index` (new reduced edge)
- `edge_fraction` / `distance_along_edge` (relative to the full section)

If you reduce **before** mapping, later `map_synapses()` can only use the reduced
geometry (straight chords). Prefer map-then-reduce for exact section positions.

## Forest connectivity

```python
forest = nr.Forest([tree_a, tree_b, tree_c])

table = forest.get_connectivity_table()
# columns: source_id, target_id, synapse_count

g = forest.get_connectivity_graph()
assert g.is_directed()
# vp["tree_id"], vp["in_forest"], ep["synapse_count"]

# Use graph-tool directly:
from graph_tool.centrality import pagerank
pr = pagerank(g, weight=g.ep["synapse_count"])
```

Direction:

- `pre` on A with partner B → `A → B`
- `post` on A with partner B → `B → A`

Deduplication (`deduplicate=`):

| Mode | Behaviour |
|------|-----------|
| `auto` (default) | Use `cleft_id` when present; otherwise count records independently |
| `id` | Require `cleft_id` or `synapse_id`; each ID counted once |
| `none` | Count every record (may double-count shared physical synapses) |

```python
g = forest.get_connectivity_graph(include_external=True, min_synapses=5)
forest.get_in_degree()      # distinct input partners
forest.get_out_strength()   # total output synapses
```

Connectivity is **invariant to morphological reduction**.

## Raw vs mapped coordinates

| Field | Meaning |
|-------|---------|
| `x,y,z` | Original connectomics observation (immutable observation) |
| `nearest_x/y/z` | Projection onto the morphology |
| `distance_to_tree` | Euclidean residual of that projection |
| `mapped` | `False` when beyond `max_distance` (row kept unless dropped) |

```python
tree.map_synapses(max_distance=100)                 # flag far synapses
tree.map_synapses(max_distance=100, drop_unmapped=True)  # delete them
tree.synapse_mapping_summary()
tree.show_3d(show_synapses=True, synapse_position="raw", show_synapse_mapping=True)
tree.show_2d(synapses="post", synapse_colour_by="partner_type")
```

Extra annotation columns (`partner_type`, neurotransmitter, …) are free-form
DataFrame columns — use `tree.synapses.add_column(...)` or include them in the
input table / `filter(**column_equals)`.

## Table schema

Required columns:

```text
synapse_id, type, x, y, z, partner_id
```

Import / export:

```python
nr.import_synapses("synapses.csv", tree=tree)
nr.export_synapses(tree, "synapses_out.csv")
```

## Serialization

Synapses are stored as a graph-tool object property `gp["synapses"]` inside
`.nr` files. Trees without synapses load unchanged (backward compatible).
Mapping columns (including post-reduction transfers) round-trip when present.

## Transforms, editing, subtrees

| Operation | Synapses |
|-----------|----------|
| translate / rotate / uniform scale / unit convert | Raw + mapped coords transformed with the morphology; edge assignment kept |
| **reduce (inplace, mapped)** | Transfer section mapping via cable provenance |
| **reduce (inplace, unmapped)** | Raw preserved; still unmapped |
| reroot (inplace) | Raw kept; **mapping invalidated** |
| `get_subtree(synapses="mapped_only")` | Default: keep `mapped=True` on retained edges; clear if never mapped |
| `get_subtree(synapses="all")` | Keep synapses on retained edges (incl. `mapped=False`); unmapped table kept, mapping cleared |
| `get_subtree(synapses="none")` | Drop all synapses |
| `copy` | Deep-copies the synapse table |
