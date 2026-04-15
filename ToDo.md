# To Do

## **Implement stricter Design rules!**

```
┌─────────────────────────────────────────────────────────┐
│  gui/          layer 6                                  │
│  widgets, event handlers, visualisation                 │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  analysis/     layer 5                                  │
│  top-level functions over user objects                  │
└───────────────────────┬─────────────────────────────────┘
                        |
┌───────────────────────▼─────────────────────────────────┐
│  io/           layer 4                                  │
│  User exposed io layer                                  │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  api/          layer 3                                  │
│  user-facing objects, methods call ops/                 │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐  - - - - - - - - - - -
│  ops/          layer 2                                  │ >  utils/  (parallel) :
│  functions applying utils/ to core/ classes             │  :                    :
└───────────────────────┬─────────────────────────────────┘  : numpy_utils.py     :
                        │                                    : graph_utils.py     :
┌───────────────────────▼─────────────────────────────────┐  : other_utils.py     :
│  core/         layer 1                                  │  - - - - - - - - - - -
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  stone.py   │  │  tree.py    │  │  forest.py  │      │
│  │  core base  │  │  single-    │  │  collection │      │
│  │  class      │  │  instance   │  │  multi-inst │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────┘
```

Going forward, use this structure
- [x] Need to refactor old (Specifically, `tree_graphs/`)
- [x] `graph_utils` needs to be its own contained graph_tool toolbox.
- [x] same with `numpy_utils`
- [x] `ops/` integrates `core/` and `utils/`
- [ ] Fix the annoying GUI bug which generates a random coulour every FUCKING time a neuron is reloaded. (default kwargs to `renderer.NeuronRenderer.render_neuron`)

## Geneal Fixes

- [x] Streamline paallelisation by merging `forrest.apply`, `forest.for_each` and `foest.apply_fn` into one, handling binding.   
- [ ] Remove 'metadata' as core attribute and update function to always call _Tree.graph.gp['metadata]. Make sure metadata updating works frorm the graph directly (ensurers consistent behaviour when updating metadata) 
- [ ] move `_Forest` subsetting and co to `Forest`, otherwise we return the sub class instance, not the user exposed one.
- [ ] Not currently happy with how radii are handled with reduced trees, but not currently important.
- [ ] Re-work aspect and scaling of dendrogram layouts.


## Functionality

### Core Structure

 - [x] Neuron class
 - [x] Forest
 - [ ] Mesh Class
 - [x] Threaded Parallelisation
 - [ ] Subsetting by metadata values

### I/O

 - [x] Read / Write swc
 - [x] Read / Write nr

### Basic Descriptives

#### Counting

 - [x] Count Nodes
 - [x] Count Edges
 - [x] Count Roots
 - [x] Count Leaves
 - [x] Count Branches
 - [x] Count Transitive Nodes

#### Indicies

 - [x] Root ind
 - [x] Leaf inds
 - [x] Branch inds
 - [x] Core (root, branch, leaf) inds
 - [x] Edge inds (vertex pairs)

#### Coordinates

 - [x] Vertex coordinates
 - [x] Edge coordinate (vertex pairs)

#### Lin. Alg.

 - [x] Pairwise (Edge) Distances
 - [ ] Distance matricies
 - [ ] Rotations
 - [ ] Alignment
 - [ ] Bifurcation Geometry

### Graph Theory

 - [x] Depth First Search
 - [x] Breadth First Search
 - [x] Post-order traversal node order

### Plotting

 - [x] Single 2D plot
 - [x] Single 3D plot
 - [x] Viewer (for multiple)
 - [x] Dendrograms

### GUI

 - [x] Load nr
 - [x] Load swc
 - [x] GUI core port
 - [ ] Point selection (works, but has not functionality to call yet)

### Tree Surgery

 - [x] Neuron reduction
 - [ ] reooting
 - [ ] Subtree masking
 - [ ] Subtree identification
 - [ ] Subtree isolation

### Topology

 - [x] Node Depth

### Paper Reproduction

...

-----





