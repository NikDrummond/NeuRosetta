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
                        |                                    : vedo_utils.py(todo):
┌───────────────────────▼─────────────────────────────────┐  : other_utils.py     :
│  core/         layer 1                                  │  - - - - - - - - - - -
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │  stone.py   │  │  tree.py    │  │  forest.py  │      │
│  │  core base  │  │  single-    │  │  collection │      │
│  │  class      │  │  instance   │  │  multi-inst │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└─────────────────────────────────────────────────────────┘
```

## Features to add
- [ ] Global settings / defaults (Pydantic?)
- [ ] Units handling (Pint/ounce?)
- [x] switch to slotted class definitions. should give slight memory improvement at the cost of the user being able to set arbritary atributes to trees, but we have metadata and graph properties. (possibly part of pydantic integration)
- [ ] Review / optimise imports


## Geneal Fixes

- [x] Remove 'metadata' as core attribute and update function to always call _Tree.graph.gp['metadata]. Make sure metadata updating works frorm the graph directly (ensurers consistent behaviour when updating metadata) 
- [ ] Not currently happy with how radii are handled with reduced trees, but not currently important.
- [ ] Re-work aspect and scaling of dendrogram layouts.
- [ ] converting to a subtree breaks plotting unless saved and reloaded. Plotting needs a fix anyway.
- [ ] GUI doesn't track edges properly from `.swc` files.
- [ ] vedo is not threadsafe and gt is no multiprocessing safe, so currently we cannot easily parallelise plotting of forests. I should separate out the threadsafe logic (getting coordinates etc) and multithread that.
- [ ] Use 'get', 'set' language consistently?
- [ ] Using 3 1-d arrays instead of one 2d is ~60x faster to access coordinates when implemented, should do.
- [ ] Numba and refine numpy backend?


## Functionality

### Core Structure

 - [x] Neuron class
 - [x] Forest
 - [x] Mesh Class (`_Forest` container can be re-used)
 - [x] Threaded Parallelisation
 - [x] Subsetting by metadata values

### I/O

 - [x] Read / Write swc
 - [x] Read / Write nr
 - [x] Read / Write meshes (use vedo)

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
 - [x] Mesh loading - use thin wrapper around vedo as patch fix
 - [x] Point selection

### Tree Surgery

 - [x] Neuron reduction
 - [x] reooting
 - [x] Subtree masking
 - [x] Subtree identification
 - [x] Subtree isolation

### Topology

 - [x] Node Depth

### Meshes

    - [ ] mesh method to set colour/alpha etc rather than setting through mesh.mesh





