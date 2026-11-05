# Sholl analysis

Sholl analysis counts intersections with distance shells. It is not a
histogram of nodes.

```python
import neurosetta as nr

tree = nr.load_example_data(nr.example_ids[0])
radii = nr.get_sholl_profile(tree, [10, 20, 40, 80], mode="path_length")
print(radii.intersections, nr.get_sholl_peak_radius(radii))
```

## Three distances

`radial`
: Straight-line distance from a center (default: the root coordinate) to a
  point on a segment. Needs `x`, `y`, `z`. An explicit center is allowed only
  in this mode.

`path_length`
: Sum of `Path_length` from the root along the directed path. This is
  physical cable. It is not rebuilt from coordinates on a reduced tree.

`euclidean_length`
: Sum of `Euclidean_length` along that same path. This is **not** radial
  distance. Each edge adds its own straight-line length, so a path that folds
  back toward the root still increases.

## How a crossing is counted

Cumulative modes use a half-open rule on the accumulated distances of an
edge's endpoints: `d(source) < radius <= d(target)`. A node that sits on a
shell is counted on the incoming edge and not again on the outgoing edges.
Radius 0 is not crossed by an edge that starts at the root. One long edge can
cross many radii.

Radial mode solves the segment–sphere quadratic. An edge with both endpoints
outside a sphere can pass through it and contribute two intersections.
Tangencies are not counted. A zero-length segment is not a crossing. The same
half-open parameter rule (`0 < t <= 1`, within tolerance) attributes a vertex
on a shell to the incoming edge only.

Peak radius is the **smallest** radius that attains the maximum intersection
count. Area is the trapezoidal integral, so the radii do not need to be
uniform. Fewer than two radii have area 0.

## Reduced trees

Radial Sholl uses the straight segments of the graph you pass in. After
reduction those segments are chords. They do not reproduce the original
curve, so radial Sholl on a reduced tree can disagree with the unreduced
reconstruction even when `Path_length` was preserved.

`ReductionMap` records which original edges were collapsed and their cable
lengths. It does not store the original vertex coordinates, so this
implementation does not rebuild the detailed geometry.

`path_length` mode still uses the preserved `Path_length`.
`euclidean_length` mode uses `Euclidean_length` (the chords, if that is what
was stored).

```{eval-rst}
.. automodule:: neurosetta.ops.tree_graphs.tree_sholl
   :members:
```
