# Publication-readiness audit

Status after the validation pass on 6 Nov 2026. This is a record of what was
checked in the source tree, not a claim that every metric has an external
proof.

Severity:

- **Critical** — wrong numbers, lost data, or a result that cannot be reproduced.
- **High** — a scientific contract that was false, or a validation gap on a
  result people will publish.
- **Medium** — maintainability, docs, or install friction.
- **Low** — tidy-up.

A missing test is not, by itself, evidence that the code is wrong.

## Critical

### Edge index was not `Graph.get_edges()` row order

- **Module:** `utils/graph_utils/vertex_inds.py`, `traversals.py` (`ReduceVisitor`),
  `ops/tree_graphs/tree_synapses.py`, `utils/graph_utils/graph_editing.py`.
- **API:** `tree.map_synapses`, `tree.get_synapse_path_distance`,
  `tree.get_reduced_tree`, `tree.get_edge_indices`, `reroot_graph`.
- **Problem:** Graphs built with `Graph(..., hashed=True)` — SWC import,
  reduction, rerooting — do not iterate edges in edge-index order.
  `PropertyMap.a` is indexed by edge index. Synapse `edge_index` is an edge
  index. Code that did `get_edges()[edge_index]` read the wrong endpoints, so
  root cable distance changed after reduction.
- **How found:** The synapses tutorial still printed `path distance invariant:
  False` (run of 29 Sep 2026). Replayed on 6 Nov 2026 with the same seed on
  example neuron `720575940596125868`: 40 synapses, max absolute error 8.07 µm
  before the fix, `8.5e-14` after. A 6-node tree with three reduced edges
  fails the same way. A single reduced edge does not, which is why the older
  straight-chain test stayed green.
- **Science:** Root cable distance of a mapped synapse is part of the
  reduction contract. Euclidean chords are not. The bug reported a different
  cable without changing coordinates.
- **Fix:** `edge_indices(root=None)` returns rows in edge-index order.
  `ReduceVisitor` records `g.edge_index`. Reroot writes migrated lengths at
  `g.edge_index`, not at the `get_edges()` row. Reduced `Path_length` is
  stored as `double`.
- **Tests:** `test_reduced_path_distance_when_edge_order_is_permuted`,
  `test_reroot_keeps_length_on_the_same_segment`.
- **Status:** Fixed.

### `get_root_distance(bind=False)` and `get_vertex_degrees` returned a dead view

- **Module:** `ops/tree_graphs/tree_path_lengths.py`, `utils/graph_utils/degrees.py`.
- **API:** `tree.get_root_distance(bind=False)`, `tree.get_degrees()`.
- **Problem:** graph-tool `PropertyMap.a` is a view. Both functions returned
  `.a` of a map that is not stored on the graph. After the call returns, the
  memory can be reused. A probe on a 6-node tree showed non-finite garbage in
  the first entries while later entries still looked plausible.
- **Science:** Any script that stores that array and reads it later can
  publish nonsense distances or degrees. Sums taken in the same statement
  often still work, so tests that only check a total can miss it.
- **Fix:** Return `numpy.array(..., copy=True)`.
- **Tests:** `test_get_root_distance_bind_false` overwrites the first array
  and checks that a second call is still zero at the root.
- **Status:** Fixed.

## High

### `get_edge_length` docstring described the wrong reduced length

- **Module:** `ops/tree_graphs/tree_path_lengths.py`, metric registry.
- **API:** `tree.get_edge_length`.
- **Problem:** The docstring said a reduced tree returns the Euclidean
  distance between surviving nodes. The code returns stored `Path_length`
  when that property exists. Reduction writes the accumulated cable there.
- **How found:** Reading the function against `reduce_graph`, then measuring
  the irregular reference polyline: cable 3.5, chord `sqrt(5.25)`.
- **Science:** Treating that array as a chord underestimates curved cable.
  `recalculate=True` really does return the chord. That distinction is now
  in the docstring and in `docs/reference/metrics.md`.
- **Tests:** `test_irregular_curve_reduction_keeps_cable_not_the_chord`.
- **Status:** Documentation fixed. Behaviour kept.

### Tutorial reused `cleft_id` and hid a reciprocal connection

- **Module:** `ops/forest_ops/forest_connectivity.py` (behaviour was already
  what the docstring says). `docs/tutorials/synapses.ipynb` was wrong.
- **API:** `forest.get_connectivity_table(deduplicate="auto")`.
- **Problem:** The notebook set `cleft_id` to `np.arange(n)` on both neurons.
  Auto dedup treats `cleft_id` as one physical synapse for the whole forest,
  so the three reverse synapses were dropped. Replayed: one directed edge of
  weight 5. With ids `f"{tree.ID}-{i}"`: weights 5 and 3, two edges.
- **Science:** This is the contract, not a bug, if `cleft_id` is a global id.
  It is a bug in the example. Reciprocal synapses need distinct ids.
  `deduplicate="none"` counts every row.
- **Tests:** `test_reused_integer_cleft_ids_are_not_two_synapses`,
  `test_deduplicate_shared_cleft`.
- **Status:** Tutorial and test updated. Algorithm unchanged.

### Reference morphologies

Hand-checked trees live in `neurosetta.testing.reference_morphologies` and
`tests/test_validation/test_reference_morphologies.py`. They cover node and
edge counts, sections, Strahler, cable, root distance, subtree cable, Sholl
path crossings, partition asymmetry, a right-angle bifurcation, a
multifurcation that is not a bifurcation, a zero-length edge, reduction
cable versus chord, tetrahedron volume 1/6, NaN coordinates, a cycle, and
two roots.

They do not cover every registered metric. Absence of a case is a gap, not
a failed metric.

### Transformation metadata

`tests/test_utils/test_metrics/test_invariance.py` already checks declared
translation, rotation, and scale invariance, with explicit exceptions.
Structured outputs (PCA eigenvectors, hull vertices) are not given a single
Boolean rotation flag. That limitation stays.

### `check_reduced` is not `metadata["isReduced"]`

- **Module:** `ops/tree_graphs/tree_checks.py`, `get_edge_length`.
- **API:** `tree.is_reduced()` / `check_reduced`.
- **Observation:** `check_reduced` is true when the graph has no degree-2
  vertices. A tree drawn with one sample per branch is "reduced" by that
  test even if `reduce_graph` never ran and `metadata["isReduced"]` is false.
  `get_edge_length` then stores a newly computed length as
  `Euclidean_length` rather than `Path_length`.
- **Science:** Cable modes (`partition_asymmetry` weighted/cable) refuse to
  run without `Path_length`. That is the right refusal. Calling the
  topological predicate `is_reduced` next to the metadata flag is easy to
  mix up. Not changed in this pass: renaming it is a public API break.
- **Status:** Documented. Not a numeric bug on trees that still have their
  original edges, because the endpoint distance is the cable.

## Medium

### Layer imports

`utils.metrics.registry` imported `ops._doc_helpers.TREE_METHOD_ALIASES`.
The table now lives in `neurosetta.api_names`. `ops._doc_helpers` re-exports
it, so docstring linking is unchanged.

Still true, and allowlisted in `tests/test_architecture/test_layer_imports.py`:

- `utils/metrics/descriptors.py` imports `api` and `ops` (`describe` calls
  tree methods).
- `core/tree.py`, `core/tree_helpers.py`, and `core/synapses.py` import `ops`
  lazily for synapses, meshes, and units.
- `ops` imports `api` mesh classes inside a few functions.

New imports in the forbidden direction fail that test.

### `.history/`

About 11 MB of editor-local history was tracked. It is not scientific data.
`.gitignore` ignores `.history/`. The index removal (`git rm -r --cached
.history`) is staged and not committed. The files are still on disk.

### `.nr` and graph-tool objects

`graph-tool` `.gt` / `.nr` files can store pickled Python objects in graph
properties. That is a trusted-file format. Workspace archives reject path
traversal, bad schema versions, and checksum mismatches
(`tests/test_io_utils/test_workspace.py`) and do not unpickle NumPy object
arrays. Do not load `.nr` files from untrusted sources.

### Install

graph-tool is not on PyPI. The docs say so. PySide6 is still a hard
dependency because `neurosetta.gui` is imported from the package root.
Headless metric work does not need the GUI; making that import lazy is
still open.

## Docs build, 6 Nov 2026

`python -m sphinx -b html docs /tmp/nr-docs` finishes. InterSphinx inventories
were not fetched (proxy 403). Two remaining docutils errors, both
`Unknown target name: "global"` on `Forest.get_partition_asymmetry`. The
same `global_` parameter is injected into every forest method; this
docstring is the one Sphinx still rejects. Duplicate `Synapses` property
descriptions are warnings. The `|r - s|` substitutions in that docstring
were the other errors; they are now written as ``abs(r - s)``.

## Low

Reduced `Path_length` was `float` (float32). It is now `double`. Totals
were already exact for the short fixtures; long cables no longer pay
float32 rounding on the stored section length.

## Tolerances used in the new tests

| Comparison | Tolerance | Why |
| --- | --- | --- |
| Counts, Strahler, Sholl crossing counts | Exact | Integers |
| Lengths that are integers or `n/2` | `atol=1e-12` | Exact in float64 at these magnitudes |
| Axis-aligned angles | `atol=1e-12` rad | Multiples of `π/2` |
| Convex hull volume `1/6` | `rtol=1e-9`, `atol=1e-12` | Qhull on a tetrahedron |
| Synapse cable through reduction | `atol=1e-9` in the unit test; example neuron was `1e-13` | Double sums along a real arbor |

Numba projection (`fastmath=True`) is not what failed in the tutorial. The
failure was edge identity, which is exact once the index is the edge index.
