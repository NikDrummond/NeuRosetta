# Scientific validation summary

Automated checks are the tests named below. They were run in the `nr`
environment on 6 Nov 2026. A green test is not a proof for metrics the test
does not call.

## What was found, and the independent check

The synapses tutorial reported `path distance invariant: False`. Replaying
the same 40-synapse draw (seed 0) on example neuron `720575940596125868`
gave a maximum cable error of 8.07 µm after reduction. The cause was not
the section-length sum. `Graph(..., hashed=True)` does not iterate edges in
edge-index order, and synapse code indexed `get_edges()` with an edge index.
After the index fix the same draw differs by `8.5e-14` µm. A 6-node tree
with three reduced edges is the regression test
`test_reduced_path_distance_when_edge_order_is_permuted`.

`get_root_distance(bind=False)` returned a graph-tool array view of a map
that is not stored on the graph. Reading it later can return non-finite
garbage. It now returns an owned NumPy copy. Same defect in
`get_vertex_degrees`.

## Reference morphologies

Defined in `src/neurosetta/testing/reference_morphologies.py`. Expected values
are from the coordinates and the branching rules, not from the function
being tested.

| Morphology | What is checked |
| --- | --- |
| Single node | 1 node, 0 edges, cable 0, Strahler 1 |
| Straight chain, spacing 1 | Cable 3, root distance 0,1,2,3, subtree cable 3,2,1,0, Sholl path crossings |
| Nonuniform chain, lengths 1, 2, 4 | Root distance 0, 1, 3, 7 |
| Right-angle bifurcation | One bifurcation, child–child angle π/2 |
| Symmetric binary, six unit edges | Strahler root 3, cable 6, 3 branches, 2 non-root bifurcations |
| Asymmetric branch | Strahler `[2,1,2,1,1]`, unweighted asymmetry 0.5 at the root, van Pelt 1, subtree cable |
| Trifurcation | 1 branch, 0 bifurcations (out-degree 3 is not a bifurcation), hull volume 1/6 |
| Zero-length edge | Two edges, cable 1 |
| Irregular polyline, cable 3.5 | Reduction keeps 3.5; the chord is `sqrt(5.25)` |
| Cycle, two roots | `validate_rooted_tree` raises |
| NaN coordinate | Total length is NaN, not a finite substitute |

Test module: `tests/test_validation/test_reference_morphologies.py`.

## Tolerances

Counts and Strahler orders are exact. Lengths that are integers or halves
use `atol=1e-12`. The tetrahedron volume uses `rtol=1e-9`. Synapse cable
through reduction uses `atol=1e-9` on the synthetic tree.

## Invariance already covered elsewhere

`tests/test_utils/test_metrics/test_invariance.py` checks the registry flags
for translation, rotation, and uniform scale, and checks that cable scales
by the factor and hull volume by the cube. PCA eigenvectors are not compared
as raw arrays.

## Reduction contract

Preserved when `Path_length` exists before reduction:

- Branching of branch points and leaves (degree-2 vertices are removed).
- Sum of `Path_length` (section cable, not the new chord).
- Mapped synapse identity, type, partner, and root cable distance.
- `edge_index` after transfer is the reduced graph's edge index.

Not preserved:

- Coordinates of deleted samples.
- Euclidean length of a collapsed edge.
- Radial Sholl, which uses the stored chords.
- Synapse mapping after reroot (mapping is invalidated).

`check_reduced()` means "no degree-2 vertices". `metadata["isReduced"]` means
`reduce_graph` ran. A branch-only reconstruction can pass the first and not
the second.

## Connectivity

`cleft_id` under `deduplicate="auto"` is one id for the whole forest.
The same id on a pre and a post record is one synapse. Two biological
synapses need two ids. Test:
`test_reused_integer_cleft_ids_are_not_two_synapses`.

## Not validated here

- Every registered metric against an analytical value.
- NAVis or NeuroM numerical agreement (those packages are not installed in
  the `nr` environment used for this pass).
- A timed comparison that could support a speed claim.
- GUI behaviour.
