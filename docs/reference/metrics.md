# Metrics overview

NeuRosetta exposes **descriptive metrics** — scalar summaries and per-element
measurements of topology, geometry, and spatial extent. They live in
{doc}`../api/tree_ops/index` (tree/forest ops) and {doc}`../api/neuropils`
(neuropil distance helpers).

This page is a catalog of what exists today. Detailed signatures and docstrings
are in the linked API pages (autodoc, always current).

## Entry points

| You have… | Prefer… |
| --- | --- |
| One tree, notebooks | {class}`~neurosetta.api.Tree` methods — e.g. ``tree.count_nodes()`` |
| Many trees | {class}`~neurosetta.api.Forest` batch methods — e.g.
  ``forest.get_total_cable_length()`` |
| Pipelines / scripts | Module functions — e.g. ``nr.count_nodes(tree)`` |

``Tree.some_method()`` and ``nr.some_method(tree)`` are interchangeable. See
{doc}`../getting_started/overview`.

## Notes

- **Reduced trees:** cable-length metrics use bound edge properties; compute
  ``Path_length`` before reducing if you need lengths along the full arbor.
  See {doc}`../tutorials/tree_surgery`.
- **Units:** length and coordinate metrics require defined spatial units.
  See {doc}`units`.
- **Not listed here:** node/edge index lookups, traversals, editing, coordinate
  transforms, and plotting — those are operations, not descriptives.

<!-- AUTO-GENERATED: table below; run ``python docs/_generate_reference.py`` -->

## Quick reference

| Category | Metric | Tree | Forest | API | Notes |
| --- | --- | --- | --- | --- | --- |
| Summary | ``forest_summary`` | — | ``forest.summary()`` | {doc}`../api/tree_ops/tree_summary` | Per-tree summary with optional TOTAL/MEAN rows. Composite: topology counts + cable length. |
| Summary | ``forest_summary_table`` | — | ``forest.summary_table()`` | {doc}`../api/tree_ops/tree_summary` | Per-tree numeric summary DataFrame. Composite: topology counts + cable length. |
| Summary | ``summary_table`` | ``tree.summary_table()`` | — | {doc}`../api/tree_ops/tree_summary` | One-row numeric summary DataFrame. Composite: topology counts + cable length. |
| Summary | ``tree_summary`` | ``tree.summary()`` | — | {doc}`../api/tree_ops/tree_summary` | Formatted summary table (HTML in notebooks). Composite: topology counts + cable length. |
| Counting | ``count_bifurcations`` | ``tree.count_bifurcations()`` | ``forest.count_bifurcations()`` | {doc}`../api/tree_ops/counting` |  |
| Counting | ``count_branches`` | ``tree.count_branches()`` | ``forest.count_branches()`` | {doc}`../api/tree_ops/counting` |  |
| Counting | ``count_core_nodes`` | ``tree.count_core_nodes()`` | ``forest.count_core_nodes()`` | {doc}`../api/tree_ops/counting` |  |
| Counting | ``count_edges`` | ``tree.count_edges()`` | ``forest.count_edges()`` | {doc}`../api/tree_ops/counting` |  |
| Counting | ``count_leaves`` | ``tree.count_leaves()`` | ``forest.count_leaves()`` | {doc}`../api/tree_ops/counting` |  |
| Counting | ``count_nodes`` | ``tree.count_nodes()`` | ``forest.count_nodes()`` | {doc}`../api/tree_ops/counting` |  |
| Counting | ``count_roots`` | ``tree.count_roots()`` | ``forest.count_roots()`` | {doc}`../api/tree_ops/counting` |  |
| Counting | ``count_sections`` | ``tree.count_sections()`` | ``forest.count_sections()`` | {doc}`../api/tree_ops/counting` |  |
| Counting | ``count_transitive_nodes`` | ``tree.count_transitive_nodes()`` | ``forest.count_transitive_nodes()`` | {doc}`../api/tree_ops/counting` |  |
| Structure | ``get_binary_ratio`` | ``tree.get_binary_ratio()`` | ``forest.get_binary_ratio()`` | {doc}`../api/tree_ops/tree_structure` |  |
| Structure | ``get_max_depth`` | ``tree.get_max_depth()`` | ``forest.get_max_depth()`` | {doc}`../api/tree_ops/tree_structure` |  |
| Structure | ``get_max_width`` | ``tree.get_max_width()`` | ``forest.get_max_width()`` | {doc}`../api/tree_ops/tree_structure` |  |
| Structure | ``get_mean_depth`` | ``tree.get_mean_depth()`` | ``forest.get_mean_depth()`` | {doc}`../api/tree_ops/tree_structure` |  |
| Structure | ``get_mean_width`` | ``tree.get_mean_width()`` | ``forest.get_mean_width()`` | {doc}`../api/tree_ops/tree_structure` |  |
| Structure | ``get_median_depth`` | ``tree.get_median_depth()`` | ``forest.get_median_depth()`` | {doc}`../api/tree_ops/tree_structure` |  |
| Structure | ``get_median_width`` | ``tree.get_median_width()`` | ``forest.get_median_width()`` | {doc}`../api/tree_ops/tree_structure` |  |
| Structure | ``get_node_depth`` | ``tree.get_node_depth()`` | ``forest.get_node_depth()`` | {doc}`../api/tree_ops/tree_structure` | Per-node depth from root. |
| Structure | ``get_tree_widths`` | ``tree.get_tree_widths()`` | ``forest.get_tree_widths()`` | {doc}`../api/tree_ops/tree_structure` | Node count at each depth. |
| Path lengths | ``get_edge_length`` | ``tree.get_edge_length()`` | ``forest.get_edge_length()`` | {doc}`../api/tree_ops/path_lengths` | Euclidean length per edge; cable semantics depend on reduction state. |
| Path lengths | ``get_total_cable_length`` | ``tree.get_total_cable_length()`` | ``forest.get_total_cable_length()`` | {doc}`../api/tree_ops/path_lengths` |  |
| Degrees | ``get_degree_distribution`` | ``tree.get_degree_distribution()`` | ``forest.get_degree_distribution()`` | {doc}`../api/tree_ops/degrees` |  |
| Degrees | ``get_degrees`` | ``tree.get_degrees()`` | ``forest.get_degrees()`` | {doc}`../api/tree_ops/degrees` | In/out/total degree per node. |
| Geometry | ``get_bifurcation_angle_sums`` | ``tree.get_bifurcation_angle_sums()`` | ``forest.get_bifurcation_angle_sums()`` | {doc}`../api/tree_ops/tree_geometry` |  |
| Geometry | ``get_bifurcation_angles`` | ``tree.get_bifurcation_angles()`` | ``forest.get_bifurcation_angles()`` | {doc}`../api/tree_ops/tree_geometry` |  |
| Geometry | ``get_bifurcation_deihedral_beta`` | ``tree.get_bifurcation_deihedral_beta()`` | ``forest.get_bifurcation_deihedral_beta()`` | {doc}`../api/tree_ops/tree_geometry` |  |
| Geometry | ``get_edge_angle_variance`` | ``tree.get_edge_angle_variance()`` | ``forest.get_edge_angle_variance()`` | {doc}`../api/tree_ops/tree_geometry` | Circular variance of get_edge_angles; same reference-frame dependence. |
| Geometry | ``get_edge_angles`` | ``tree.get_edge_angles()`` | ``forest.get_edge_angles()`` | {doc}`../api/tree_ops/tree_geometry` | Angles relative to caller-supplied between_vector. |
| Geometry | ``get_mean_edge_angle`` | ``tree.get_mean_edge_angle()`` | ``forest.get_mean_edge_angle()`` | {doc}`../api/tree_ops/tree_geometry` | Circular mean of get_edge_angles; same reference-frame dependence. |
| Geometry | ``get_radial_angle`` | ``tree.get_radial_angle()`` | ``forest.get_radial_angle()`` | {doc}`../api/tree_ops/tree_geometry` | Unsigned mode is intrinsic; signed mode needs alignment_vector and loses rotation invariance relative to that axis. |
| Geometry | ``get_section_angular_deviation`` | ``tree.get_section_angular_deviation()`` | — | {doc}`../api/tree_ops/tree_geometry` | Requires a non-reduced tree. |
| Coordinates | ``coordinate_pca`` | ``tree.coordinate_pca()`` | ``forest.coordinate_pca()`` | {doc}`../api/tree_ops/coordinates` | Returns (evals, evecs); evals can be shape-descriptive, evecs are lab-frame orientations. |
| Coordinates | ``get_convex_hull`` | ``tree.get_convex_hull()`` | ``forest.get_convex_hull()`` | {doc}`../api/tree_ops/coordinates` | Returns ConvexHull with absolute vertex coordinates. |
| Coordinates | ``get_convex_hull_volume`` | ``tree.get_convex_hull_volume()`` | ``forest.get_convex_hull_volume()`` | {doc}`../api/tree_ops/coordinates` |  |
| Coordinate moments | ``coordinate_extent_along_axis`` | ``tree.coordinate_extent_along_axis()`` | ``forest.coordinate_extent_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` | pmax - pmin along caller-supplied axis. |
| Coordinate moments | ``coordinate_mean_absolute_along_axis`` | ``tree.coordinate_mean_absolute_along_axis()`` | ``forest.coordinate_mean_absolute_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` | Mean \|projection\| onto caller-supplied axis. |
| Coordinate moments | ``coordinate_mean_along_axis`` | ``tree.coordinate_mean_along_axis()`` | ``forest.coordinate_mean_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` | Mean of absolute projections onto caller-supplied axis. |
| Coordinate moments | ``coordinate_minmax_along_axis`` | ``tree.coordinate_minmax_along_axis()`` | ``forest.coordinate_minmax_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` | Min/max absolute projections onto caller-supplied axis. |
| Coordinate moments | ``coordinate_projection_moments`` | ``tree.coordinate_projection_moments()`` | ``forest.coordinate_projection_moments()`` | {doc}`../api/tree_ops/coordinate_moments` | Compound mean/var/std/minmax along caller-supplied axis. |
| Coordinate moments | ``coordinate_rms_along_axis`` | ``tree.coordinate_rms_along_axis()`` | ``forest.coordinate_rms_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` | RMS of absolute projections onto caller-supplied axis. |
| Coordinate moments | ``coordinate_std_along_axis`` | ``tree.coordinate_std_along_axis()`` | ``forest.coordinate_std_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` | Std of projections onto caller-supplied axis. |
| Coordinate moments | ``coordinate_variance_along_axis`` | ``tree.coordinate_variance_along_axis()`` | ``forest.coordinate_variance_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` | Variance of projections onto caller-supplied axis. |
| Subtrees | ``get_max_subtree_node`` | ``tree.get_max_subtree_node()`` | ``forest.get_max_subtree_node()`` | {doc}`../api/tree_ops/subtrees` |  |
| Subtrees | ``get_partition_asymmetry`` | ``tree.get_partition_asymmetry()`` | ``forest.get_partition_asymmetry()`` | {doc}`../api/tree_ops/subtrees` | Default weighted=True uses Path_length cable fractions; unweighted mode is topological. |
| Subtrees | ``get_subtree_scores`` | ``tree.get_subtree_scores()`` | ``forest.get_subtree_scores()`` | {doc}`../api/tree_ops/subtrees` |  |
| Shape fitting | ``fit_circle`` | ``tree.fit_circle()`` | ``forest.fit_circle()`` | {doc}`../api/tree_ops/shape_fitting` | Returns center/radius/normal; mixed invariance. |
| Shape fitting | ``fit_line`` | ``tree.fit_line()`` | ``forest.fit_line()`` | {doc}`../api/tree_ops/shape_fitting` | Returns slope/center/variances; mixed invariance. |
| Shape fitting | ``fit_plane`` | ``tree.fit_plane()`` | ``forest.fit_plane()`` | {doc}`../api/tree_ops/shape_fitting` | Returns normal/center/variances; mixed invariance. |
| Shape fitting | ``fit_sphere`` | ``tree.fit_sphere()`` | ``forest.fit_sphere()`` | {doc}`../api/tree_ops/shape_fitting` | Returns center+radius (or vedo object); mixed invariance. |
| Neuropil distances | ``distance_from_neuropil_surface`` | ``nr.distance_from_neuropil_surface()`` | — | {doc}`../api/neuropils` | Standalone function; requires a neuropil mesh. |
| Neuropil distances | ``neuropil_point_depth`` | ``nr.neuropil_point_depth()`` | — | {doc}`../api/neuropils` | Standalone function; requires a neuropil mesh. Invariance is neuron-vs-fixed-mesh (not joint transforms). |

## Morphology descriptors

Morphology measurements in NeuRosetta can be grouped into:

1. **topology** — how a neuron branches (connectivity);
2. **intrinsic geometry** — the shape of the arbor independent of anatomical placement;
3. **embedding** — how that arbor is situated within an anatomical context.

An :class:`~neurosetta.AnatomicalFrame` provides that external context
(reference mesh, optional surface pair, named axes). It does **not** replace
existing Neuropil/mesh classes or low-level distance/depth functions.

:func:`neurosetta.describe` builds a standardized table from the metric
registry. Array-valued registry metrics are turned into explicitly named
summary descriptors (mean / std / quartiles).

```python
import neurosetta as nr

df = nr.describe(tree)                 # wide, one row
df = nr.describe(forest)               # wide, one row per tree
df = nr.describe(forest, domains="topology")
df = nr.describe(forest, output="long")

frame = nr.AnatomicalFrame(
    name="example_neuropil",
    reference_mesh=neuropil,
    axes={"depth": (0, 0, 1)},
)
embedding = nr.describe(
    forest,
    domains="embedding",
    reference_frame=frame,
)
# equivalent: frame.describe(forest)
```

Low-level mesh/axis arguments remain valid::

    nr.distance_from_neuropil_surface(neuropil, points)
    nr.neuropil_point_depth(neuropil, points, t=0.0)
    tree.coordinate_extent_along_axis((0, 0, 1))

Default ``describe`` selection is topology + intrinsic geometry. Embedding
requires a usable ``reference_frame``. When both ``domains`` and ``metrics``
are given, the selection is their intersection.

## Scientific metadata

Each catalog entry carries morphology metadata independent of ``category``.

| Field | Meaning |
| --- | --- |
| ``domain`` | Scientific morphology domain |
| ``level`` | Natural level of the returned measurement |
| ``translation_invariant`` | Unchanged by global translation |
| ``rotation_invariant`` | Unchanged by global rotation |
| ``scale_invariant`` | Numerically unchanged by uniform scaling |
| ``requires_coordinates`` | Requires spatial coordinates |
| ``requires_reference_frame`` | Requires external anatomical/spatial reference |

### Domains

| Domain | Meaning |
| --- | --- |
| ``topology`` | Connectivity independent of coordinates |
| ``intrinsic_geometry`` | Spatial properties intrinsic to the neuron and independent of its anatomical placement |
| ``embedding`` | Properties describing the neuron's relationship to an external spatial or anatomical frame |

Invariance metadata describes transforming the neuron while keeping any
external reference frame fixed.

Filter with ``nr.list_metric_definitions(domain=...)`` and/or ``level=...``.

Use :func:`neurosetta.describe` for a standardized per-neuron descriptor table
over topology and intrinsic geometry (and embedding when a reference frame is
supplied).

### Classification review

Generated from the registry (source of truth). Columns ``T`` / ``R`` / ``S``
are translation / rotation / scale invariance; ``Coords`` / ``Ref`` are
``requires_coordinates`` / ``requires_reference_frame``.

| Metric | Category | Domain | Level | T | R | S | Coords | Ref |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ``forest_summary`` | Summary | ``intrinsic_geometry`` | ``population`` | True | True | False | True | False |
| ``forest_summary_table`` | Summary | ``intrinsic_geometry`` | ``population`` | True | True | False | True | False |
| ``summary_table`` | Summary | ``intrinsic_geometry`` | ``tree`` | True | True | False | True | False |
| ``tree_summary`` | Summary | ``intrinsic_geometry`` | ``tree`` | True | True | False | True | False |
| ``count_bifurcations`` | Counting | ``topology`` | ``tree`` | True | True | True | False | False |
| ``count_branches`` | Counting | ``topology`` | ``tree`` | True | True | True | False | False |
| ``count_core_nodes`` | Counting | ``topology`` | ``tree`` | True | True | True | False | False |
| ``count_edges`` | Counting | ``topology`` | ``tree`` | True | True | True | False | False |
| ``count_leaves`` | Counting | ``topology`` | ``tree`` | True | True | True | False | False |
| ``count_nodes`` | Counting | ``topology`` | ``tree`` | True | True | True | False | False |
| ``count_roots`` | Counting | ``topology`` | ``tree`` | True | True | True | False | False |
| ``count_sections`` | Counting | ``topology`` | ``tree`` | True | True | True | False | False |
| ``count_transitive_nodes`` | Counting | ``topology`` | ``tree`` | True | True | True | False | False |
| ``get_binary_ratio`` | Structure | ``topology`` | ``tree`` | True | True | True | False | False |
| ``get_max_depth`` | Structure | ``topology`` | ``tree`` | True | True | True | False | False |
| ``get_max_width`` | Structure | ``topology`` | ``tree`` | True | True | True | False | False |
| ``get_mean_depth`` | Structure | ``topology`` | ``tree`` | True | True | True | False | False |
| ``get_mean_width`` | Structure | ``topology`` | ``tree`` | True | True | True | False | False |
| ``get_median_depth`` | Structure | ``topology`` | ``tree`` | True | True | True | False | False |
| ``get_median_width`` | Structure | ``topology`` | ``tree`` | True | True | True | False | False |
| ``get_node_depth`` | Structure | ``topology`` | ``node`` | True | True | True | False | False |
| ``get_tree_widths`` | Structure | ``topology`` | ``distribution`` | True | True | True | False | False |
| ``get_edge_length`` | Path lengths | ``intrinsic_geometry`` | ``edge`` | True | True | False | True | False |
| ``get_total_cable_length`` | Path lengths | ``intrinsic_geometry`` | ``tree`` | True | True | False | True | False |
| ``get_degree_distribution`` | Degrees | ``topology`` | ``distribution`` | True | True | True | False | False |
| ``get_degrees`` | Degrees | ``topology`` | ``node`` | True | True | True | False | False |
| ``get_bifurcation_angle_sums`` | Geometry | ``intrinsic_geometry`` | ``bifurcation`` | True | True | True | True | False |
| ``get_bifurcation_angles`` | Geometry | ``intrinsic_geometry`` | ``bifurcation`` | True | True | True | True | False |
| ``get_bifurcation_deihedral_beta`` | Geometry | ``intrinsic_geometry`` | ``bifurcation`` | True | True | True | True | False |
| ``get_edge_angle_variance`` | Geometry | ``embedding`` | ``tree`` | True | False | True | True | True |
| ``get_edge_angles`` | Geometry | ``embedding`` | ``edge`` | True | False | True | True | True |
| ``get_mean_edge_angle`` | Geometry | ``embedding`` | ``tree`` | True | False | True | True | True |
| ``get_radial_angle`` | Geometry | ``intrinsic_geometry`` | ``edge`` | True | True | True | True | False |
| ``get_section_angular_deviation`` | Geometry | ``intrinsic_geometry`` | ``section`` | True | True | True | True | False |
| ``coordinate_pca`` | Coordinates | ``intrinsic_geometry`` | ``tree`` | True | False | False | True | False |
| ``get_convex_hull`` | Coordinates | ``intrinsic_geometry`` | ``tree`` | False | False | False | True | False |
| ``get_convex_hull_volume`` | Coordinates | ``intrinsic_geometry`` | ``tree`` | True | True | False | True | False |
| ``coordinate_extent_along_axis`` | Coordinate moments | ``embedding`` | ``tree`` | True | False | False | True | True |
| ``coordinate_mean_absolute_along_axis`` | Coordinate moments | ``embedding`` | ``tree`` | False | False | False | True | True |
| ``coordinate_mean_along_axis`` | Coordinate moments | ``embedding`` | ``tree`` | False | False | False | True | True |
| ``coordinate_minmax_along_axis`` | Coordinate moments | ``embedding`` | ``tree`` | False | False | False | True | True |
| ``coordinate_projection_moments`` | Coordinate moments | ``embedding`` | ``tree`` | False | False | False | True | True |
| ``coordinate_rms_along_axis`` | Coordinate moments | ``embedding`` | ``tree`` | False | False | False | True | True |
| ``coordinate_std_along_axis`` | Coordinate moments | ``embedding`` | ``tree`` | True | False | False | True | True |
| ``coordinate_variance_along_axis`` | Coordinate moments | ``embedding`` | ``tree`` | True | False | False | True | True |
| ``get_max_subtree_node`` | Subtrees | ``intrinsic_geometry`` | ``tree`` | True | True | True | True | False |
| ``get_partition_asymmetry`` | Subtrees | ``intrinsic_geometry`` | ``node`` | True | True | True | True | False |
| ``get_subtree_scores`` | Subtrees | ``intrinsic_geometry`` | ``node`` | True | True | True | True | False |
| ``fit_circle`` | Shape fitting | ``intrinsic_geometry`` | ``tree`` | False | False | False | True | False |
| ``fit_line`` | Shape fitting | ``intrinsic_geometry`` | ``tree`` | False | False | False | True | False |
| ``fit_plane`` | Shape fitting | ``intrinsic_geometry`` | ``tree`` | False | False | False | True | False |
| ``fit_sphere`` | Shape fitting | ``intrinsic_geometry`` | ``tree`` | False | False | False | True | False |
| ``distance_from_neuropil_surface`` | Neuropil distances | ``embedding`` | ``point`` | False | False | False | True | True |
| ``neuropil_point_depth`` | Neuropil distances | ``embedding`` | ``point`` | False | False | False | True | True |

## Programmatic reference

```python
import neurosetta as nr

for metric in nr.list_metric_definitions(domain="topology"):
    print(metric.domain, metric.level, metric.name)

nr.format_metrics_reference_table()  # DataFrame with classification columns

# High-level descriptors (topology + intrinsic geometry by default)
df = nr.describe(tree)
df = nr.describe(forest, domains=["topology", "intrinsic_geometry"])
df = nr.describe(forest, output="long")
```
