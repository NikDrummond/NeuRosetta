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
| Summary | ``forest_summary`` | — | ``forest.summary()`` | {doc}`../api/tree_ops/tree_summary` | Per-tree summary with optional TOTAL/MEAN rows. |
| Summary | ``forest_summary_table`` | — | ``forest.summary_table()`` | {doc}`../api/tree_ops/tree_summary` | Per-tree numeric summary DataFrame. |
| Summary | ``summary_table`` | ``tree.summary_table()`` | — | {doc}`../api/tree_ops/tree_summary` | One-row numeric summary DataFrame. |
| Summary | ``tree_summary`` | ``tree.summary()`` | — | {doc}`../api/tree_ops/tree_summary` | Formatted summary table (HTML in notebooks). |
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
| Geometry | ``get_edge_angle_variance`` | ``tree.get_edge_angle_variance()`` | ``forest.get_edge_angle_variance()`` | {doc}`../api/tree_ops/tree_geometry` |  |
| Geometry | ``get_edge_angles`` | ``tree.get_edge_angles()`` | ``forest.get_edge_angles()`` | {doc}`../api/tree_ops/tree_geometry` |  |
| Geometry | ``get_mean_edge_angle`` | ``tree.get_mean_edge_angle()`` | ``forest.get_mean_edge_angle()`` | {doc}`../api/tree_ops/tree_geometry` |  |
| Geometry | ``get_radial_angle`` | ``tree.get_radial_angle()`` | ``forest.get_radial_angle()`` | {doc}`../api/tree_ops/tree_geometry` |  |
| Geometry | ``get_section_angular_deviation`` | ``tree.get_section_angular_deviation()`` | — | {doc}`../api/tree_ops/tree_geometry` | Requires a non-reduced tree. |
| Coordinates | ``coordinate_pca`` | ``tree.coordinate_pca()`` | ``forest.coordinate_pca()`` | {doc}`../api/tree_ops/coordinates` |  |
| Coordinates | ``get_convex_hull`` | ``tree.get_convex_hull()`` | ``forest.get_convex_hull()`` | {doc}`../api/tree_ops/coordinates` |  |
| Coordinates | ``get_convex_hull_volume`` | ``tree.get_convex_hull_volume()`` | ``forest.get_convex_hull_volume()`` | {doc}`../api/tree_ops/coordinates` |  |
| Coordinate moments | ``coordinate_extent_along_axis`` | ``tree.coordinate_extent_along_axis()`` | ``forest.coordinate_extent_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` |  |
| Coordinate moments | ``coordinate_mean_absolute_along_axis`` | ``tree.coordinate_mean_absolute_along_axis()`` | ``forest.coordinate_mean_absolute_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` |  |
| Coordinate moments | ``coordinate_mean_along_axis`` | ``tree.coordinate_mean_along_axis()`` | ``forest.coordinate_mean_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` |  |
| Coordinate moments | ``coordinate_minmax_along_axis`` | ``tree.coordinate_minmax_along_axis()`` | ``forest.coordinate_minmax_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` |  |
| Coordinate moments | ``coordinate_projection_moments`` | ``tree.coordinate_projection_moments()`` | ``forest.coordinate_projection_moments()`` | {doc}`../api/tree_ops/coordinate_moments` |  |
| Coordinate moments | ``coordinate_rms_along_axis`` | ``tree.coordinate_rms_along_axis()`` | ``forest.coordinate_rms_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` |  |
| Coordinate moments | ``coordinate_std_along_axis`` | ``tree.coordinate_std_along_axis()`` | ``forest.coordinate_std_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` |  |
| Coordinate moments | ``coordinate_variance_along_axis`` | ``tree.coordinate_variance_along_axis()`` | ``forest.coordinate_variance_along_axis()`` | {doc}`../api/tree_ops/coordinate_moments` |  |
| Subtrees | ``get_max_subtree_node`` | ``tree.get_max_subtree_node()`` | ``forest.get_max_subtree_node()`` | {doc}`../api/tree_ops/subtrees` |  |
| Subtrees | ``get_partition_asymmetry`` | ``tree.get_partition_asymmetry()`` | ``forest.get_partition_asymmetry()`` | {doc}`../api/tree_ops/subtrees` |  |
| Subtrees | ``get_subtree_scores`` | ``tree.get_subtree_scores()`` | ``forest.get_subtree_scores()`` | {doc}`../api/tree_ops/subtrees` |  |
| Shape fitting | ``fit_circle`` | ``tree.fit_circle()`` | ``forest.fit_circle()`` | {doc}`../api/tree_ops/shape_fitting` |  |
| Shape fitting | ``fit_line`` | ``tree.fit_line()`` | ``forest.fit_line()`` | {doc}`../api/tree_ops/shape_fitting` |  |
| Shape fitting | ``fit_plane`` | ``tree.fit_plane()`` | ``forest.fit_plane()`` | {doc}`../api/tree_ops/shape_fitting` |  |
| Shape fitting | ``fit_sphere`` | ``tree.fit_sphere()`` | ``forest.fit_sphere()`` | {doc}`../api/tree_ops/shape_fitting` |  |
| Neuropil distances | ``distance_from_neuropil_surface`` | ``nr.distance_from_neuropil_surface()`` | — | {doc}`../api/neuropils` | Standalone function; requires a neuropil mesh. |
| Neuropil distances | ``neuropil_point_depth`` | ``nr.neuropil_point_depth()`` | — | {doc}`../api/neuropils` | Standalone function; requires a neuropil mesh. |

## Programmatic reference

```python
import neurosetta as nr

for metric in nr.list_metric_definitions():
    print(metric.category, metric.name, metric.tree_method, metric.forest_method)

print(nr.format_metrics_reference_table())
```
