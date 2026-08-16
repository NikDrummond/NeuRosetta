"""Catalog of descriptive metrics exposed by NeuRosetta."""

from __future__ import annotations

from dataclasses import dataclass

from ...ops._doc_helpers import TREE_METHOD_ALIASES

# Ops bound as batch methods on Forest (mirrors forest_class metric bindings).
_FOREST_BATCH_OPS: frozenset[str] = frozenset(
    {
        "count_bifurcations",
        "count_branches",
        "count_core_nodes",
        "count_edges",
        "count_leaves",
        "count_nodes",
        "count_roots",
        "count_sections",
        "count_transitive_nodes",
        "coordinate_extent_along_axis",
        "coordinate_mean_absolute_along_axis",
        "coordinate_mean_along_axis",
        "coordinate_minmax_along_axis",
        "coordinate_pca",
        "coordinate_projection_moments",
        "coordinate_rms_along_axis",
        "coordinate_std_along_axis",
        "coordinate_variance_along_axis",
        "fit_circle",
        "fit_line",
        "fit_plane",
        "fit_sphere",
        "get_bifurcation_angle_sums",
        "get_bifurcation_angles",
        "get_bifurcation_deihedral_beta",
        "get_binary_ratio",
        "get_convex_hull",
        "get_convex_hull_volume",
        "get_degree_distribution",
        "get_degrees",
        "get_edge_angle_variance",
        "get_edge_angles",
        "get_edge_length",
        "get_max_depth",
        "get_max_subtree_node",
        "get_max_width",
        "get_mean_depth",
        "get_mean_edge_angle",
        "get_mean_width",
        "get_median_depth",
        "get_median_width",
        "get_node_depth",
        "get_partition_asymmetry",
        "get_radial_angle",
        "get_subtree_scores",
        "get_total_cable_length",
        "get_tree_widths",
    }
)

_CATEGORY_ORDER: tuple[str, ...] = (
    "Summary",
    "Counting",
    "Structure",
    "Path lengths",
    "Degrees",
    "Geometry",
    "Coordinates",
    "Coordinate moments",
    "Subtrees",
    "Shape fitting",
    "Neuropil distances",
)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """One documented metric and where it is exposed."""

    name: str
    category: str
    api_page: str
    tree_method: str | None = None
    forest_method: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.tree_method is None and self.forest_method is None:
            raise ValueError(f"metric {self.name!r} must be exposed on Tree and/or Forest")


def _tree(name: str, category: str, api_page: str, *, notes: str = "") -> MetricDefinition:
    method = TREE_METHOD_ALIASES.get(name, name)
    forest = name if name in _FOREST_BATCH_OPS else None
    return MetricDefinition(
        name, category, api_page, tree_method=method, forest_method=forest, notes=notes
    )


def _forest_only(
    name: str,
    category: str,
    api_page: str,
    *,
    method: str,
    notes: str = "",
) -> MetricDefinition:
    return MetricDefinition(name, category, api_page, forest_method=method, notes=notes)


def _function(name: str, category: str, api_page: str, *, notes: str = "") -> MetricDefinition:
    return MetricDefinition(name, category, api_page, tree_method="", forest_method="", notes=notes)


METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    # --- Summary ---
    MetricDefinition(
        "tree_summary",
        "Summary",
        "tree_summary",
        tree_method="summary",
        notes="Formatted summary table (HTML in notebooks).",
    ),
    MetricDefinition(
        "summary_table",
        "Summary",
        "tree_summary",
        tree_method="summary_table",
        notes="One-row numeric summary DataFrame.",
    ),
    _forest_only(
        "forest_summary",
        "Summary",
        "tree_summary",
        method="summary",
        notes="Per-tree summary with optional TOTAL/MEAN rows.",
    ),
    _forest_only(
        "forest_summary_table",
        "Summary",
        "tree_summary",
        method="summary_table",
        notes="Per-tree numeric summary DataFrame.",
    ),
    # --- Counting ---
    _tree("count_roots", "Counting", "counting"),
    _tree("count_nodes", "Counting", "counting"),
    _tree("count_edges", "Counting", "counting"),
    _tree("count_leaves", "Counting", "counting"),
    _tree("count_branches", "Counting", "counting"),
    _tree("count_transitive_nodes", "Counting", "counting"),
    _tree("count_sections", "Counting", "counting"),
    _tree("count_bifurcations", "Counting", "counting"),
    _tree("count_core_nodes", "Counting", "counting"),
    # --- Structure ---
    _tree("get_node_depth", "Structure", "tree_structure", notes="Per-node depth from root."),
    _tree("get_max_depth", "Structure", "tree_structure"),
    _tree("get_mean_depth", "Structure", "tree_structure"),
    _tree("get_median_depth", "Structure", "tree_structure"),
    _tree("get_tree_widths", "Structure", "tree_structure", notes="Node count at each depth."),
    _tree("get_max_width", "Structure", "tree_structure"),
    _tree("get_mean_width", "Structure", "tree_structure"),
    _tree("get_median_width", "Structure", "tree_structure"),
    _tree("get_binary_ratio", "Structure", "tree_structure"),
    # --- Path lengths ---
    _tree(
        "get_edge_length",
        "Path lengths",
        "path_lengths",
        notes="Euclidean length per edge; cable semantics depend on reduction state.",
    ),
    _tree("get_total_cable_length", "Path lengths", "path_lengths"),
    # --- Degrees ---
    _tree("get_degrees", "Degrees", "degrees", notes="In/out/total degree per node."),
    _tree("get_degree_distribution", "Degrees", "degrees"),
    # --- Geometry ---
    _tree("get_edge_angles", "Geometry", "tree_geometry"),
    _tree("get_mean_edge_angle", "Geometry", "tree_geometry"),
    _tree("get_edge_angle_variance", "Geometry", "tree_geometry"),
    _tree("get_radial_angle", "Geometry", "tree_geometry"),
    _tree("get_bifurcation_angles", "Geometry", "tree_geometry"),
    _tree("get_bifurcation_angle_sums", "Geometry", "tree_geometry"),
    _tree("get_bifurcation_deihedral_beta", "Geometry", "tree_geometry"),
    MetricDefinition(
        "get_section_angular_deviation",
        "Geometry",
        "tree_geometry",
        tree_method="get_section_angular_deviation",
        notes="Requires a non-reduced tree.",
    ),
    # --- Coordinates ---
    _tree("coordinate_pca", "Coordinates", "coordinates"),
    _tree("get_convex_hull", "Coordinates", "coordinates"),
    _tree("get_convex_hull_volume", "Coordinates", "coordinates"),
    # --- Coordinate moments ---
    _tree("coordinate_mean_along_axis", "Coordinate moments", "coordinate_moments"),
    _tree("coordinate_variance_along_axis", "Coordinate moments", "coordinate_moments"),
    _tree("coordinate_std_along_axis", "Coordinate moments", "coordinate_moments"),
    _tree("coordinate_minmax_along_axis", "Coordinate moments", "coordinate_moments"),
    _tree("coordinate_extent_along_axis", "Coordinate moments", "coordinate_moments"),
    _tree("coordinate_rms_along_axis", "Coordinate moments", "coordinate_moments"),
    _tree("coordinate_mean_absolute_along_axis", "Coordinate moments", "coordinate_moments"),
    _tree("coordinate_projection_moments", "Coordinate moments", "coordinate_moments"),
    # --- Subtrees ---
    _tree("get_subtree_scores", "Subtrees", "subtrees"),
    _tree("get_max_subtree_node", "Subtrees", "subtrees"),
    _tree("get_partition_asymmetry", "Subtrees", "subtrees"),
    # --- Shape fitting ---
    _tree("fit_sphere", "Shape fitting", "shape_fitting"),
    _tree("fit_line", "Shape fitting", "shape_fitting"),
    _tree("fit_plane", "Shape fitting", "shape_fitting"),
    _tree("fit_circle", "Shape fitting", "shape_fitting"),
    # --- Neuropil distances ---
    _function(
        "distance_from_neuropil_surface",
        "Neuropil distances",
        "neuropils",
        notes="Standalone function; requires a neuropil mesh.",
    ),
    _function(
        "neuropil_point_depth",
        "Neuropil distances",
        "neuropils",
        notes="Standalone function; requires a neuropil mesh.",
    ),
)


def list_metric_definitions() -> tuple[MetricDefinition, ...]:
    """Return the documented metric catalog."""
    return METRIC_DEFINITIONS


def _format_tree_column(defn: MetricDefinition) -> str:
    if defn.tree_method == "":
        return f"``nr.{defn.name}()``"
    if defn.tree_method is None:
        return "—"
    return f"``tree.{defn.tree_method}()``"


def _format_forest_column(defn: MetricDefinition) -> str:
    if defn.forest_method == "":
        return "—"
    if defn.forest_method is None:
        return "—"
    return f"``forest.{defn.forest_method}()``"


def _api_link(defn: MetricDefinition) -> str:
    if defn.api_page == "neuropils":
        return "{doc}`../api/neuropils`"
    return f"{{doc}}`../api/tree_ops/{defn.api_page}`"


def format_metrics_reference_table() -> str:
    """Return a markdown table of all documented metrics."""
    lines = [
        "| Category | Metric | Tree | Forest | API | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    order = {name: index for index, name in enumerate(_CATEGORY_ORDER)}
    sorted_defs = sorted(
        METRIC_DEFINITIONS,
        key=lambda item: (order.get(item.category, len(_CATEGORY_ORDER)), item.name),
    )
    for defn in sorted_defs:
        notes = defn.notes.replace("|", "\\|")
        lines.append(
            "| "
            f"{defn.category} | "
            f"``{defn.name}`` | "
            f"{_format_tree_column(defn)} | "
            f"{_format_forest_column(defn)} | "
            f"{_api_link(defn)} | "
            f"{notes} |"
        )
    return "\n".join(lines)


__all__ = [
    "MetricDefinition",
    "METRIC_DEFINITIONS",
    "list_metric_definitions",
    "format_metrics_reference_table",
]
