"""Shared helpers for documenting ops bound onto API classes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Tree method name when it differs from the underlying op function name.
TREE_METHOD_ALIASES: dict[str, str] = {
    "breadth_first_search": "tree_breadth_first_search",
    "breadth_first_iterator": "tree_breadth_first_iterator",
    "depth_first_search": "tree_depth_first_search",
    "depth_first_iterator": "tree_depth_first_iterator",
    "check_reduced": "is_reduced",
    "reduce_tree": "get_reduced_tree",
    "reroot_tree": "get_rerooted_tree",
    "mask_subtree_from_root": "subtree_mask_from_root",
}

# Forest method name when it differs from the underlying op function name.
FOREST_METHOD_ALIASES: dict[str, str] = {
    "check_reduced": "is_reduced",
}

# Ops in tree_graphs that are also exposed on Forest (via Forest.apply batch API).
FOREST_BOUND_OPS: frozenset[str] = frozenset(
    {
        "align_coordinates",
        "align_coordinates_to_vector",
        "apply_rotation_steps_to_coordinates",
        "center_coordinates_at_centroid",
        "center_coordinates_at_root",
        "check_reduced",
        "coordinate_extent_along_axis",
        "coordinate_mean_absolute_along_axis",
        "coordinate_mean_along_axis",
        "coordinate_minmax_along_axis",
        "coordinate_pca",
        "coordinate_projection_moments",
        "coordinate_rms_along_axis",
        "coordinate_std_along_axis",
        "coordinate_variance_along_axis",
        "count_bifurcations",
        "count_branches",
        "count_edges",
        "count_leaves",
        "count_nodes",
        "count_roots",
        "count_sections",
        "count_transitive_nodes",
        "fit_circle",
        "fit_line",
        "fit_plane",
        "fit_sphere",
        "get_bifurcation_angle_sums",
        "get_bifurcation_angles",
        "get_bifurcation_deihedral_beta",
        "get_bifurcation_indices",
        "get_branch_indices",
        "get_convex_hull",
        "get_convex_hull_volume",
        "get_core_indices",
        "get_degree_distribution",
        "get_degrees",
        "get_edge_angle_variance",
        "get_edge_angles",
        "get_edge_coordinates",
        "get_edge_indices",
        "get_edge_length",
        "get_leaf_indices",
        "get_max_subtree_node",
        "get_mean_edge_angle",
        "get_node_coordinates",
        "get_node_depth",
        "get_partition_asymmetry",
        "get_radial_angle",
        "get_root_coordinate",
        "get_root_index",
        "get_subtree",
        "get_subtree_scores",
        "get_total_cable_length",
        "has_property",
        "recenter_coordinates",
        "reduce_tree",
        "rotate_coordinates",
        "rotate_coordinates_about",
        "scale_coordinates",
        "scale_coordinates_about",
        "scale_coordinates_along_pca",
        "translate_coordinates",
        "update_reduced",
    }
)


def api_binding_note(func_name: str) -> str:
    """Return a See Also block for Tree (and Forest when applicable) bindings."""

    tree_method = TREE_METHOD_ALIASES.get(func_name, func_name)
    lines = [
        "",
        "See Also",
        "--------",
        f":meth:`~neurosetta.api.tree_class.Tree.{tree_method}`",
    ]
    if func_name in FOREST_BOUND_OPS:
        forest_method = FOREST_METHOD_ALIASES.get(func_name, func_name)
        lines.append(f":meth:`~neurosetta.api.forest_class.Forest.{forest_method}`")
    return "\n".join(lines)


def enrich_tree_graph_docstrings(module_globals: dict[str, Any]) -> None:
    """Append API-binding notes to public callables defined in a tree_graphs submodule."""

    module_name = module_globals.get("__name__")
    for name, obj in module_globals.items():
        if name.startswith("_") or not isinstance(obj, Callable):
            continue
        if getattr(obj, "__module__", None) != module_name:
            continue
        note = api_binding_note(name)
        doc = obj.__doc__ or ""
        if "See Also" not in doc:
            obj.__doc__ = doc.rstrip() + note
