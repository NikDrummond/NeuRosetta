"""Catalog of descriptive metrics exposed by NeuRosetta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from ...ops._doc_helpers import TREE_METHOD_ALIASES

MetricDomain = Literal["topology", "intrinsic_geometry", "embedding"]
MetricLevel = Literal[
    "tree",
    "node",
    "edge",
    "branch",
    "bifurcation",
    "section",
    "point",
    "distribution",
    "population",
]

_METRIC_DOMAINS: frozenset[str] = frozenset(
    {"topology", "intrinsic_geometry", "embedding"}
)
_METRIC_LEVELS: frozenset[str] = frozenset(
    {
        "tree",
        "node",
        "edge",
        "branch",
        "bifurcation",
        "section",
        "point",
        "distribution",
        "population",
    }
)

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

# Shared metadata presets for common metric classes.
_TOPOLOGY = dict(
    domain="topology",
    translation_invariant=True,
    rotation_invariant=True,
    scale_invariant=True,
    requires_coordinates=False,
    requires_reference_frame=False,
)
_INTRINSIC_ANGLE = dict(
    domain="intrinsic_geometry",
    translation_invariant=True,
    rotation_invariant=True,
    scale_invariant=True,
    requires_coordinates=True,
    requires_reference_frame=False,
)
_INTRINSIC_LENGTH = dict(
    domain="intrinsic_geometry",
    translation_invariant=True,
    rotation_invariant=True,
    scale_invariant=False,
    requires_coordinates=True,
    requires_reference_frame=False,
)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """One documented metric and where it is exposed.

    Scientific metadata fields describe morphology domain, observational
    level, invariance under rigid/similarity transforms of neuron
    coordinates, and computational requirements. They are independent of
    ``category``, which remains a documentation grouping.
    """

    name: str
    category: str
    api_page: str
    tree_method: str | None = None
    forest_method: str | None = None
    notes: str = ""
    # Scientific morphology metadata (defaults preserve backward-compatible
    # construction as pure topology until callers set explicit values).
    domain: MetricDomain = "topology"
    level: MetricLevel = "tree"
    translation_invariant: bool = True
    rotation_invariant: bool = True
    scale_invariant: bool = True
    requires_coordinates: bool = False
    requires_reference_frame: bool = False

    def __post_init__(self) -> None:
        if self.tree_method is None and self.forest_method is None:
            raise ValueError(f"metric {self.name!r} must be exposed on Tree and/or Forest")
        if self.domain not in _METRIC_DOMAINS:
            raise ValueError(
                f"metric {self.name!r} has invalid domain {self.domain!r}; "
                f"expected one of {sorted(_METRIC_DOMAINS)}"
            )
        if self.level not in _METRIC_LEVELS:
            raise ValueError(
                f"metric {self.name!r} has invalid level {self.level!r}; "
                f"expected one of {sorted(_METRIC_LEVELS)}"
            )
        if self.requires_reference_frame and not self.requires_coordinates:
            raise ValueError(
                f"metric {self.name!r}: requires_reference_frame=True implies "
                "requires_coordinates=True"
            )


def _tree(
    name: str,
    category: str,
    api_page: str,
    *,
    domain: MetricDomain,
    level: MetricLevel,
    translation_invariant: bool,
    rotation_invariant: bool,
    scale_invariant: bool,
    requires_coordinates: bool,
    requires_reference_frame: bool = False,
    notes: str = "",
) -> MetricDefinition:
    method = TREE_METHOD_ALIASES.get(name, name)
    forest = name if name in _FOREST_BATCH_OPS else None
    return MetricDefinition(
        name,
        category,
        api_page,
        tree_method=method,
        forest_method=forest,
        notes=notes,
        domain=domain,
        level=level,
        translation_invariant=translation_invariant,
        rotation_invariant=rotation_invariant,
        scale_invariant=scale_invariant,
        requires_coordinates=requires_coordinates,
        requires_reference_frame=requires_reference_frame,
    )


def _forest_only(
    name: str,
    category: str,
    api_page: str,
    *,
    method: str,
    domain: MetricDomain,
    level: MetricLevel,
    translation_invariant: bool,
    rotation_invariant: bool,
    scale_invariant: bool,
    requires_coordinates: bool,
    requires_reference_frame: bool = False,
    notes: str = "",
) -> MetricDefinition:
    return MetricDefinition(
        name,
        category,
        api_page,
        forest_method=method,
        notes=notes,
        domain=domain,
        level=level,
        translation_invariant=translation_invariant,
        rotation_invariant=rotation_invariant,
        scale_invariant=scale_invariant,
        requires_coordinates=requires_coordinates,
        requires_reference_frame=requires_reference_frame,
    )


def _function(
    name: str,
    category: str,
    api_page: str,
    *,
    domain: MetricDomain,
    level: MetricLevel,
    translation_invariant: bool,
    rotation_invariant: bool,
    scale_invariant: bool,
    requires_coordinates: bool,
    requires_reference_frame: bool = False,
    notes: str = "",
) -> MetricDefinition:
    return MetricDefinition(
        name,
        category,
        api_page,
        tree_method="",
        forest_method="",
        notes=notes,
        domain=domain,
        level=level,
        translation_invariant=translation_invariant,
        rotation_invariant=rotation_invariant,
        scale_invariant=scale_invariant,
        requires_coordinates=requires_coordinates,
        requires_reference_frame=requires_reference_frame,
    )


METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    # --- Summary ---
    MetricDefinition(
        "tree_summary",
        "Summary",
        "tree_summary",
        tree_method="summary",
        notes=(
            "Formatted summary table (HTML in notebooks). "
            "Composite: topology counts + cable length."
        ),
        domain="intrinsic_geometry",
        level="tree",
        translation_invariant=True,
        rotation_invariant=True,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=False,
    ),
    MetricDefinition(
        "summary_table",
        "Summary",
        "tree_summary",
        tree_method="summary_table",
        notes=(
            "One-row numeric summary DataFrame. "
            "Composite: topology counts + cable length."
        ),
        domain="intrinsic_geometry",
        level="tree",
        translation_invariant=True,
        rotation_invariant=True,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=False,
    ),
    _forest_only(
        "forest_summary",
        "Summary",
        "tree_summary",
        method="summary",
        notes=(
            "Per-tree summary with optional TOTAL/MEAN rows. "
            "Composite: topology counts + cable length."
        ),
        domain="intrinsic_geometry",
        level="population",
        translation_invariant=True,
        rotation_invariant=True,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=False,
    ),
    _forest_only(
        "forest_summary_table",
        "Summary",
        "tree_summary",
        method="summary_table",
        notes=(
            "Per-tree numeric summary DataFrame. "
            "Composite: topology counts + cable length."
        ),
        domain="intrinsic_geometry",
        level="population",
        translation_invariant=True,
        rotation_invariant=True,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=False,
    ),
    # --- Counting ---
    _tree("count_roots", "Counting", "counting", **_TOPOLOGY, level="tree"),
    _tree("count_nodes", "Counting", "counting", **_TOPOLOGY, level="tree"),
    _tree("count_edges", "Counting", "counting", **_TOPOLOGY, level="tree"),
    _tree("count_leaves", "Counting", "counting", **_TOPOLOGY, level="tree"),
    _tree("count_branches", "Counting", "counting", **_TOPOLOGY, level="tree"),
    _tree("count_transitive_nodes", "Counting", "counting", **_TOPOLOGY, level="tree"),
    _tree("count_sections", "Counting", "counting", **_TOPOLOGY, level="tree"),
    _tree("count_bifurcations", "Counting", "counting", **_TOPOLOGY, level="tree"),
    _tree("count_core_nodes", "Counting", "counting", **_TOPOLOGY, level="tree"),
    # --- Structure ---
    _tree(
        "get_node_depth",
        "Structure",
        "tree_structure",
        notes="Per-node depth from root.",
        **_TOPOLOGY,
        level="node",
    ),
    _tree("get_max_depth", "Structure", "tree_structure", **_TOPOLOGY, level="tree"),
    _tree("get_mean_depth", "Structure", "tree_structure", **_TOPOLOGY, level="tree"),
    _tree("get_median_depth", "Structure", "tree_structure", **_TOPOLOGY, level="tree"),
    _tree(
        "get_tree_widths",
        "Structure",
        "tree_structure",
        notes="Node count at each depth.",
        **_TOPOLOGY,
        level="distribution",
    ),
    _tree("get_max_width", "Structure", "tree_structure", **_TOPOLOGY, level="tree"),
    _tree("get_mean_width", "Structure", "tree_structure", **_TOPOLOGY, level="tree"),
    _tree("get_median_width", "Structure", "tree_structure", **_TOPOLOGY, level="tree"),
    _tree("get_binary_ratio", "Structure", "tree_structure", **_TOPOLOGY, level="tree"),
    # --- Path lengths ---
    _tree(
        "get_edge_length",
        "Path lengths",
        "path_lengths",
        notes="Euclidean length per edge; cable semantics depend on reduction state.",
        **_INTRINSIC_LENGTH,
        level="edge",
    ),
    _tree(
        "get_total_cable_length",
        "Path lengths",
        "path_lengths",
        **_INTRINSIC_LENGTH,
        level="tree",
    ),
    # --- Degrees ---
    _tree(
        "get_degrees",
        "Degrees",
        "degrees",
        notes="In/out/total degree per node.",
        **_TOPOLOGY,
        level="node",
    ),
    _tree(
        "get_degree_distribution",
        "Degrees",
        "degrees",
        **_TOPOLOGY,
        level="distribution",
    ),
    # --- Geometry ---
    # REVIEW: between_vector is a caller-supplied external direction.
    _tree(
        "get_edge_angles",
        "Geometry",
        "tree_geometry",
        domain="embedding",
        level="edge",
        translation_invariant=True,
        rotation_invariant=False,
        scale_invariant=True,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="Angles relative to caller-supplied between_vector.",
    ),
    _tree(
        "get_mean_edge_angle",
        "Geometry",
        "tree_geometry",
        domain="embedding",
        level="tree",
        translation_invariant=True,
        rotation_invariant=False,
        scale_invariant=True,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="Circular mean of get_edge_angles; same reference-frame dependence.",
    ),
    _tree(
        "get_edge_angle_variance",
        "Geometry",
        "tree_geometry",
        domain="embedding",
        level="tree",
        translation_invariant=True,
        rotation_invariant=False,
        scale_invariant=True,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="Circular variance of get_edge_angles; same reference-frame dependence.",
    ),
    # REVIEW: unsigned angles are root-relative intrinsic geometry. Signed
    # mode requires alignment_vector (external) and is then not RI.
    _tree(
        "get_radial_angle",
        "Geometry",
        "tree_geometry",
        **_INTRINSIC_ANGLE,
        level="edge",
        notes=(
            "Unsigned mode is intrinsic; signed mode needs alignment_vector "
            "and loses rotation invariance relative to that axis."
        ),
    ),
    _tree(
        "get_bifurcation_angles",
        "Geometry",
        "tree_geometry",
        **_INTRINSIC_ANGLE,
        level="bifurcation",
    ),
    _tree(
        "get_bifurcation_angle_sums",
        "Geometry",
        "tree_geometry",
        **_INTRINSIC_ANGLE,
        level="bifurcation",
    ),
    _tree(
        "get_bifurcation_deihedral_beta",
        "Geometry",
        "tree_geometry",
        **_INTRINSIC_ANGLE,
        level="bifurcation",
    ),
    MetricDefinition(
        "get_section_angular_deviation",
        "Geometry",
        "tree_geometry",
        tree_method="get_section_angular_deviation",
        notes="Requires a non-reduced tree.",
        domain="intrinsic_geometry",
        level="section",
        translation_invariant=True,
        rotation_invariant=True,
        scale_invariant=True,
        requires_coordinates=True,
        requires_reference_frame=False,
    ),
    # --- Coordinates ---
    # REVIEW: returns (evals, evecs). Eigenvalues (esp. norm=True) describe
    # intrinsic shape; eigenvectors encode lab-frame orientation → RI=False.
    # SI=False conservatively because norm=False returns absolute eigenvalues.
    _tree(
        "coordinate_pca",
        "Coordinates",
        "coordinates",
        domain="intrinsic_geometry",
        level="tree",
        translation_invariant=True,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=False,
        notes=(
            "Returns (evals, evecs); evals can be shape-descriptive, evecs are "
            "lab-frame orientations."
        ),
    ),
    # REVIEW: SciPy ConvexHull stores absolute vertex coordinates, so the
    # returned object is not T/R/S invariant even though hull *shape* is.
    _tree(
        "get_convex_hull",
        "Coordinates",
        "coordinates",
        domain="intrinsic_geometry",
        level="tree",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=False,
        notes="Returns ConvexHull with absolute vertex coordinates.",
    ),
    _tree(
        "get_convex_hull_volume",
        "Coordinates",
        "coordinates",
        **_INTRINSIC_LENGTH,
        level="tree",
    ),
    # --- Coordinate moments ---
    # REVIEW: all require a caller-supplied axis treated as an external frame.
    _tree(
        "coordinate_mean_along_axis",
        "Coordinate moments",
        "coordinate_moments",
        domain="embedding",
        level="tree",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="Mean of absolute projections onto caller-supplied axis.",
    ),
    _tree(
        "coordinate_variance_along_axis",
        "Coordinate moments",
        "coordinate_moments",
        domain="embedding",
        level="tree",
        translation_invariant=True,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="Variance of projections onto caller-supplied axis.",
    ),
    _tree(
        "coordinate_std_along_axis",
        "Coordinate moments",
        "coordinate_moments",
        domain="embedding",
        level="tree",
        translation_invariant=True,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="Std of projections onto caller-supplied axis.",
    ),
    _tree(
        "coordinate_minmax_along_axis",
        "Coordinate moments",
        "coordinate_moments",
        domain="embedding",
        level="tree",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="Min/max absolute projections onto caller-supplied axis.",
    ),
    _tree(
        "coordinate_extent_along_axis",
        "Coordinate moments",
        "coordinate_moments",
        domain="embedding",
        level="tree",
        translation_invariant=True,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="pmax - pmin along caller-supplied axis.",
    ),
    # RMS of absolute projections onto a fixed axis (origin-dependent).
    _tree(
        "coordinate_rms_along_axis",
        "Coordinate moments",
        "coordinate_moments",
        domain="embedding",
        level="tree",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="RMS of absolute projections onto caller-supplied axis.",
    ),
    # Origin-absolute |projection|; translation shifts the value.
    _tree(
        "coordinate_mean_absolute_along_axis",
        "Coordinate moments",
        "coordinate_moments",
        domain="embedding",
        level="tree",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="Mean |projection| onto caller-supplied axis.",
    ),
    # Compound (mean/var/std/min/max); TI false from absolute mean/minmax.
    _tree(
        "coordinate_projection_moments",
        "Coordinate moments",
        "coordinate_moments",
        domain="embedding",
        level="tree",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=True,
        notes="Compound mean/var/std/minmax along caller-supplied axis.",
    ),
    # --- Subtrees ---
    # Cable-fraction + leaf-fraction score; ratios → scale invariant.
    _tree(
        "get_subtree_scores",
        "Subtrees",
        "subtrees",
        domain="intrinsic_geometry",
        level="node",
        translation_invariant=True,
        rotation_invariant=True,
        scale_invariant=True,
        requires_coordinates=True,
        requires_reference_frame=False,
    ),
    _tree(
        "get_max_subtree_node",
        "Subtrees",
        "subtrees",
        domain="intrinsic_geometry",
        level="tree",
        translation_invariant=True,
        rotation_invariant=True,
        scale_invariant=True,
        requires_coordinates=True,
        requires_reference_frame=False,
    ),
    # REVIEW: unweighted PA is pure topology (leaf counts). Default
    # weighted=True multiplies by subtree cable fraction → classified as
    # intrinsic_geometry with requires_coordinates=True.
    _tree(
        "get_partition_asymmetry",
        "Subtrees",
        "subtrees",
        domain="intrinsic_geometry",
        level="node",
        translation_invariant=True,
        rotation_invariant=True,
        scale_invariant=True,
        requires_coordinates=True,
        requires_reference_frame=False,
        notes=(
            "Default weighted=True uses Path_length cable fractions; "
            "unweighted mode is topological."
        ),
    ),
    # --- Shape fitting ---
    # REVIEW: multi-output (center/radius/direction/variances) with mixed
    # invariance. Flags are conservative for absolute pose parts.
    _tree(
        "fit_sphere",
        "Shape fitting",
        "shape_fitting",
        domain="intrinsic_geometry",
        level="tree",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=False,
        notes="Returns center+radius (or vedo object); mixed invariance.",
    ),
    _tree(
        "fit_line",
        "Shape fitting",
        "shape_fitting",
        domain="intrinsic_geometry",
        level="tree",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=False,
        notes="Returns slope/center/variances; mixed invariance.",
    ),
    _tree(
        "fit_plane",
        "Shape fitting",
        "shape_fitting",
        domain="intrinsic_geometry",
        level="tree",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=False,
        notes="Returns normal/center/variances; mixed invariance.",
    ),
    _tree(
        "fit_circle",
        "Shape fitting",
        "shape_fitting",
        domain="intrinsic_geometry",
        level="tree",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=False,
        notes="Returns center/radius/normal; mixed invariance.",
    ),
    # --- Neuropil distances ---
    _function(
        "distance_from_neuropil_surface",
        "Neuropil distances",
        "neuropils",
        notes="Standalone function; requires a neuropil mesh.",
        domain="embedding",
        level="point",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=True,
    ),
    _function(
        "neuropil_point_depth",
        "Neuropil distances",
        "neuropils",
        notes=(
            "Standalone function; requires a neuropil mesh. "
            "Invariance is neuron-vs-fixed-mesh (not joint transforms)."
        ),
        domain="embedding",
        level="point",
        translation_invariant=False,
        rotation_invariant=False,
        scale_invariant=False,
        requires_coordinates=True,
        requires_reference_frame=True,
    ),
)


def list_metric_definitions(
    *,
    domain: MetricDomain | None = None,
    level: MetricLevel | None = None,
) -> tuple[MetricDefinition, ...]:
    """Return the documented metric catalog.

    Parameters
    ----------
    domain : MetricDomain | None, optional
        If given, keep only metrics with this morphology domain.
    level : MetricLevel | None, optional
        If given, keep only metrics with this observational level.
    """
    definitions = METRIC_DEFINITIONS
    if domain is not None:
        if domain not in _METRIC_DOMAINS:
            raise ValueError(
                f"invalid domain {domain!r}; expected one of {sorted(_METRIC_DOMAINS)}"
            )
        definitions = tuple(item for item in definitions if item.domain == domain)
    if level is not None:
        if level not in _METRIC_LEVELS:
            raise ValueError(
                f"invalid level {level!r}; expected one of {sorted(_METRIC_LEVELS)}"
            )
        definitions = tuple(item for item in definitions if item.level == level)
    return definitions


def _sorted_metric_definitions() -> list[MetricDefinition]:
    order = {name: index for index, name in enumerate(_CATEGORY_ORDER)}
    return sorted(
        METRIC_DEFINITIONS,
        key=lambda item: (order.get(item.category, len(_CATEGORY_ORDER)), item.name),
    )


def _format_tree_column(defn: MetricDefinition, *, rst: bool) -> str:
    if defn.tree_method == "":
        text = f"nr.{defn.name}()"
    elif defn.tree_method is None:
        text = "—"
    else:
        text = f"tree.{defn.tree_method}()"
    return f"``{text}``" if rst and text != "—" else text


def _format_forest_column(defn: MetricDefinition, *, rst: bool) -> str:
    if defn.forest_method == "":
        text = "—"
    elif defn.forest_method is None:
        text = "—"
    else:
        text = f"forest.{defn.forest_method}()"
    return f"``{text}``" if rst and text != "—" else text


def _api_link(defn: MetricDefinition) -> str:
    if defn.api_page == "neuropils":
        return "{doc}`../api/neuropils`"
    return f"{{doc}}`../api/tree_ops/{defn.api_page}`"


def format_metrics_reference_table() -> pd.DataFrame:
    """Return a table of all documented metrics.

    In notebooks the DataFrame renders as an HTML table. Columns include
    scientific classification metadata. Use
    :func:`format_metrics_reference_markdown` for the Sphinx quick-reference
    table, or :func:`format_metrics_classification_markdown` for the full
    classification review table.
    """
    rows = [
        {
            "category": defn.category,
            "metric": defn.name,
            "domain": defn.domain,
            "level": defn.level,
            "translation_invariant": defn.translation_invariant,
            "rotation_invariant": defn.rotation_invariant,
            "scale_invariant": defn.scale_invariant,
            "requires_coordinates": defn.requires_coordinates,
            "requires_reference_frame": defn.requires_reference_frame,
            "tree": _format_tree_column(defn, rst=False),
            "forest": _format_forest_column(defn, rst=False),
            "api": defn.api_page,
            "notes": defn.notes,
        }
        for defn in _sorted_metric_definitions()
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "category",
            "metric",
            "domain",
            "level",
            "translation_invariant",
            "rotation_invariant",
            "scale_invariant",
            "requires_coordinates",
            "requires_reference_frame",
            "tree",
            "forest",
            "api",
            "notes",
        ],
    )


def format_metrics_reference_markdown() -> str:
    """Return a Sphinx markdown table of all documented metrics."""
    lines = [
        "| Category | Metric | Tree | Forest | API | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for defn in _sorted_metric_definitions():
        notes = defn.notes.replace("|", "\\|")
        lines.append(
            "| "
            f"{defn.category} | "
            f"``{defn.name}`` | "
            f"{_format_tree_column(defn, rst=True)} | "
            f"{_format_forest_column(defn, rst=True)} | "
            f"{_api_link(defn)} | "
            f"{notes} |"
        )
    return "\n".join(lines)


def format_metrics_classification_markdown() -> str:
    """Return a markdown classification table for scientific review.

    One row per registered metric with domain, level, invariances, and
    requirements. Generated from :data:`METRIC_DEFINITIONS` (source of truth).
    """
    lines = [
        "| Metric | Category | Domain | Level | T | R | S | Coords | Ref |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for defn in _sorted_metric_definitions():
        lines.append(
            "| "
            f"``{defn.name}`` | "
            f"{defn.category} | "
            f"``{defn.domain}`` | "
            f"``{defn.level}`` | "
            f"{defn.translation_invariant} | "
            f"{defn.rotation_invariant} | "
            f"{defn.scale_invariant} | "
            f"{defn.requires_coordinates} | "
            f"{defn.requires_reference_frame} |"
        )
    return "\n".join(lines)


__all__ = [
    "MetricDefinition",
    "MetricDomain",
    "MetricLevel",
    "METRIC_DEFINITIONS",
    "list_metric_definitions",
    "format_metrics_reference_table",
    "format_metrics_reference_markdown",
    "format_metrics_classification_markdown",
]
