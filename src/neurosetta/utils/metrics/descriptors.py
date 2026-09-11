"""High-level morphology descriptor tables from registered metrics.

``describe()`` orchestrates existing Tree/Forest metric methods. It does not
implement new morphometrics. Non-scalar metric outputs are converted into
explicit named summary descriptors (mean, std, …).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from ...api import Forest, Tree
from ...api.anatomical_frame import (
    METRIC_FRAME_REQUIREMENTS,
    AnatomicalFrame,
    FrameRequirementError,
)
from ...core import _Forest, _Tree
from ...ops.tree_graphs.tree_coordinates import get_node_coordinates
from ...ops.units import get_units
from .registry import (
    METRIC_DEFINITIONS,
    MetricDefinition,
    MetricDomain,
    _METRIC_DOMAINS,
)

DescribeOutput = Literal["wide", "long"]
DescribeErrors = Literal["raise", "warn", "ignore"]
SummaryName = Literal["count", "mean", "std", "median", "min", "max", "q25", "q75"]

DEFAULT_SUMMARIES: tuple[SummaryName, ...] = (
    "count",
    "mean",
    "std",
    "median",
    "q25",
    "q75",
)
_VALID_SUMMARIES: frozenset[str] = frozenset(DEFAULT_SUMMARIES) | {"min", "max"}
_DEFAULT_DOMAINS: tuple[MetricDomain, ...] = ("topology", "intrinsic_geometry")
_DOMAIN_ORDER: tuple[str, ...] = ("topology", "intrinsic_geometry", "embedding")

# Levels whose outputs are summarised across observations.
_DISTRIBUTION_LEVELS: frozenset[str] = frozenset(
    {"node", "edge", "branch", "bifurcation", "section", "distribution", "point"}
)

# Metrics that need an external axis vector (embedding).
_AXIS_METRICS: frozenset[str] = frozenset(
    {
        "get_edge_angles",
        "get_mean_edge_angle",
        "get_edge_angle_variance",
        "coordinate_mean_along_axis",
        "coordinate_variance_along_axis",
        "coordinate_std_along_axis",
        "coordinate_minmax_along_axis",
        "coordinate_extent_along_axis",
        "coordinate_rms_along_axis",
        "coordinate_mean_absolute_along_axis",
        "coordinate_projection_moments",
    }
)

# Metrics that need neuropil/mesh / depth surfaces.
_MESH_METRICS: frozenset[str] = frozenset(
    {
        "distance_from_neuropil_surface",
        "neuropil_point_depth",
    }
)

# Explicitly unsupported in describe() (structured / display / multi-output).
_UNSUPPORTED: dict[str, str] = {
    "tree_summary": "Display SummaryTable; not a numeric descriptor.",
    "summary_table": "Mixed DataFrame summary; not a single descriptor family.",
    "forest_summary": "Population display aggregate; omitted from per-tree describe().",
    "forest_summary_table": "Population DataFrame aggregate; omit by default.",
    "get_convex_hull": "Returns SciPy ConvexHull object; use get_convex_hull_volume.",
    "fit_sphere": "Multi-output fit with mixed semantics.",
    "fit_line": "Multi-output fit with mixed semantics.",
    "fit_plane": "Multi-output fit with mixed semantics.",
    "fit_circle": "Multi-output fit with mixed semantics.",
    "get_degree_distribution": "Histogram (bins, masses); not summarised by default.",
    "get_section_angular_deviation": "Heterogeneous (means, variances) section tuple.",
}


def _metric_label(name: str) -> str:
    """Public column label for a registry metric name."""
    return name.removeprefix("get_")


def _normalize_domains(
    domains: MetricDomain | Sequence[MetricDomain] | None,
) -> tuple[MetricDomain, ...] | None:
    if domains is None:
        return None
    if isinstance(domains, str):
        items: Sequence[str] = (domains,)
    else:
        items = domains
    out: list[MetricDomain] = []
    for item in items:
        if item not in _METRIC_DOMAINS:
            raise ValueError(
                f"invalid domain {item!r}; expected one of {sorted(_METRIC_DOMAINS)}"
            )
        out.append(item)  # type: ignore[arg-type]
    # Preserve caller order but drop duplicates.
    seen: set[str] = set()
    unique: list[MetricDomain] = []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return tuple(unique)


def _normalize_metrics(metrics: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if metrics is None:
        return None
    if isinstance(metrics, str):
        names = (metrics,)
    else:
        names = tuple(metrics)
    known = {defn.name for defn in METRIC_DEFINITIONS}
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"unknown metric name(s): {unknown}")
    return names


def _normalize_summaries(
    summaries: SummaryName | Sequence[SummaryName] | None,
) -> tuple[str, ...]:
    if summaries is None:
        return DEFAULT_SUMMARIES
    if isinstance(summaries, str):
        items: Sequence[str] = (summaries,)
    else:
        items = summaries
    bad = [s for s in items if s not in _VALID_SUMMARIES]
    if bad:
        raise ValueError(
            f"invalid summary name(s) {bad}; expected subset of {sorted(_VALID_SUMMARIES)}"
        )
    # Stable order following DEFAULT_SUMMARIES then any extras (min/max).
    order = list(DEFAULT_SUMMARIES) + [s for s in ("min", "max") if s not in DEFAULT_SUMMARIES]
    selected = set(items)
    return tuple(s for s in order if s in selected)


def _coerce_anatomical_frame(reference_frame: Any) -> AnatomicalFrame | None:
    """Normalize user reference input to an :class:`AnatomicalFrame` or None.

    Accepted forms
    --------------
    * :class:`~neurosetta.AnatomicalFrame`
    * length-3 sequence / ndarray → frame with a single ``"primary"`` axis
    * Neuropil / ``_Mesh`` → frame with ``reference_mesh``
    * mapping with optional keys ``axis``, ``axes``, ``mesh`` / ``neuropil`` /
      ``reference_mesh``, ``inner_surface``, ``outer_surface``, ``name``,
      ``metadata``, ``default_axis``
    """
    if reference_frame is None:
        return None

    if isinstance(reference_frame, AnatomicalFrame):
        return reference_frame

    if isinstance(reference_frame, Mapping):
        axis = reference_frame.get("axis")
        axes = dict(reference_frame.get("axes") or {})
        if axis is not None:
            axes.setdefault("primary", axis)
        mesh = reference_frame.get(
            "reference_mesh",
            reference_frame.get("mesh", reference_frame.get("neuropil")),
        )
        return AnatomicalFrame(
            name=reference_frame.get("name"),
            reference_mesh=mesh,
            inner_surface=reference_frame.get("inner_surface"),
            outer_surface=reference_frame.get("outer_surface"),
            axes=axes or None,
            default_axis=reference_frame.get("default_axis"),
            metadata=reference_frame.get("metadata"),
        )

    if hasattr(reference_frame, "mesh") and not isinstance(
        reference_frame, (list, tuple, np.ndarray)
    ):
        return AnatomicalFrame(reference_mesh=reference_frame)

    arr = np.asarray(reference_frame, dtype=float)
    if arr.shape == (3,):
        return AnatomicalFrame(axes={"primary": arr}, default_axis="primary")

    raise TypeError(
        "reference_frame must be an AnatomicalFrame, length-3 axis, "
        "Neuropil/_Mesh-like object, or a mapping with keys such as "
        "'axis'/'axes' and 'mesh'/'reference_mesh'/'neuropil'"
    )


def _frame_can_run(frame: AnatomicalFrame | None, metric_name: str) -> bool:
    if metric_name not in METRIC_FRAME_REQUIREMENTS:
        return True
    if frame is None:
        return False
    return frame.check_metric(metric_name)


def _support_kind(defn: MetricDefinition) -> str:
    """Classify how describe() handles a metric."""
    if defn.name in _UNSUPPORTED:
        return "unsupported"
    if defn.level == "population":
        return "population"
    if defn.name == "coordinate_pca":
        return "structured_pca"
    if defn.name == "coordinate_minmax_along_axis":
        return "structured_minmax"
    if defn.name == "coordinate_projection_moments":
        return "structured_moments"
    if defn.name == "get_bifurcation_angles":
        return "numeric_distribution"
    if defn.level in _DISTRIBUTION_LEVELS:
        return "numeric_distribution"
    if defn.level == "tree":
        return "direct_scalar"
    return "unsupported"


def select_describe_definitions(
    *,
    domains: MetricDomain | Sequence[MetricDomain] | None = None,
    metrics: str | Sequence[str] | None = None,
    reference_frame: Any = None,
    include_unsupported: bool = False,
) -> list[MetricDefinition]:
    """Select registry metrics for describe().

    When both *domains* and *metrics* are given, the selection is the
    **intersection**. Default domains (when both filters are omitted) are
    topology + intrinsic_geometry. Metrics with frame requirements are
    included only when the supplied ``reference_frame`` /
    :class:`~neurosetta.AnatomicalFrame` satisfies them.
    """
    domain_filter = _normalize_domains(domains)
    metric_filter = _normalize_metrics(metrics)
    frame = _coerce_anatomical_frame(reference_frame)

    if domain_filter is None and metric_filter is None:
        domain_filter = _DEFAULT_DOMAINS

    selected: list[MetricDefinition] = []
    for defn in METRIC_DEFINITIONS:
        if domain_filter is not None and defn.domain not in domain_filter:
            continue
        if metric_filter is not None and defn.name not in metric_filter:
            continue

        kind = _support_kind(defn)
        if kind == "unsupported" and not include_unsupported:
            continue
        if kind == "population":
            continue

        if defn.requires_reference_frame and not _frame_can_run(frame, defn.name):
            continue

        if defn.tree_method in (None, "") and defn.name not in _MESH_METRICS:
            continue

        selected.append(defn)

    index = {defn.name: i for i, defn in enumerate(METRIC_DEFINITIONS)}
    domain_rank = {name: i for i, name in enumerate(_DOMAIN_ORDER)}
    selected.sort(key=lambda d: (domain_rank.get(d.domain, 99), index[d.name]))
    return selected


def _tree_unit_str(tree: _Tree) -> str:
    try:
        return str(get_units(tree))
    except Exception:
        return "dimensionless"


def _unit_for_metric(defn: MetricDefinition, tree: _Tree) -> str:
    if not defn.requires_coordinates:
        return "dimensionless"
    if defn.scale_invariant:
        return "dimensionless"
    if defn.name == "neuropil_point_depth":
        return "dimensionless"
    return _tree_unit_str(tree)


def _invoke_raw(
    defn: MetricDefinition,
    tree: _Tree,
    *,
    frame: AnatomicalFrame | None,
) -> Any:
    """Evaluate one metric on a tree."""
    name = defn.name

    if name in _MESH_METRICS:
        if frame is None:
            raise ValueError(f"{name} requires an AnatomicalFrame / reference_frame")
        frame.require_for_metric(name)
        points = get_node_coordinates(tree, SoA=False)
        if name == "distance_from_neuropil_surface":
            return frame.surface_distance(points)
        return frame.normalized_depth(points)

    if defn.tree_method in (None, ""):
        raise ValueError(f"{name} is not callable on Tree")

    method = getattr(tree, defn.tree_method)
    kwargs: dict[str, Any] = {}

    if name in _AXIS_METRICS:
        if frame is None:
            raise ValueError(f"{name} requires an AnatomicalFrame / reference_frame")
        frame.require_for_metric(name)
        axis = frame.require_axis(metric=name)
        if name in {"get_edge_angles", "get_mean_edge_angle", "get_edge_angle_variance"}:
            kwargs["between_vector"] = tuple(axis.tolist())
            if name == "get_mean_edge_angle":
                kwargs["perspective_vector"] = None
            kwargs["signed"] = False
        else:
            kwargs["axis"] = tuple(axis.tolist())

    if name in {
        "get_edge_angles",
        "get_edge_length",
        "get_subtree_scores",
        "get_partition_asymmetry",
        "get_node_depth",
        "get_radial_angle",
    }:
        kwargs["bind"] = False

    if name == "get_radial_angle":
        kwargs["signed"] = False

    if name == "get_partition_asymmetry":
        kwargs["weighted"] = True

    return method(**kwargs)


def _to_float_array(value: Any) -> np.ndarray:
    if isinstance(value, tuple):
        parts = [_to_float_array(v) for v in value]
        if not parts:
            return np.asarray([], dtype=float)
        return np.concatenate([np.ravel(p) for p in parts])
    if isinstance(value, list):
        if not value:
            return np.asarray([], dtype=float)
        return np.asarray(value, dtype=float).ravel()
    arr = np.asarray(value, dtype=float).ravel()
    return arr


def _summary_value(arr: np.ndarray, summary: str) -> float:
    if arr.size == 0:
        return 0.0 if summary == "count" else float("nan")
    if summary == "count":
        return float(arr.size)
    if summary == "mean":
        return float(np.mean(arr))
    if summary == "std":
        return float(np.std(arr))
    if summary == "median":
        return float(np.median(arr))
    if summary == "min":
        return float(np.min(arr))
    if summary == "max":
        return float(np.max(arr))
    if summary == "q25":
        return float(np.quantile(arr, 0.25))
    if summary == "q75":
        return float(np.quantile(arr, 0.75))
    raise ValueError(f"unknown summary {summary!r}")


def _descriptor_items(
    defn: MetricDefinition,
    raw: Any,
    summaries: Sequence[str],
    tree: _Tree,
) -> list[tuple[str, str | None, float, str]]:
    """Return ``(metric_label, summary_or_None, value, unit)`` tuples."""
    label = _metric_label(defn.name)
    unit = _unit_for_metric(defn, tree)
    kind = _support_kind(defn)

    if kind == "direct_scalar":
        val = float(np.asarray(raw).reshape(()))
        return [(label, None, val, unit)]

    if kind == "numeric_distribution":
        arr = _to_float_array(raw)
        return [
            (label, summary, _summary_value(arr, summary), unit) for summary in summaries
        ]

    if kind == "structured_pca":
        evals, _evecs = raw
        evals = np.asarray(evals, dtype=float).ravel()
        return [
            (f"{label}.eigenvalue_{i + 1}", None, float(evals[i]), "dimensionless")
            for i in range(len(evals))
        ]

    if kind == "structured_minmax":
        vmin, vmax = raw
        return [
            (f"{label}.min", None, float(vmin), unit),
            (f"{label}.max", None, float(vmax), unit),
        ]

    if kind == "structured_moments":
        mean, var, std, vmin, vmax = raw
        return [
            (f"{label}.mean", None, float(mean), unit),
            (f"{label}.variance", None, float(var), unit),
            (f"{label}.std", None, float(std), unit),
            (f"{label}.min", None, float(vmin), unit),
            (f"{label}.max", None, float(vmax), unit),
        ]

    raise ValueError(f"metric {defn.name!r} is unsupported by describe()")


def _column_name(domain: str, metric_label: str, summary: str | None) -> str:
    if summary is None:
        return f"{domain}.{metric_label}"
    return f"{domain}.{metric_label}.{summary}"


def _describe_tree_records(
    tree: _Tree,
    definitions: Sequence[MetricDefinition],
    summaries: Sequence[str],
    *,
    frame: AnatomicalFrame | None,
    errors: DescribeErrors,
) -> list[dict[str, Any]]:
    """Long-form records for one tree."""
    records: list[dict[str, Any]] = []
    neuron_id = tree.ID

    if frame is not None:
        try:
            frame.check_units_compatible(_tree_unit_str(tree))
        except ValueError as exc:
            if errors == "raise":
                raise
            if errors == "warn":
                warnings.warn(str(exc), stacklevel=3)

    for defn in definitions:
        try:
            if defn.name in _UNSUPPORTED:
                raise ValueError(_UNSUPPORTED[defn.name])
            if defn.requires_reference_frame and not _frame_can_run(frame, defn.name):
                raise FrameRequirementError(
                    f"metric {defn.name!r} requires a suitable AnatomicalFrame / "
                    f"reference_frame; got {frame!r}",
                    metric=defn.name,
                    frame=frame,
                )

            raw = _invoke_raw(defn, tree, frame=frame)
            items = _descriptor_items(defn, raw, summaries, tree)
        except Exception as exc:
            if errors == "raise":
                raise RuntimeError(
                    f"describe failed for neuron_id={neuron_id!r}, metric={defn.name!r}: {exc}"
                ) from exc
            if errors == "warn":
                warnings.warn(
                    f"describe: neuron_id={neuron_id!r}, metric={defn.name!r}: {exc}",
                    stacklevel=3,
                )
            if errors in {"warn", "ignore"}:
                kind = _support_kind(defn)
                label = _metric_label(defn.name)
                unit = _unit_for_metric(defn, tree)
                if kind == "numeric_distribution":
                    items = [(label, s, float("nan"), unit) for s in summaries]
                elif kind == "structured_pca":
                    items = [
                        (f"{label}.eigenvalue_{i}", None, float("nan"), "dimensionless")
                        for i in range(1, 4)
                    ]
                else:
                    items = [(label, None, float("nan"), unit)]
            else:
                raise

        for metric_label, summary, value, unit in items:
            records.append(
                {
                    "neuron_id": neuron_id,
                    "domain": defn.domain,
                    "metric": metric_label,
                    "summary": summary,
                    "value": value,
                    "unit": unit,
                }
            )
    return records


def _records_to_wide(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["neuron_id"])

    # Preserve first-seen descriptor column order.
    col_order: list[str] = []
    seen: set[str] = set()
    rows: dict[Any, dict[str, Any]] = {}
    neuron_order: list[Any] = []

    for rec in records:
        nid = rec["neuron_id"]
        if nid not in rows:
            rows[nid] = {"neuron_id": nid}
            neuron_order.append(nid)
        col = _column_name(rec["domain"], rec["metric"], rec["summary"])
        if col not in seen:
            seen.add(col)
            col_order.append(col)
        rows[nid][col] = rec["value"]

    frame = pd.DataFrame([rows[nid] for nid in neuron_order])
    return frame.loc[:, ["neuron_id", *col_order]]


def _append_metadata(
    wide: pd.DataFrame,
    obj: _Tree | _Forest,
    keys: Sequence[str] | None,
) -> pd.DataFrame:
    if keys is None:
        # Conservative defaults: only stable, flat keys when present.
        keys = ("units",)
    trees: Sequence[_Tree]
    if isinstance(obj, (_Tree, Tree)):
        trees = (obj,)
    else:
        trees = list(obj)

    meta_cols: dict[str, list[Any]] = {f"meta.{key}": [] for key in keys}
    for tree in trees:
        for key in keys:
            meta_cols[f"meta.{key}"].append(tree.metadata.get(key))
    for col, values in meta_cols.items():
        wide[col] = values
    return wide


def describe(
    obj: Tree | Forest | _Tree | _Forest,
    *,
    domains: MetricDomain | Sequence[MetricDomain] | None = None,
    metrics: str | Sequence[str] | None = None,
    summaries: SummaryName | Sequence[SummaryName] | None = None,
    output: DescribeOutput = "wide",
    include_metadata: bool = False,
    metadata_keys: Sequence[str] | None = None,
    reference_frame: Any = None,
    errors: DescribeErrors = "raise",
    parallel: bool | None = None,
) -> pd.DataFrame:
    """Compute a tidy table of morphology descriptors for a Tree or Forest.

    This is an orchestration layer over the metric registry. It does not add
    new morphometric algorithms. Array-valued metrics are converted into
    explicitly named summary descriptors.

    Parameters
    ----------
    obj : Tree or Forest
        Morphology to describe.
    domains : str or sequence of str, optional
        Morphology domains to include (``topology``, ``intrinsic_geometry``,
        ``embedding``). When both *domains* and *metrics* are given, the
        selection is their **intersection**. When both are omitted, defaults
        to topology + intrinsic geometry (no embedding without a reference).
    metrics : str or sequence of str, optional
        Registered metric names to include (intersection with *domains* when
        both are set).
    summaries : str or sequence of str, optional
        Summaries for distribution-valued metrics. Default:
        ``count``, ``mean``, ``std``, ``median``, ``q25``, ``q75``.
    output : {"wide", "long"}, optional
        Wide: one row per neuron, columns ``<domain>.<metric>[.<summary>]``.
        Long: columns ``neuron_id``, ``domain``, ``metric``, ``summary``,
        ``value``, ``unit``.
    include_metadata : bool, optional
        If True, append selected metadata columns (``meta.<key>``) in wide
        mode. By default False.
    metadata_keys : sequence of str, optional
        Metadata keys to include when *include_metadata* is True. Default
        ``("units",)``.
    reference_frame : AnatomicalFrame, axis, Neuropil/_Mesh, or mapping, optional
        External reference for embedding metrics. Prefer
        :class:`~neurosetta.AnatomicalFrame`. Legacy forms (length-3 axis,
        mesh-like object, or ``{"axis": ..., "mesh": ...}``) are coerced to a
        frame. When ``domains`` explicitly includes ``"embedding"`` and no
        usable frame is supplied, an error is raised.
    errors : {"raise", "warn", "ignore"}, optional
        Failure policy. Default ``raise``.
    parallel : bool or None, optional
        Forwarded to :meth:`Forest.apply` when *obj* is a Forest.

    Returns
    -------
    pandas.DataFrame
        Descriptor table.

    Notes
    -----
    * Metric values are numeric. Spatial units are reported in long-form
      ``unit`` (from tree metadata for length-like metrics; ``dimensionless``
      for topology/angles/ratios/normalized depth). Wide output stores bare
      floats in those tree units.
    * Population-level registry entries are omitted from the per-tree table.
    * Metrics that return unstructured objects (hulls, fits, display tables)
      are excluded from the default selection.
    """
    if output not in {"wide", "long"}:
        raise ValueError("output must be 'wide' or 'long'")
    if errors not in {"raise", "warn", "ignore"}:
        raise ValueError("errors must be 'raise', 'warn', or 'ignore'")

    summary_list = _normalize_summaries(summaries)
    frame = _coerce_anatomical_frame(reference_frame)
    domain_filter = _normalize_domains(domains)
    metric_filter = _normalize_metrics(metrics)
    embedding_requested = domain_filter is not None and "embedding" in domain_filter

    if embedding_requested and frame is None:
        raise ValueError(
            "domains includes 'embedding' but no reference_frame / AnatomicalFrame "
            "was supplied"
        )

    # Explicit request for unsupported / missing-reference metrics → error.
    if metric_filter is not None:
        for name in metric_filter:
            if name in _UNSUPPORTED:
                raise ValueError(
                    f"metric {name!r} is unsupported by describe(): {_UNSUPPORTED[name]}"
                )
            defn = next(d for d in METRIC_DEFINITIONS if d.name == name)
            if defn.level == "population":
                raise ValueError(
                    f"metric {name!r} is population-level and is not included in "
                    "per-tree describe() output"
                )
            if defn.requires_reference_frame and not _frame_can_run(frame, name):
                if frame is None:
                    raise ValueError(
                        f"metric {name!r} requires reference_frame / AnatomicalFrame"
                    )
                frame.require_for_metric(name)

    definitions = select_describe_definitions(
        domains=domains,
        metrics=metrics,
        reference_frame=frame,
    )

    if embedding_requested and not any(d.domain == "embedding" for d in definitions):
        avail = list(frame.available_components()) if frame is not None else []
        raise ValueError(
            "domains includes 'embedding' but no embedding metrics can be evaluated "
            f"with the supplied frame (available components={avail}). "
            "Provide reference_mesh and/or axes as required."
        )

    if isinstance(obj, (_Tree, Tree)):
        trees: list[_Tree] = [obj]
        apply_forest = None
    elif isinstance(obj, (_Forest, Forest)):
        trees = list(obj)
        apply_forest = obj
    else:
        raise TypeError("obj must be a Tree or Forest")

    if not trees:
        empty_long = pd.DataFrame(
            columns=["neuron_id", "domain", "metric", "summary", "value", "unit"]
        )
        return empty_long if output == "long" else pd.DataFrame(columns=["neuron_id"])

    def _one(tree: _Tree) -> list[dict[str, Any]]:
        return _describe_tree_records(
            tree,
            definitions,
            summary_list,
            frame=frame,
            errors=errors,
        )

    if apply_forest is not None:
        per_tree = apply_forest.apply(_one, parallel=parallel)
    else:
        per_tree = [_one(trees[0])]

    records: list[dict[str, Any]] = []
    for chunk in per_tree:
        records.extend(chunk)

    long_df = pd.DataFrame.from_records(
        records,
        columns=["neuron_id", "domain", "metric", "summary", "value", "unit"],
    )

    if output == "long":
        return long_df

    wide = _records_to_wide(records)
    if include_metadata:
        wide = _append_metadata(wide, obj, metadata_keys)
    return wide


__all__ = [
    "describe",
    "select_describe_definitions",
    "DEFAULT_SUMMARIES",
]
