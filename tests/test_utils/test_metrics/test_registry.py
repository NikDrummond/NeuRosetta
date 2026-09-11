"""Tests for documented metric catalog."""

import pytest

import pandas as pd

from neurosetta.utils.metrics.registry import (
    METRIC_DEFINITIONS,
    MetricDefinition,
    _METRIC_DOMAINS,
    _METRIC_LEVELS,
    format_metrics_classification_markdown,
    format_metrics_reference_markdown,
    format_metrics_reference_table,
    list_metric_definitions,
)


def test_list_metric_definitions_matches_registry():
    assert list_metric_definitions() == METRIC_DEFINITIONS
    assert len(METRIC_DEFINITIONS) >= 40


def test_format_metrics_reference_table_dataframe():
    table = format_metrics_reference_table()
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == [
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
    ]
    assert len(table) == len(METRIC_DEFINITIONS)
    nodes = table.loc[table["metric"] == "count_nodes"].iloc[0]
    assert nodes["tree"] == "tree.count_nodes()"
    assert nodes["forest"] == "forest.count_nodes()"
    assert nodes["domain"] == "topology"
    assert nodes["api"] == "counting"
    assert nodes["translation_invariant"]
    assert not nodes["requires_coordinates"]
    neuropil = table.loc[table["metric"] == "distance_from_neuropil_surface"].iloc[0]
    assert neuropil["tree"] == "nr.distance_from_neuropil_surface()"
    assert neuropil["domain"] == "embedding"
    assert neuropil["requires_reference_frame"]


def test_format_metrics_reference_markdown():
    table = format_metrics_reference_markdown()
    assert "| Category | Metric | Tree | Forest | API | Notes |" in table
    assert "``count_nodes``" in table
    assert "``tree.count_nodes()``" in table
    assert "``forest.count_nodes()``" in table
    assert "{doc}`../api/tree_ops/counting`" in table
    assert "``nr.distance_from_neuropil_surface()``" in table


def test_format_metrics_classification_markdown():
    table = format_metrics_classification_markdown()
    assert "| Metric | Category | Domain | Level | T | R | S | Coords | Ref |" in table
    assert "``count_nodes``" in table
    assert "``topology``" in table
    assert "``intrinsic_geometry``" in table
    assert "``embedding``" in table
    assert len(table.splitlines()) == len(METRIC_DEFINITIONS) + 2


def test_metric_names_are_unique():
    names = [item.name for item in METRIC_DEFINITIONS]
    assert len(names) == len(set(names))


def test_metric_definition_stores_scientific_metadata():
    metric = MetricDefinition(
        "example_metric",
        "Geometry",
        "tree_geometry",
        tree_method="example_metric",
        domain="intrinsic_geometry",
        level="bifurcation",
        translation_invariant=True,
        rotation_invariant=True,
        scale_invariant=True,
        requires_coordinates=True,
        requires_reference_frame=False,
    )
    assert metric.domain == "intrinsic_geometry"
    assert metric.level == "bifurcation"
    assert metric.translation_invariant is True
    assert metric.rotation_invariant is True
    assert metric.scale_invariant is True
    assert metric.requires_coordinates is True
    assert metric.requires_reference_frame is False


def test_metric_definition_rejects_invalid_domain_and_level():
    with pytest.raises(ValueError, match="invalid domain"):
        MetricDefinition(
            "bad_domain",
            "Counting",
            "counting",
            tree_method="bad_domain",
            domain="not_a_domain",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="invalid level"):
        MetricDefinition(
            "bad_level",
            "Counting",
            "counting",
            tree_method="bad_level",
            level="not_a_level",  # type: ignore[arg-type]
        )


def test_metric_definition_requires_coordinates_for_reference_frame():
    with pytest.raises(ValueError, match="requires_reference_frame"):
        MetricDefinition(
            "bad_requirements",
            "Geometry",
            "tree_geometry",
            tree_method="bad_requirements",
            domain="embedding",
            level="tree",
            requires_coordinates=False,
            requires_reference_frame=True,
        )


def test_registry_scientific_metadata_integrity():
    for metric in METRIC_DEFINITIONS:
        assert metric.domain is not None
        assert metric.level is not None
        assert metric.domain in _METRIC_DOMAINS
        assert metric.level in _METRIC_LEVELS
        assert isinstance(metric.translation_invariant, bool)
        assert isinstance(metric.rotation_invariant, bool)
        assert isinstance(metric.scale_invariant, bool)
        assert isinstance(metric.requires_coordinates, bool)
        assert isinstance(metric.requires_reference_frame, bool)
        if metric.requires_reference_frame:
            assert metric.requires_coordinates is True
        if metric.domain == "topology":
            assert metric.requires_coordinates is False
            assert metric.requires_reference_frame is False
            assert metric.translation_invariant is True
            assert metric.rotation_invariant is True
            assert metric.scale_invariant is True
        if metric.domain == "embedding":
            assert metric.requires_coordinates is True
            assert metric.requires_reference_frame is True
        if metric.domain == "intrinsic_geometry":
            assert metric.requires_coordinates is True
            assert metric.requires_reference_frame is False


def test_list_metric_definitions_filters_by_domain_and_level():
    topology = list_metric_definitions(domain="topology")
    assert topology
    assert all(item.domain == "topology" for item in topology)

    intrinsic = list_metric_definitions(domain="intrinsic_geometry")
    assert intrinsic
    assert all(item.domain == "intrinsic_geometry" for item in intrinsic)

    embedding = list_metric_definitions(domain="embedding")
    assert embedding
    assert all(item.domain == "embedding" for item in embedding)

    # Every registered metric is classified into exactly one domain filter.
    assert len(topology) + len(intrinsic) + len(embedding) == len(METRIC_DEFINITIONS)

    bifurcations = list_metric_definitions(level="bifurcation")
    assert bifurcations
    assert all(item.level == "bifurcation" for item in bifurcations)

    composed = list_metric_definitions(domain="topology", level="node")
    assert composed
    assert all(item.domain == "topology" and item.level == "node" for item in composed)


def test_list_metric_definitions_rejects_invalid_filters():
    with pytest.raises(ValueError, match="invalid domain"):
        list_metric_definitions(domain="not_a_domain")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid level"):
        list_metric_definitions(level="not_a_level")  # type: ignore[arg-type]
