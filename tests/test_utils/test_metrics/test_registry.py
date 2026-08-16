"""Tests for documented metric catalog."""

from neurosetta.utils.metrics.registry import (
    METRIC_DEFINITIONS,
    format_metrics_reference_table,
    list_metric_definitions,
)


def test_list_metric_definitions_matches_registry():
    assert list_metric_definitions() == METRIC_DEFINITIONS
    assert len(METRIC_DEFINITIONS) >= 40


def test_format_metrics_reference_table_markdown():
    table = format_metrics_reference_table()
    assert "| Category | Metric | Tree | Forest | API | Notes |" in table
    assert "``count_nodes``" in table
    assert "``tree.count_nodes()``" in table
    assert "``forest.count_nodes()``" in table
    assert "{doc}`../api/tree_ops/counting`" in table
    assert "``nr.distance_from_neuropil_surface()``" in table


def test_metric_names_are_unique():
    names = [item.name for item in METRIC_DEFINITIONS]
    assert len(names) == len(set(names))
