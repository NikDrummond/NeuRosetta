"""Forest batch API alignment with global settings."""

from __future__ import annotations

from graph_tool.all import Graph

from neurosetta.api import Forest, Tree
from neurosetta.config import configure
from neurosetta.config.settings import _reset_settings_for_tests
from neurosetta.core.forest import _Forest


def _make_forest():
    g = Graph(directed=False)
    g.add_vertex(1)
    t1 = Tree(ID=1, metadata={}, graph=g)
    t2 = Tree(ID=2, metadata={}, graph=g)
    return Forest([t1, t2])


def test_forest_op_passes_unset_execution_kwargs(monkeypatch):
    captured: list[dict] = []
    original_apply = _Forest.apply

    def spy_apply(self, fn, *args, **kwargs):
        captured.append(kwargs)
        return original_apply(self, fn, *args, **kwargs)

    monkeypatch.setattr(_Forest, "apply", spy_apply)

    _reset_settings_for_tests()
    _make_forest().count_nodes()

    assert captured[-1]["parallel"] is None
    assert captured[-1]["max_workers"] is None
    assert captured[-1]["show_progress"] is None


def test_forest_op_explicit_kwargs_override_config():
    _reset_settings_for_tests()
    configure(parallel_forest=False, show_progress=True, max_workers=2)

    forest = _make_forest()
    results = forest.count_nodes(
        parallel=True,
        max_workers=1,
        show_progress=False,
    )

    assert results == [1, 1]
