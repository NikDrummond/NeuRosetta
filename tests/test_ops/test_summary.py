"""Tests for Tree.summary and Forest.summary."""

from neurosetta.api import Forest, Tree
from neurosetta.ops.tree_graphs.summary import SummaryTable, _summary_row


def test_summary_row_values(simple_tree):
    tree = Tree(
        ID=42,
        metadata={"isReduced": True, "Flag": True, "units": "nm"},
        graph=simple_tree,
    )
    row = _summary_row(tree)

    assert row["ID"] == 42
    assert row["nodes"] == tree.count_nodes()
    assert row["branches"] == tree.count_branches()
    assert row["leaves"] == tree.count_leaves()
    assert row["cable"] == tree.get_total_cable_length()
    assert row["units"] == tree.get_units()
    assert row["isReduced"] is True
    assert row["Flag"] is True


def test_tree_summary_contains_core_fields(simple_tree):
    tree = Tree(ID=1, metadata={"Flag": False, "isReduced": False}, graph=simple_tree)
    summary = tree.summary()

    assert isinstance(summary, SummaryTable)
    assert summary.title == "Tree summary"
    text = str(summary)
    assert "ID" in text
    assert "nodes" in text
    assert "branches" in text
    assert "leaves" in text
    assert "cable" in text
    assert "isReduced" in text
    assert "Flag" in text
    assert "1" in text
    assert "<table" in summary._repr_html_()


def test_tree_summary_table_matches_row(simple_tree):
    tree = Tree(ID=7, metadata={}, graph=simple_tree)
    frame = tree.summary_table()
    row = frame.iloc[0]

    assert len(frame) == 1
    assert row["ID"] == 7
    assert row["nodes"] == tree.count_nodes()


def test_forest_summary_rows_and_total(simple_tree):
    t1 = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    t2 = t1.copy()
    t2.ID = 2
    forest = Forest([t1, t2])

    summary = forest.summary()
    table = forest.summary_table()

    assert isinstance(summary, SummaryTable)
    assert summary.title == "Forest summary (2 trees)"
    text = str(summary)
    assert "TOTAL" in text
    assert "MEAN" in text
    assert len(table) == 2
    assert table["nodes"].sum() == t1.count_nodes() + t2.count_nodes()
    assert len(summary.dataframe) == 4


def test_forest_summary_empty():
    forest = Forest([])
    summary = forest.summary()
    assert isinstance(summary, SummaryTable)
    assert summary.title == "Forest summary (0 trees)"
    assert summary.dataframe.empty


def test_forest_summary_skips_cable_total_for_mixed_units(simple_tree):
    t1 = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    t2 = t1.copy()
    t2.ID = 2
    t2.set_units("um")
    forest = Forest([t1, t2])

    text = str(forest.summary())
    assert "TOTAL" in text
    assert "—" in text
