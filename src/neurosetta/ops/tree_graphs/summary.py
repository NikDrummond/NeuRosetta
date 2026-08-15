"""Formatted summary tables for neuron trees."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from ...core import _Tree
from .._doc_helpers import enrich_tree_graph_docstrings
from ..units import get_units
from .counting import count_branches, count_leaves, count_nodes
from .path_lengths import get_total_cable_length

SUMMARY_COLUMNS = ("ID", "nodes", "branches", "leaves", "cable", "units", "isReduced", "Flag")
NUMERIC_SUMMARY_COLUMNS = ("nodes", "branches", "leaves", "cable")


class SummaryTable:
    """Display-friendly summary table for trees and forests.

    In notebooks the table renders as HTML. Use ``str(summary)`` or ``print(summary)``
    for plain-text output, or ``summary.dataframe`` for the underlying table.
    """

    __slots__ = ("title", "frame", "style")

    def __init__(
        self,
        title: str,
        frame: pd.DataFrame,
        *,
        style: Literal["plain", "markdown"] = "plain",
    ) -> None:
        self.title = title
        self.frame = frame
        self.style = style

    @property
    def dataframe(self) -> pd.DataFrame:
        """Formatted summary table."""
        return self.frame

    def to_string(self) -> str:
        """Plain-text table with title."""
        return f"{self.title}\n{self.frame.to_string(index=False)}"

    def to_markdown(self) -> str:
        """Markdown table with title."""
        try:
            body = self.frame.to_markdown(index=False)
        except ImportError:
            header = "| " + " | ".join(self.frame.columns) + " |"
            separator = "| " + " | ".join("---" for _ in self.frame.columns) + " |"
            rows = [
                "| " + " | ".join(str(row[col]) for col in self.frame.columns) + " |"
                for _, row in self.frame.iterrows()
            ]
            body = "\n".join((header, separator, *rows))
        return f"**{self.title}**\n\n{body}"

    def _repr_html_(self) -> str:
        table = self.frame.to_html(index=False, border=0)
        return (
            "<div>"
            f'<p style="margin:0 0 0.5em 0;"><strong>{self.title}</strong></p>'
            f"{table}"
            "</div>"
        )

    def _repr_markdown_(self) -> str:
        return self.to_markdown()

    def __str__(self) -> str:
        if self.style == "markdown":
            return self.to_markdown()
        return self.to_string()

    def __repr__(self) -> str:
        return f"SummaryTable({self.title!r}, rows={len(self.frame)})"


def _summary_row(tree: _Tree) -> dict:
    meta = tree.metadata
    return {
        "ID": tree.ID,
        "nodes": count_nodes(tree),
        "branches": count_branches(tree),
        "leaves": count_leaves(tree),
        "cable": get_total_cable_length(tree),
        "units": get_units(tree),
        "isReduced": bool(meta.get("isReduced", False)),
        "Flag": bool(meta.get("Flag", False)),
    }


def _format_display_value(column: str, value) -> str:
    if column in ("isReduced", "Flag"):
        return str(value)
    if column == "units":
        return str(value)
    if column == "ID":
        return str(value)
    if value == "":
        return ""
    if value is None:
        return "—"
    if column == "cable" and isinstance(value, (int, float)):
        return f"{value:.1f}"
    if column in NUMERIC_SUMMARY_COLUMNS and isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.1f}"
    return str(value)


def _format_display_row(row: dict) -> dict:
    return {key: _format_display_value(key, value) for key, value in row.items()}


def _aggregate_rows(
    rows: list[dict],
    *,
    aggregate: Literal["total", "mean", "both", "none"],
) -> list[dict]:
    if not rows or aggregate == "none":
        return []

    units = {row["units"] for row in rows}
    can_sum_cable = len(units) == 1
    n = len(rows)
    extra: list[dict] = []

    def _total_row() -> dict:
        return {
            "ID": "TOTAL",
            "nodes": sum(row["nodes"] for row in rows),
            "branches": sum(row["branches"] for row in rows),
            "leaves": sum(row["leaves"] for row in rows),
            "cable": sum(row["cable"] for row in rows) if can_sum_cable else None,
            "units": next(iter(units)) if can_sum_cable else "",
            "isReduced": "",
            "Flag": "",
        }

    def _mean_row() -> dict:
        return {
            "ID": "MEAN",
            "nodes": sum(row["nodes"] for row in rows) / n,
            "branches": sum(row["branches"] for row in rows) / n,
            "leaves": sum(row["leaves"] for row in rows) / n,
            "cable": sum(row["cable"] for row in rows) / n if can_sum_cable else None,
            "units": next(iter(units)) if can_sum_cable else "",
            "isReduced": "",
            "Flag": "",
        }

    if aggregate in ("total", "mean", "both"):
        extra.append(_total_row())
    if aggregate in ("mean", "both"):
        extra.append(_mean_row())
    return extra


def _build_summary_table(
    rows: list[dict],
    *,
    title: str,
    aggregate: Literal["total", "mean", "both", "none"] = "none",
    style: Literal["plain", "markdown"] = "plain",
) -> SummaryTable:
    display_rows = [_format_display_row(row) for row in rows]
    display_rows.extend(_format_display_row(row) for row in _aggregate_rows(rows, aggregate=aggregate))
    frame = pd.DataFrame(display_rows, columns=list(SUMMARY_COLUMNS))
    return SummaryTable(title, frame, style=style)


def tree_summary(
    tree: _Tree,
    *,
    style: Literal["plain", "markdown"] = "plain",
) -> SummaryTable:
    """Return a formatted summary table for a single tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    style : {"plain", "markdown"}, optional
        Plain-text format when using ``print(summary)``. Notebook display
        always uses an HTML table. By default ``"plain"``.

    Returns
    -------
    SummaryTable
        Display-friendly summary table. Evaluate in a notebook cell to render
        HTML, or use ``summary.dataframe`` for the underlying table.
    """
    return _build_summary_table(
        [_summary_row(tree)],
        title="Tree summary",
        aggregate="none",
        style=style,
    )


def summary_table(tree: _Tree) -> pd.DataFrame:
    """Return raw numeric summary metrics for a tree as a one-row DataFrame."""
    return pd.DataFrame([_summary_row(tree)], columns=list(SUMMARY_COLUMNS))


enrich_tree_graph_docstrings(globals())
