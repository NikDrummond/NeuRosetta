"""Formatted summary tables for neuron forests."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from ...core import _Forest
from ..tree_graphs.summary import (
    SUMMARY_COLUMNS,
    SummaryTable,
    _build_summary_table,
    _summary_row,
)


def forest_summary_table(forest: _Forest) -> pd.DataFrame:
    """Return raw numeric per-tree summary metrics as a DataFrame."""
    rows = [_summary_row(tree) for tree in forest]
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def forest_summary(
    forest: _Forest,
    *,
    aggregate: Literal["total", "mean", "both", "none"] = "both",
    style: Literal["plain", "markdown"] = "plain",
) -> SummaryTable:
    """Return a formatted summary table for every tree in a forest.

    Parameters
    ----------
    forest : _Forest
        Container of neuron trees.
    aggregate : {"total", "mean", "both", "none"}, optional
        Append aggregate rows after the per-tree rows. By default ``"both"``.
    style : {"plain", "markdown"}, optional
        Plain-text format when using ``print(summary)``. Notebook display
        always uses an HTML table. By default ``"plain"``.

    Returns
    -------
    SummaryTable
        Display-friendly summary table. Evaluate in a notebook cell to render
        HTML, or use ``summary.dataframe`` for the underlying table.
    """
    n = len(forest)
    if n == 0:
        return SummaryTable(
            "Forest summary (0 trees)",
            pd.DataFrame(columns=list(SUMMARY_COLUMNS)),
            style=style,
        )

    rows = [_summary_row(tree) for tree in forest]
    return _build_summary_table(
        rows,
        title=f"Forest summary ({n} trees)",
        aggregate=aggregate,
        style=style,
    )
