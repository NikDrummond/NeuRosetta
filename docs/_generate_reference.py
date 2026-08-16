"""Regenerate reference docs from Python registries.

Run from the repository root::

    python docs/_generate_reference.py

Or via ``make html`` in ``docs/`` (runs automatically before Sphinx).
"""

from __future__ import annotations

from pathlib import Path

from neurosetta.utils.metrics import format_metrics_reference_table

DOCS = Path(__file__).resolve().parent

METRICS_HEADER = """\
# Metrics overview

NeuRosetta exposes **descriptive metrics** — scalar summaries and per-element
measurements of topology, geometry, and spatial extent. They live in
{doc}`../api/tree_ops/index` (tree/forest ops) and {doc}`../api/neuropils`
(neuropil distance helpers).

This page is a catalog of what exists today. Detailed signatures and docstrings
are in the linked API pages (autodoc, always current).

## Entry points

| You have… | Prefer… |
| --- | --- |
| One tree, notebooks | {class}`~neurosetta.api.Tree` methods — e.g. ``tree.count_nodes()`` |
| Many trees | {class}`~neurosetta.api.Forest` batch methods — e.g.
  ``forest.get_total_cable_length()`` |
| Pipelines / scripts | Module functions — e.g. ``nr.count_nodes(tree)`` |

``Tree.some_method()`` and ``nr.some_method(tree)`` are interchangeable. See
{doc}`../getting_started/overview`.

## Notes

- **Reduced trees:** cable-length metrics use bound edge properties; compute
  ``Path_length`` before reducing if you need lengths along the full arbor.
  See {doc}`../tutorials/tree_surgery`.
- **Units:** length and coordinate metrics require defined spatial units.
  See {doc}`units`.
- **Not listed here:** node/edge index lookups, traversals, editing, coordinate
  transforms, and plotting — those are operations, not descriptives.

<!-- AUTO-GENERATED: table below; run ``python docs/_generate_reference.py`` -->

## Quick reference

"""

METRICS_FOOTER = """

## Programmatic reference

```python
import neurosetta as nr

for metric in nr.list_metric_definitions():
    print(metric.category, metric.name, metric.tree_method, metric.forest_method)

print(nr.format_metrics_reference_table())
```
"""


def write_metrics_reference() -> Path:
    """Write ``reference/metrics.md`` from the metric registry."""
    path = DOCS / "reference" / "metrics.md"
    body = METRICS_HEADER + format_metrics_reference_table() + METRICS_FOOTER
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    path = write_metrics_reference()
    print(f"wrote {path.relative_to(DOCS)}")


if __name__ == "__main__":
    main()
