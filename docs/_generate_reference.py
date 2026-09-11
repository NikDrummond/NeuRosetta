"""Regenerate reference docs from Python registries.

Run from the repository root::

    python docs/_generate_reference.py

Or via ``make html`` in ``docs/`` (runs automatically before Sphinx).
"""

from __future__ import annotations

from pathlib import Path

from neurosetta.utils.metrics import (
    format_metrics_classification_markdown,
    format_metrics_reference_markdown,
)

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

METRICS_FOOTER_PREFIX = """

## Morphology descriptors

Morphology measurements in NeuRosetta can be grouped into:

1. **topology** — how a neuron branches (connectivity);
2. **intrinsic geometry** — the shape of the arbor independent of anatomical placement;
3. **embedding** — how that arbor is situated within an anatomical context.

An :class:`~neurosetta.AnatomicalFrame` provides that external context
(reference mesh, optional surface pair, named axes). It does **not** replace
existing Neuropil/mesh classes or low-level distance/depth functions.

:func:`neurosetta.describe` builds a standardized table from the metric
registry. Array-valued registry metrics are turned into explicitly named
summary descriptors (mean / std / quartiles).

```python
import neurosetta as nr

df = nr.describe(tree)                 # wide, one row
df = nr.describe(forest)               # wide, one row per tree
df = nr.describe(forest, domains="topology")
df = nr.describe(forest, output="long")

frame = nr.AnatomicalFrame(
    name="example_neuropil",
    reference_mesh=neuropil,
    axes={"depth": (0, 0, 1)},
)
embedding = nr.describe(
    forest,
    domains="embedding",
    reference_frame=frame,
)
# equivalent: frame.describe(forest)
```

Low-level mesh/axis arguments remain valid::

    nr.distance_from_neuropil_surface(neuropil, points)
    nr.neuropil_point_depth(neuropil, points, t=0.0)
    tree.coordinate_extent_along_axis((0, 0, 1))

Default ``describe`` selection is topology + intrinsic geometry. Embedding
requires a usable ``reference_frame``. When both ``domains`` and ``metrics``
are given, the selection is their intersection.

## Scientific metadata

Each catalog entry carries morphology metadata independent of ``category``.

| Field | Meaning |
| --- | --- |
| ``domain`` | Scientific morphology domain |
| ``level`` | Natural level of the returned measurement |
| ``translation_invariant`` | Unchanged by global translation |
| ``rotation_invariant`` | Unchanged by global rotation |
| ``scale_invariant`` | Numerically unchanged by uniform scaling |
| ``requires_coordinates`` | Requires spatial coordinates |
| ``requires_reference_frame`` | Requires external anatomical/spatial reference |

### Domains

| Domain | Meaning |
| --- | --- |
| ``topology`` | Connectivity independent of coordinates |
| ``intrinsic_geometry`` | Spatial properties intrinsic to the neuron and independent of its anatomical placement |
| ``embedding`` | Properties describing the neuron's relationship to an external spatial or anatomical frame |

Invariance metadata describes transforming the neuron while keeping any
external reference frame fixed.

Filter with ``nr.list_metric_definitions(domain=...)`` and/or ``level=...``.

Use :func:`neurosetta.describe` for a standardized per-neuron descriptor table
over topology and intrinsic geometry (and embedding when a reference frame is
supplied).

### Classification review

Generated from the registry (source of truth). Columns ``T`` / ``R`` / ``S``
are translation / rotation / scale invariance; ``Coords`` / ``Ref`` are
``requires_coordinates`` / ``requires_reference_frame``.

"""

METRICS_FOOTER_SUFFIX = """

## Programmatic reference

```python
import neurosetta as nr

for metric in nr.list_metric_definitions(domain="topology"):
    print(metric.domain, metric.level, metric.name)

nr.format_metrics_reference_table()  # DataFrame with classification columns

# High-level descriptors (topology + intrinsic geometry by default)
df = nr.describe(tree)
df = nr.describe(forest, domains=["topology", "intrinsic_geometry"])
df = nr.describe(forest, output="long")
```
"""


def write_metrics_reference() -> Path:
    """Write ``reference/metrics.md`` from the metric registry."""
    path = DOCS / "reference" / "metrics.md"
    body = (
        METRICS_HEADER
        + format_metrics_reference_markdown()
        + METRICS_FOOTER_PREFIX
        + format_metrics_classification_markdown()
        + METRICS_FOOTER_SUFFIX
    )
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    path = write_metrics_reference()
    print(f"wrote {path.relative_to(DOCS)}")


if __name__ == "__main__":
    main()
