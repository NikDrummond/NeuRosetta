"""One-off helper to (re)build tutorial notebooks. Not part of the docs build."""

from __future__ import annotations

import json
from pathlib import Path

DOCS = Path(__file__).resolve().parent
EXAMPLE_ID = 720575940596125868


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb(cells), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(DOCS)}")


SETUP = """import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import neurosetta as nr

%matplotlib inline

EXAMPLE_ID = 720575940596125868
tree = nr.load_example_data(EXAMPLE_ID)
forest = nr.load_example_data()
print(tree)
print(f"Forest ids: {forest.ids()}")"""


def build_example_data() -> None:
    write(
        DOCS / "getting_started" / "example_data.ipynb",
        [
            md(
                """# Example data walkthrough

Load bundled FlyWire morphologies with {func}`~neurosetta.load_example_data`,
inspect a {class}`~neurosetta.api.Tree`, and plot a 2D projection.

See also {doc}`io` and the markdown tutorials {doc}`../tutorials/tree_basics`.

**Run in the browser:** [Launch on Binder](https://mybinder.org/v2/gh/NikDrummond/NeuRosetta/main?filepath=docs%2Fgetting_started%2Fexample_data.ipynb)

Re-execute before committing code changes: `python docs/_execute_notebooks.py`"""
            ),
            code(
                """import matplotlib.pyplot as plt
import neurosetta as nr

%matplotlib inline

print(f"neurosetta {nr.__version__}")
print(f"NR data dir: {nr.example_data_dir()}")
print(f"SWC data dir: {nr.example_data_dir(format='swc')}")"""
            ),
            code(
                """EXAMPLE_ID = 720575940596125868

forest = nr.load_example_data()
tree = nr.load_example_data(EXAMPLE_ID)

print(forest)
print(tree)"""
            ),
            code(
                """print("ID:", tree.ID)
print("metadata:", tree.metadata)
print("vertices:", tree.graph.num_vertices(), "edges:", tree.graph.num_edges())"""
            ),
            code(
                """print(f"nodes:      {tree.count_nodes()}")
print(f"branches:   {tree.count_branches()}")
print(f"leaves:     {tree.count_leaves()}")
print(f"cable len:  {tree.get_total_cable_length():.1f}")"""
            ),
            code(
                """ax = tree.show_2d()
ax.set_title(f"Tree {tree.ID}")
plt.show()"""
            ),
            code(
                """import tempfile
from pathlib import Path

out = Path(tempfile.mkdtemp())
tree.set_units("nm")
tree.save_tree(out / f"{tree.ID}.nr")
reloaded = nr.load(out / f"{tree.ID}.nr")
print(reloaded.metadata.get("units"), reloaded.count_nodes())"""
            ),
        ],
    )


def build_io() -> None:
    write(
        DOCS / "getting_started" / "io.ipynb",
        [
            md(
                """# Reading and writing neuron files

NeuRosetta reads and writes three main formats:

| Format | Functions | Typical use |
|--------|-----------|-------------|
| **SWC** | `import_swc`, `export_swc` | Standard morphology exchange |
| **NR** | `save`, `load` | Native graph-tool based `.nr` files |
| **Mesh** | `import_mesh`, `export_mesh` | Neuron / neuropil surfaces (developmental) |

Bundled FlyWire examples live under `docs/data/`. See {doc}`../reference/example_data`."""
            ),
            code(SETUP),
            md("## Import `.swc` files"),
            code(
                """swc_dir = nr.example_data_dir(format="swc")
tree = nr.import_swc(swc_dir / f"{EXAMPLE_ID}.swc")
print(tree)

forest = nr.import_swc(swc_dir)
print(forest.ids())"""
            ),
            md(
                """SWC stems must be **integer IDs** (FlyWire segment IDs). A directory import
returns a {class}`~neurosetta.api.Forest`.

See {doc}`../tutorials/tree_basics` for the `Tree` class and
{doc}`../tutorials/forests` for forests."""
            ),
            md("## Export `.swc` files"),
            code(
                """out = Path(tempfile.mkdtemp())
nr.export_swc(tree, out)
print(list(out.glob("*.swc")))"""
            ),
            md("## Native `.nr` files"),
            code(
                """out = Path(tempfile.mkdtemp())
tree.save_tree(out / f"{tree.ID}.nr")
reloaded = nr.load(out / f"{tree.ID}.nr")
print(reloaded.count_nodes(), reloaded.metadata.get("units"))"""
            ),
            md(
                """`.nr` files preserve bound graph properties and metadata — prefer them for
NeuRosetta-native workflows. Use SWC when sharing with other tools.

Next: {doc}`../tutorials/tree_basics`."""
            ),
        ],
    )


def build_tree_basics() -> None:
    write(
        DOCS / "tutorials" / "tree_basics.ipynb",
        [
            md(
                """# Tree basics

The {class}`~neurosetta.api.Tree` class is the core handle for one neuron morphology.

See {doc}`../getting_started/io` for loading files."""
            ),
            code(SETUP),
            md("## ID, metadata, and graph"),
            code(
                """print(tree.ID)
print(tree.metadata)
print(type(tree.graph))"""
            ),
            md("## Graph properties"),
            code(
                """print(tree.list_properties())
print(tree.list_properties("g"))
print(tree.has_property("coordinates", level="all"))"""
            ),
            code(
                """import numpy as np

data = np.zeros(tree.count_nodes())
tree.set_property(name="zeros", data=data, level="v", create=True, dtype="int")
print(tree.get_property("zeros")[:5])"""
            ),
            md("## Counts and indices"),
            code(
                """print(tree.count_nodes(), tree.count_branches(), tree.count_leaves())
print("root:", tree.get_root_index())
print("branches:", tree.get_branch_indices()[:5])"""
            ),
            md("## Coordinates"),
            code(
                """coords = tree.get_node_coordinates()
print(coords.shape)
branch_coords = tree.get_node_coordinates(subset=tree.get_branch_indices())
print(branch_coords.shape)"""
            ),
            md("## Spatial units"),
            code(
                """tree.set_units("nm")
print(tree.metadata["units"])
tree.convert_units("um")
print(tree.metadata["units"])"""
            ),
            md(
                """Every tree method mirrors a function in {doc}`../api/tree_ops/index`.

Next: {doc}`forests`."""
            ),
        ],
    )


def build_tree_surgery() -> None:
    write(
        DOCS / "tutorials" / "tree_surgery.ipynb",
        [
            md(
                """# Tree surgery and subtrees

Reduce morphologies, reroot, and extract subtrees.

```{important}
`get_reduced_tree(inplace=False)` and `get_rerooted_tree(..., inplace=False)` return a
**graph-tool Graph**, not a `Tree`. `get_subtree()` edits the tree **in place**.
```"""
            ),
            code(SETUP),
            md("## Reduce"),
            code(
                """print("before:", tree.count_nodes(), "reduced?", tree.is_reduced())
tree.get_reduced_tree(inplace=True)
print("after:", tree.count_nodes(), "reduced?", tree.is_reduced())"""
            ),
            md("## Reroot"),
            code(
                """tree = nr.load_example_data(EXAMPLE_ID)
leaves = tree.get_leaf_indices()
tree.get_rerooted_tree(int(leaves[0]), inplace=True)
print("new root index:", tree.get_root_index())"""
            ),
            md("## Subtree scores and extraction"),
            code(
                """tree = nr.load_example_data(EXAMPLE_ID)
scores = tree.get_subtree_scores(bind=False)
best = tree.get_max_subtree_node()
print("best subtree root:", best, "score:", scores[best])

branches = tree.get_branch_indices()
if len(branches):
    v = int(branches[0])
    tree.subtree_mask_from_root(v)
    tree.get_subtree()
    print("subtree nodes:", tree.count_nodes())"""
            ),
            md("Next: {doc}`gui`."),
        ],
    )


def build_plotting() -> None:
    write(
        DOCS / "tutorials" / "plotting.ipynb",
        [
            md(
                """# Plotting

Matplotlib for 2D / dendrogram; vedo for 3D and {class}`~neurosetta.ops.plotting.Viewer`."""
            ),
            code(SETUP),
            md("## 2D projection"),
            code(
                """ax = tree.show_2d()
ax.set_title(f"Tree {tree.ID}")
plt.show()"""
            ),
            md("## Dendrogram"),
            code("ax = tree.show_dendrogram()\nplt.show()"),
            md("## Custom Viewer (headless screenshot)"),
            code(
                """from neurosetta import Viewer

tree_a = nr.load_example_data(forest.ids()[0])
tree_b = nr.load_example_data(forest.ids()[1])

with Viewer(interactive=False) as viewer:
    viewer.add_neuron(tree_a)
    viewer.add_neuron(tree_b)
    out = Path(tempfile.mkdtemp()) / "two_neurons.png"
    viewer.screenshot(str(out))
    print("wrote", out)"""
            ),
            md("Next: {doc}`tree_surgery`."),
        ],
    )


def build_forests() -> None:
    write(
        DOCS / "tutorials" / "forests.ipynb",
        [
            md("# Forests"),
            code(SETUP),
            md("## Metadata filter"),
            code(
                """ids = forest.ids()
forest.by_id(ids[0]).metadata["Neuron_type"] = "A"
forest.by_id(ids[1]).metadata["Neuron_type"] = "B"

subset = forest.filter(Neuron_type="A")
print(subset.ids())"""
            ),
            md("## Batch descriptives"),
            code(
                """print(forest.count_nodes())
print(forest.get_total_cable_length())"""
            ),
            md("## Custom `apply`"),
            code(
                """counts = forest.apply(lambda t: t.count_nodes(), parallel=False)
print(counts)"""
            ),
            md("Next: {doc}`plotting`."),
        ],
    )


def build_extending() -> None:
    write(
        DOCS / "development" / "extending_nr.ipynb",
        [
            md(
                """# Extending NeuRosetta

Patterns for custom analysis without forking the library. Read
{doc}`architecture` first for the layer model (`utils` → `ops` → `api`).

This notebook shows **user-level extension**: attach analysis results as graph
properties and reuse the functional API."""
            ),
            code(SETUP),
            md("## Attach a custom vertex property"),
            code(
                """import numpy as np

# Example: distance of each node from the root in microns
tree.set_units("nm")
root = tree.get_root_coordinate()
coords = tree.get_node_coordinates()
dist_nm = np.linalg.norm(coords - root, axis=1)
tree.set_property("dist_from_root", dist_nm, level="v", create=True, dtype="double")
print(tree.get_property("dist_from_root")[:5])"""
            ),
            md("## Persist custom properties in `.nr`"),
            code(
                """out = Path(tempfile.mkdtemp())
tree.save_tree(out / f"{tree.ID}.nr")
reloaded = nr.load(out / f"{tree.ID}.nr")
print(reloaded.has_property("dist_from_root", level="v"))
print(reloaded.get_property("dist_from_root")[:5])"""
            ),
            md("## Functional API + `Forest.apply`"),
            code(
                """def mean_radius(t):
    return float(t.get_property("radius").mean())

radii = forest.apply(mean_radius, parallel=False)
print(dict(zip(forest.ids(), radii)))"""
            ),
            md(
                """To add a **library** operation, implement graph logic in `utils/`, wrap in
`ops/tree_graphs/`, bind on {class}`~neurosetta.api.Tree`, and optionally re-export
from `neurosetta.__init__`. See {doc}`architecture`.

Next: {doc}`../api/index`."""
            ),
        ],
    )


def main() -> None:
    build_example_data()
    build_io()
    build_tree_basics()
    build_tree_surgery()
    build_plotting()
    build_forests()
    build_extending()


if __name__ == "__main__":
    main()
