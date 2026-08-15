from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar
from warnings import warn

from graph_tool.all import Graph
from numpy import (
    float64,
    int32,
    ndarray,
    unique,
    where,
    zeros_like,
)
from pandas import DataFrame, read_csv
from tqdm import tqdm

from ..core import _Tree
from ..ops.tree_graphs import has_property
from ..utils.graph_utils.vertex_inds import root_index
from ..utils.units import DEFAULT_UNITS

T = TypeVar("T")

### swc utils


def _check_swc_columns(df, error_type=ValueError):
    """
    Checks if required columns exist in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to check.
        required_columns (list): List of required column names.
        error_type (Exception): Type of exception to raise (default: ValueError).

    Raises:
        error_type: If any required column is missing.
    """
    required_columns = ["node_id", "type", "x", "y", "z", "radius", "parent_id"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise error_type(f"Missing required columns: {missing}")


def _table_from_swc(file_path: str) -> DataFrame:
    """
    Node table from swc file, using pandas
    """
    df = read_csv(
        file_path,
        names=["node_id", "type", "x", "y", "z", "radius", "parent_id"],
        comment="#",
        engine="c",
        sep="\s+",
        dtype={
            "node_id": int32,
            "type": int32,
            "x": float64,
            "y": float64,
            "z": float64,
            "radius": float64,
            "parent_id": int32,
        },
        index_col=False,
    )

    # check for duplicate node ids
    if unique(df.node_id.values).size != len(df.node_id.values):
        warn("Duplicate node_id values found in DataFrame.", stacklevel=2)

    return df


def _node_inds(g: Graph, df: DataFrame) -> list[int]:
    """
    Map table node ids to graph vertex indices for attribute assignment.
    """
    # node id orders - graph node order matches node ids in the table
    ids = g.vp["ids"].a
    # nodes in the table
    nodes = df["node_id"].values
    inds = [where(nodes == i)[0][0] for i in ids]

    return inds


def _infer_node_types(g: Graph) -> ndarray:
    """
    Infer node types: root (-1), branch (5), terminal (6), transitory (0).

    Returns a np.array

    """
    # indicies
    out_deg = g.get_out_degrees(g.get_vertices())
    in_deg = g.get_in_degrees(g.get_vertices())
    ends = where(out_deg == 0)
    branches = where(out_deg > 1)
    root = where(in_deg == 0)
    node_types = zeros_like(g.get_vertices(), dtype=int32)
    node_types[ends] = 6
    node_types[branches] = 5
    node_types[root] = -1

    return node_types


def _graph_from_table(df: DataFrame) -> Graph:
    """
    From a node table, generate a graph-tool graph
    """

    # check we have recognisable columns
    _check_swc_columns(df)

    # get edges
    edges = df.loc[df.parent_id != -1, ["parent_id", "node_id"]].values

    # create new (hashed) graph with edges
    g = Graph(edges, hashed=True, hash_type="int")

    # table indices matching the order nodes were added to the graph

    # indicies of graph nodes in table
    inds = _node_inds(g, df)

    # add some attributed from node table
    # initilise vertex properties - radius, coordinates
    vprop_rad = g.new_vp("double")
    vprop_x = g.new_vp("double")
    vprop_y = g.new_vp("double")
    vprop_z = g.new_vp("double")

    # populate them
    vprop_rad.a = df.radius.values[inds]
    vprop_x.a = df["x"].values[inds]
    vprop_y.a = df["y"].values[inds]
    vprop_z.a = df["z"].values[inds]

    # add them
    g.vp["radius"] = vprop_rad
    g.vp["x"] = vprop_x
    g.vp["y"] = vprop_y
    g.vp["z"] = vprop_z
    g.vp["node_type"] = g.new_vp("int", _infer_node_types(g))

    return g


def _swc_table(tree: _Tree) -> DataFrame:
    """Build an SWC node table from a tree graph.

    Rows use SWC ``node_id`` / ``parent_id`` values from ``g.vp['ids']``, matching
    :func:`_graph_from_table` import semantics.
    """
    g = tree.graph
    ids = g.vp["ids"].a if "ids" in g.vp else g.get_vertices()
    root = root_index(g)
    node_types = _infer_node_types(g)

    x = g.vp["x"].a
    y = g.vp["y"].a
    z = g.vp["z"].a
    radius = g.vp["radius"].a

    node_id = []
    parent_id = []
    types = []
    xs = []
    ys = []
    zs = []
    radii = []

    for v in g.get_vertices():
        node_id.append(int(ids[v]))
        if v == root:
            parent_id.append(-1)
        else:
            parent_id.append(int(ids[int(g.get_in_neighbors(v)[0])]))
        types.append(int(node_types[v]))
        xs.append(float(x[v]))
        ys.append(float(y[v]))
        zs.append(float(z[v]))
        radii.append(float(radius[v]))

    return (
        DataFrame(
            {
                "node_id": node_id,
                "type": types,
                "x": xs,
                "y": ys,
                "z": zs,
                "radius": radii,
                "parent_id": parent_id,
            }
        )
        .sort_values("node_id")
        .reset_index(drop=True)
    )


def _base_meta():
    """Basic metadata info (identity lives on ``tree.ID`` / ``gp['ID']``, not here)."""
    return {"units": DEFAULT_UNITS, "file_path": "", "isReduced": False, "Flag": False}


def _apply_import_units(
    tree: _Tree,
    set_units: str | None,
    *,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> None:
    """Assign spatial units to an imported tree without rescaling geometry."""
    if set_units is None:
        return
    from ..ops.units import set_units as assign_tree_units

    assign_tree_units(
        tree,
        set_units,
        voxel_size=voxel_size,
        voxel_unit=voxel_unit,
    )


### nr utils


def _bind_core(tree: _Tree):
    """Assert core identity gps are bound (Tree.__init__ always binds them)."""
    if not has_property(tree, "ID", "g") or not has_property(tree, "metadata", "g"):
        raise RuntimeError(
            "Tree graph is missing core gps ('ID', 'metadata'); "
            "construct via Tree(...) so they are bound."
        )


### parallel utils


def _map_with_progress(
    fn: Callable[[T], T],
    items: list[T],
    *,
    parallel: bool,
    max_workers: int | None,
    show_progress: bool,
    desc: str,
) -> list[T]:
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fn, item) for item in items]
            it = tqdm(futures, total=len(items), desc=desc) if show_progress else futures
            return [fut.result() for fut in it]
    else:
        it = tqdm(items, total=len(items), desc=desc) if show_progress else items
        return [fn(x) for x in it]


def _foreach_with_progress(
    fn: Callable[[T], None],
    items: list[T],
    *,
    parallel: bool,
    max_workers: int | None,
    show_progress: bool,
    desc: str,
) -> None:
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fn, item) for item in items]
            it = tqdm(futures, total=len(items), desc=desc) if show_progress else futures
            for fut in it:
                fut.result()
    else:
        it = tqdm(items, total=len(items), desc=desc) if show_progress else items
        for x in it:
            fn(x)
