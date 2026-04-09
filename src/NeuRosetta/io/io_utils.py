from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar, List
from warnings import warn
from numpy import (
    int32,
    float64,
    unique,
    where,
    ndarray,
    zeros_like,
    vstack,
    array,
)
from pandas import DataFrame, read_csv
from graph_tool.all import Graph
from tqdm import tqdm

from ..core import _Tree

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
        warn("Duplicate node_id values found in DataFrame.")

    return df


def _node_inds(g: Graph, df: DataFrame) -> List[int]:
    """
    Given a graph, with ids as a vp, and a df with the same set of ids, find the order of indicies to order things going from the table to the graph

    This is important whenever adding an attribute to nodes in a graph
    """
    # node id orders - this is the order of nodes in the graph, which match the node ids in the table
    ids = g.vp["ids"].a
    # nodes in the table
    nodes = df["node_id"].values
    inds = [where(nodes == i)[0][0] for i in ids]

    return inds


def _infer_node_types(g: Graph) -> ndarray:
    """
    Infer node types as root(-1), branch (5), terminal (6), and transitory (0) based on in/out degree

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

    # we want to know the indicies of the nodes in the table that match how the nodes were added to the graph

    # indicies of graph nodes in table
    inds = _node_inds(g, df)

    # add some attributed from node table
    # initilise vertex properties - radius, coordinates
    vprop_rad = g.new_vp("double")
    vprop_coords = g.new_vp("vector<double>")

    # populate them
    vprop_rad.a = df.radius.values[inds]
    vprop_coords.set_2d_array(df[["x", "y", "z"]].values[inds].T)

    # add them
    g.vp["radius"] = vprop_rad
    g.vp["coordinates"] = vprop_coords
    g.vp["node_type"] = g.new_vp("int", _infer_node_types(g))

    return g


def _swc_table(tree: _Tree) -> DataFrame:

    # get node ids, radius and cooridnates
    # ids = tree.graph.vp["ids"].a
    radius = tree.graph.vp["radius"].a
    coords = tree.graph.vp["coordinates"].get_2d_array().T
    # get edges - reorder columns so node to parent
    edges = tree.graph.get_edges()[:, [1, 0]]

    # get node types
    node_types = _infer_node_types(tree.graph)

    # append root,-1 to the top of the edge list
    edges = vstack((array([0, -1]), edges))

    df = (
        DataFrame(
            {
                "node_id": edges[:, 0],
                "type": node_types,
                "x": coords[:, 0],
                "y": coords[:, 1],
                "z": coords[:, 2],
                "radius": radius,
                "parent_id": edges[:, 1],
            }
        )
        .sort_values("node_id")
        .reset_index(drop=True)
    )

    return df


def _base_meta():
    """Basic metadata info"""
    meta = {"ID": 1, "units": "Undefined", "file_path": "", "isReduced": False}

    return meta


### nr utils

def _bind_core(tree: _Tree):
    """
    Make sure ID and metadata are bound to the graph
    """

    # check if we have bound ID
    if not tree.check_property("ID", "g"):
        tree.graph.gp["ID"] = tree.graph.new_gp("long", tree.ID)

    # check if we have bound metadata
    if not tree.check_property("metadata","g"):
        tree.graph.gp["metadata"] = tree.graph.new_gp("object", tree.metadata)


### parallel utils

def _map_with_progress(
    fn: Callable[[T], T],
    items: list[T],
    *,
    parallel: bool,
    max_workers: int | None,
    progress: bool,
    desc: str,
):
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            it = ex.map(fn, items)
            if progress:
                it = tqdm(it, total=len(items), desc=desc)
            return list(it)
    else:
        it = items
        if progress:
            it = tqdm(it, total=len(items), desc=desc)
        return [fn(x) for x in it]

def _foreach_with_progress(
    fn: Callable[[T], None],
    items: list[T],
    *,
    parallel: bool,
    max_workers: int | None,
    progress: bool,
    desc: str,
) -> None:
    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            it = ex.map(fn, items)
            if progress:
                it = tqdm(it, total=len(items), desc=desc)
            list(it)
    else:
        it = items
        if progress:
            it = tqdm(it, total=len(items), desc=desc)
        for x in it:
            fn(x)