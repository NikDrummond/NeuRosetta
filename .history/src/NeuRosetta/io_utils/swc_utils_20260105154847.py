""" Functions for reading and writing .swc files"""

### Imports
from numpy import int32, float64, unique, where, ndarray, zeros_like, vstack, array, savetxt
from pandas import DataFrame, read_csv
from warnings import warn
from typing import List
from graph_tool.all import Graph
import os



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
    required_columns = ['node_id','type','x','y','z','radius','parent_id']
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

    return g

    