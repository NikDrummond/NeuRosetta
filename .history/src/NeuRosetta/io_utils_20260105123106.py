from pandas import DataFrame, read_csv
from numpy import int32, float64, unique, where, asarray, zeros_like, savetxt
from graph_tool.all import Graph
from warnings import warn
import os

from .tree_class import Tree_graph
from .errors import _check_swc_columns
from typing import List

### swc utils


### read swc




def read_swc(fpath: str, units=None, meta=None) -> Tree_graph:

    ID = os.path.splitext(os.path.basename(fpath))[0]
    df = _table_from_swc(fpath)
    graph = _graph_from_table(df)
    return Tree_graph(ID=ID, units=units, meta=meta, graph=graph)





def write_swc(tree: Tree_graph, fpath: str) -> None:

    df = tree_to_SWCtable(tree)
    savetxt(
        fpath,
        df,
        header="SWC Generated using Neurosetta \n Columns \n" + str(df.columns),
    )
