### read and write utils for .nr files

### Imports
from graph_tool.all import load_graph
from typing import TYPE_CHECKING

from ..errors.errors import _check_internal_property
from ..core import _Tree
if TYPE_CHECKING:
    from ..classes import Tree

def _bind_core(tree: _Tree):
    """
    Make sure ID and metadata are bound to the graph
    """

    # check if we have bound ID
    try:
        _check_internal_property(tree.graph, 'ID')
    except:
        tree.graph.gp['ID'] = tree.graph.new_gp('string', str(tree.ID))

    # check if we have bound metadata
    try:
        _check_internal_property(tree.graph, 'metadata')
    except:
        tree.graph.gp['metadata'] = tree.graph.new_gp('object', tree.metadata)

def save(tree: _Tree, fpath: str | None):
    """Save tree as .nr file

    The current ID of the neuron is used as the file name for simplicity.

    WSaving this way ensures core properties (ID, metadata, and any bound graph properties) are also saved.

    Parameters
    ----------
    tree : Tree
        Neuron tree
    fpath : str | None
        save path. If None will save to current directory
    """

    # check we have the core properties bound to the graph and bind them
    _bind_core(tree)

    # sort file path
    if fpath is None:
        sp = str(tree.ID) + ".nr"
    else:
        sp = fpath + str(tree.ID) + ".nr"

    # save
    tree.graph.save(sp, fmt = "gt")

def load(fpath) -> "Tree":
    """_summary_

    Parameters
    ----------
    fpath : _type_
        _description_

    Returns
    -------
    Tree
        _description_
    """
    from ..classes import Tree

    g = load_graph(fpath, fmt = 'gt')

    return Tree(ID = g.gp['ID'], metadata = g.gp['metadata'], graph = g)

