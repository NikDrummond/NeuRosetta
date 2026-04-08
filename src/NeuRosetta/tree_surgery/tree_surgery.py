from ..classes import Tree, Forest
from ..tree_graphs.traversals import reduce_graph


def reduce_tree(
    tree: Tree | Forest,
    parallel: bool = True,
    max_workers: int = 4,
    progress: bool = True,
) -> Tree | Forest:
    """Reduce a tree graph to only have root, branch, and leaf ndoes"""

    ### if we are handling a single tree:
    if isinstance(tree, Tree):
        g = tree.reduce_tree()
        return Tree(ID=g.gp["ID"], metadata=g.gp["metadata"], graph=g)

    if isinstance(tree, Forest):
        graphs = tree.apply(
            reduce_graph,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )
        red_forest = Forest(
            [Tree(ID=g.gp["ID"], metadata=g.gp["metadata"], graph=g) for g in graphs]
        )
        return red_forest
