"""Analysis functions that return new tree objects.

This module contains functions that operate on trees or forests and return
new tree/forest objects rather than modifying them in place.

Currently these functions are commented out and not actively used.
"""

# from ..api import Tree, Forest
# from ..ops.tree_graphs.traversals import reduce_graph


# def reduce_tree(
#     tree: Tree | Forest,
#     parallel: bool = True,
#     max_workers: int = 4,
#     progress: bool = True,
# ) -> Tree | Forest:
#     """Reduce a tree graph to only have root, branch, and leaf nodes.
#
#     Parameters
#     ----------
#     tree : Tree | Forest
#         Single Tree or Forest to reduce.
#     parallel : bool, optional
#         If True, apply reduction in parallel across forest. By default True.
#     max_workers : int, optional
#         Maximum number of worker processes. By default 4.
#     progress : bool, optional
#         If True, show progress bar. By default True.
#
#     Returns
#     -------
#     Tree | Forest
#         Reduced tree or forest with only root, branch, and leaf nodes.
#     """
#
#     # if we are handling a single tree:
#     if isinstance(tree, Tree):
#         g = tree.reduce_tree()
#         return Tree(ID=g.gp["ID"], metadata=g.gp["metadata"], graph=g)
#
#     if isinstance(tree, Forest):
#         graphs = tree.apply(
#             reduce_graph,
#             parallel=parallel,
#             max_workers=max_workers,
#             show_progress=progress,
#         )
#         red_forest = Forest(
#             [Tree(ID=g.gp["ID"], metadata=g.gp["metadata"], graph=g) for g in graphs]
#         )
#         return red_forest
