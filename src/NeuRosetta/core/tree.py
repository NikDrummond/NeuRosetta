""" base underlying _Tree class """
from __future__ import annotations
from typing import TYPE_CHECKING

from graph_tool.all import Graph

from .stone import _Stone

if TYPE_CHECKING:
    from ..ops.plotting.utils import TreePlot3D

class _Tree(_Stone):
    """Underlying tree graph class"""

    # constructor
    def __init__(self, ID: int, metadata: dict, graph: Graph) -> None:
        super().__init__(ID, metadata)
        self.graph = graph

    # list properties (temp implementation)
    def list_properties(self):
        """List internal (bound) graph properties"""
        self.graph.list_properties()

    @property
    def _plot3d(self) -> TreePlot3D | None:
        return getattr(self, "_3d_plot", None)

    @_plot3d.setter
    def _plot3d(self, value: TreePlot3D) -> None:
        self._3d_plot = value

    def make_plot3d(
        self,
        show_root: bool = True,
        line_kwargs: dict | None = None,
        root_kwargs: dict | None = None,
        random_c: bool = False,
        cache: bool = False,
    ) -> TreePlot3D:
        """Build a TreePlot3D for this tree.

        Parameters
        ----------
        cache : bool, optional
            Store the result on ``self._plot3d``, by default False.
        """
        from ..ops.plotting.utils import TreePlot3D  # runtime import

        plot = TreePlot3D(
            tree=self,
            show_root=show_root,
            line_kwargs=line_kwargs,
            root_kwargs=root_kwargs,
            random_c=random_c,
        )
        if cache:
            self._plot3d = plot
        return plot
