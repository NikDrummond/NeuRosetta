"""Base underlying _Tree class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from graph_tool.all import Graph

from .stone import _Stone

if TYPE_CHECKING:
    from ..ops.plotting.utils import TreePlot3D


class _Tree(_Stone):
    """Underlying tree graph class.

    ``ID`` and ``metadata`` are properties over ``graph.gp`` — the graph is
    the source of truth. Parent ``_Stone`` slots for those names are unused.
    """

    __slots__ = ("graph", "_3d_plot")

    def __init__(self, ID: int, metadata: dict, graph: Graph) -> None:
        self.graph = graph
        self._3d_plot = None
        # Bind / overwrite core gps (graph is source of truth)
        self.ID = ID
        self.metadata = metadata

    # --- identity (graph gp) ---

    @property
    def ID(self) -> int:
        return int(self.graph.gp["ID"])

    @ID.setter
    def ID(self, value: int) -> None:
        value = int(value)
        g = self.graph
        if "ID" not in g.gp:
            g.gp["ID"] = g.new_gp("long", value)
        else:
            g.gp["ID"] = value

    @property
    def metadata(self) -> dict:
        return self.graph.gp["metadata"]

    @metadata.setter
    def metadata(self, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("metadata must be a dict")
        g = self.graph
        if "metadata" not in g.gp:
            g.gp["metadata"] = g.new_gp("object", value)
        else:
            g.gp["metadata"] = value

    # --- constructors / copy ---

    @classmethod
    def from_graph(cls, graph: Graph) -> Self:
        """Wrap a graph that already has ``gp['ID']`` and ``gp['metadata']``."""
        return cls(ID=int(graph.gp["ID"]), metadata=graph.gp["metadata"], graph=graph)

    def copy(self) -> Self:
        """Shallow graph copy with a shallow-copied metadata dict."""
        g = self.graph.copy()
        meta = dict(g.gp["metadata"])
        g.gp["metadata"] = meta
        return type(self)(ID=int(g.gp["ID"]), metadata=meta, graph=g)

    clone = copy

    # --- repr / props ---

    def __repr__(self) -> str:
        return f"Tree(ID={self.ID}) with {self.graph.num_vertices()} nodes"

    def list_properties(self):
        """List internal (bound) graph properties."""
        self.graph.list_properties()

    # --- 3d plot cache ---

    @property
    def plot3d(self) -> TreePlot3D | None:
        return self._3d_plot

    @plot3d.setter
    def plot3d(self, value: TreePlot3D | None) -> None:
        self._3d_plot = value

    def make_plot3d(
        self,
        show_root: bool = True,
        line_kwargs: dict | None = None,
        root_kwargs: dict | None = None,
        random_c: bool = False,
        cache: bool = True,
    ) -> TreePlot3D:
        """Build a TreePlot3D for this tree.

        Parameters
        ----------
        cache : bool, optional
            Store the result on ``self._plot3d``, by default True.
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
            self.plot3d = plot
        return plot
