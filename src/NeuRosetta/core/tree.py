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

    # --- repr / graph properties ---

    def __repr__(self) -> str:
        return f"Tree(ID={self.ID}) with {self.graph.num_vertices()} nodes"

    def list_properties(self, level: str = "all") -> list | dict:
        """List bound property names.

        Parameters
        ----------
        level : str, optional
            ``"all"`` returns ``{"g": [...], "v": [...], "e": [...]}``.
            ``"g"`` / ``"v"`` / ``"e"`` returns a list of names at that level.
        """
        from ..utils.graph_utils import list_properties as _list_properties

        return _list_properties(self.graph, level)

    def has_property(self, prop: str, level: str = "all") -> bool:
        """Return True if ``prop`` exists at ``level`` (``g``/``v``/``e``/``all``)."""
        from ..utils.graph_utils import g_has_property

        return g_has_property(self.graph, prop, level)

    def get_property(
        self,
        name: str,
        level: str,
        *,
        as_array: bool = False,
        SoA: bool = False,
    ):
        """Get a property map or its array values from the underlying graph.

        Vector properties (e.g. ``coordinates`` / ``vector<double>``) use
        ``get_2d_array()``, not ``.a``. With ``as_array=True`` and ``SoA=False``
        (default) returns shape ``(N, d)``; ``SoA=True`` returns ``(d, N)``.
        """
        from ..utils.graph_utils import get_property as _get_property

        return _get_property(
            self.graph, name, level, as_array=as_array, SoA=SoA
        )

    def set_property(
        self,
        name: str,
        data,
        level: str,
        *,
        dtype: str | None = None,
        create: bool = False,
        SoA: bool = False,
    ) -> None:
        """Set an existing graph property, or create one with ``create=True``.

        Vector properties are written with ``set_2d_array``. Pass ``data`` as
        ``(N, d)`` (``SoA=False``, default) or ``(d, N)`` (``SoA=True``).
        """
        from ..utils.graph_utils import set_property as _set_property

        _set_property(
            self.graph,
            name,
            data,
            level,
            dtype=dtype,
            create=create,
            SoA=SoA,
        )

    def del_property(self, name: str, level: str) -> None:
        """Delete a property from the underlying graph."""
        from ..utils.graph_utils import del_property as _del_property

        _del_property(self.graph, name, level)

    def revert_core_properties(self) -> None:
        """Drop non-core vertex/edge/graph properties from the graph."""
        from ..utils.graph_utils import revert_core_properties as _revert

        _revert(self.graph)

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
