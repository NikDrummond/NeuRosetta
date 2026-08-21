"""Base underlying _Tree class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from graph_tool.all import Graph

from .metadata import (
    PROTECTED_META_KEYS,
    _MetadataDict,
    check_protected_meta_key,
    set_core_meta,
)
from .stone import _Stone
from .tree_helpers import (
    bind_tree_id,
    bind_tree_metadata,
    copy_tree_graph,
    ensure_edge_lengths,
    metadata_from_graph,
)

if TYPE_CHECKING:
    from ..ops.plotting.utils import TreePlot3D


class _Tree(_Stone):
    """Underlying tree graph class.

    ``ID`` and ``metadata`` are properties over ``graph.gp`` — the graph is
    the source of truth. Parent ``_Stone`` slots for those names are unused.

    Attributes
    ----------
    plot3d : TreePlot3D
        3D plot bound to this tree, styleable straight away but holding no vedo
        actors until :meth:`make_plot3d` (or ``plot3d.build()``) generates them.
    """

    __slots__ = ("graph", "plot3d")

    def __init__(self, ID: int, metadata: dict, graph: Graph) -> None:
        from ..ops.plotting.utils import TreePlot3D  # runtime import

        self.graph = graph
        self.ID = ID
        self.metadata = metadata
        # Bound but unbuilt: styling and graph lookups work before any geometry.
        self.plot3d = TreePlot3D(tree=self, build=False)
        ensure_edge_lengths(self)

    # --- identity (graph gp) ---

    @property
    def ID(self) -> int:
        return int(self.graph.gp["ID"])

    @ID.setter
    def ID(self, value: int) -> None:
        bind_tree_id(self.graph, value)

    @property
    def metadata(self) -> _MetadataDict:
        return self.graph.gp["metadata"]

    @metadata.setter
    def metadata(self, value: dict | _MetadataDict) -> None:
        bind_tree_metadata(self.graph, value)

    # --- user metadata (protected core keys) ---

    def set_meta(self, key: str, value) -> None:
        """Set a user metadata entry.

        Core keys (``units``, ``file_path``, ``isReduced``, ``Flag``, and
        voxel fields) are protected — use the dedicated tree APIs instead.
        """
        check_protected_meta_key(key)
        self.metadata[key] = value

    def del_meta(self, key: str) -> None:
        """Delete a user metadata entry.

        Core keys are protected — use the dedicated tree APIs instead.
        """
        check_protected_meta_key(key)
        del self.metadata[key]

    def list_meta(self, *, include_protected: bool = False) -> list[str]:
        """List metadata keys present on this tree.

        By default only user-editable keys are returned. Pass
        ``include_protected=True`` to include core metadata keys as well.
        """
        keys = list(self.metadata)
        if include_protected:
            return sorted(keys)
        return sorted(k for k in keys if k not in PROTECTED_META_KEYS)

    def meta_summary(self, *, include_protected: bool = False) -> dict[str, int]:
        """Return ``{key: 1}`` for each metadata key defined on this tree."""
        return {key: 1 for key in self.list_meta(include_protected=include_protected)}

    def set_flag(self, value: bool) -> None:
        """Set the ``Flag`` metadata entry."""
        set_core_meta(self.metadata, "Flag", bool(value))

    # --- constructors / copy ---

    @classmethod
    def from_graph(cls, graph: Graph) -> Self:
        """Wrap a graph that already has ``gp['ID']`` and ``gp['metadata']``."""
        meta = metadata_from_graph(graph)
        return cls(ID=int(graph.gp["ID"]), metadata=meta, graph=graph)

    def copy(self) -> Self:
        """Shallow graph copy with a shallow-copied metadata dict."""
        tree_id, meta, graph = copy_tree_graph(self.graph)
        return type(self)(ID=tree_id, metadata=meta, graph=graph)

    clone = copy

    def __eq__(self, other: object) -> bool:
        """Return True when *other* is a tree wrapping the same graph object."""
        if not isinstance(other, _Tree):
            return NotImplemented
        return self.graph is other.graph

    def __repr__(self) -> str:
        """Return a short summary of tree ID and node count."""
        return f"Tree(ID={self.ID}) with {self.graph.num_vertices()} nodes"

    # --- graph properties ---

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
        level: str | None = None,
        *,
        as_array: bool = True,
        SoA: bool = False,
    ):
        """Get a property map or its array values from the underlying graph.

        If ``level`` is omitted, all of ``g``/``v``/``e`` are searched. A single
        match is returned directly; if the name exists at multiple levels, a
        warning is issued and ``{level: value, ...}`` is returned.

        Vector properties (e.g. ``coordinates`` / ``vector<double>``) use
        ``get_2d_array()``, not ``.a``. With ``as_array=True`` and ``SoA=False``
        (default) returns shape ``(N, d)``; ``SoA=True`` returns ``(d, N)``.
        """
        from ..utils.graph_utils import get_property as _get_property

        return _get_property(self.graph, name, level, as_array=as_array, SoA=SoA)

    def set_property(
        self,
        name: str,
        data,
        level: str,
        *,
        dtype: str | None = None,
        create: bool = False,
    ) -> None:
        """Set an existing graph property, or create one with ``create=True``.

        Vector properties are written with ``set_2d_array`` as ``(d, n)``.
        Pass ``(d, n)`` or ``(n, d)`` — layout is inferred from the level size
        ``n`` (vertices or edges). Square ``(n, n)`` warns and assumes SoA.
        """
        from ..utils.graph_utils import set_property as _set_property

        _set_property(
            self.graph,
            name,
            data,
            level,
            dtype=dtype,
            create=create,
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

    def make_plot3d(
        self,
        show_root: bool | None = None,
        line_kwargs: dict | None = None,
        root_kwargs: dict | None = None,
        random_c: bool = False,
        cache: bool = True,
        force_refresh: bool = False,
        **style_kwargs,
    ) -> TreePlot3D:
        """Build or return the cached :class:`~neurosetta.ops.plotting.utils.TreePlot3D`.

        Style arguments are applied whether or not geometry needs building, so this
        is safe to call repeatedly.

        Parameters
        ----------
        show_root : bool | None, optional
            Whether the root marker is included in the plot's actors.
            By default None (leave the plot's current setting).
        line_kwargs, root_kwargs : dict | None, optional
            Raw vedo constructor kwargs for the lines and root marker.
        random_c : bool, optional
            Pick a random colour, honoured only when geometry is actually built.
            By default False.
        cache : bool, optional
            Populate ``self.plot3d`` when True. When False, return a standalone
            copy that inherits the stored style without modifying the tree's
            plot. By default True.
        force_refresh : bool, optional
            Rebuild vedo actors even when a cached plot exists. By default False.
        **style_kwargs
            Further style overrides accepted by ``TreePlot3D.set_style``.
        """
        if not cache:
            plot = self.plot3d.copy()
            return plot.build(
                self,
                show_root=show_root,
                line_kwargs=line_kwargs,
                root_kwargs=root_kwargs,
                random_c=random_c,
                force=True,
                **style_kwargs,
            )

        return self.plot3d.build(
            self,
            show_root=show_root,
            line_kwargs=line_kwargs,
            root_kwargs=root_kwargs,
            random_c=random_c,
            force=force_refresh,
            **style_kwargs,
        )
