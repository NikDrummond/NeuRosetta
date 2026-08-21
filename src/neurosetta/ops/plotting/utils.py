"""Vedo-based 3D plot helpers for neuron trees."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from vedo import Assembly, Lines

from ...config.vedo_settings import is_notebook_vedo_backend
from ...core import _Tree
from ...utils.vedo_utils import (
    MarkerActor,
    combined_bounds,
    get_marker_size,
    make_assembly,
    make_lines,
    make_point_marker,
    random_colour,
    resolve_colour,
    set_actor_alpha,
    set_actor_colour,
    set_marker_size,
)
from .style import (
    LINE_KWARG_ALIASES,
    ROOT_KWARG_ALIASES,
    Plot3DStyle,
    split_style_kwargs,
)

RootActor = MarkerActor

EdgeReduce = Literal["mean", "min", "max", "start", "stop"]

_SENTINEL = object()


class TreePlot3D:
    """3D plot representation of a neuron tree using vedo primitives.

    A plot always carries a :class:`~neurosetta.ops.plotting.style.Plot3DStyle`,
    while its vedo actors are generated on demand. Style can therefore be set on
    the unbuilt plot every tree exposes as ``tree.plot3d``: it is applied when
    :meth:`build` generates the actors, and re-applied by :meth:`rebuild` after the
    tree geometry changes.

    Parameters
    ----------
    tree : _Tree | None
        Tree to plot. If None, creates an unbound plot shell with default style.
    build : bool, optional
        Generate vedo actors immediately when *tree* is given. Pass False to bind
        the tree but defer geometry until first use. By default True.
    show_root : bool, optional
        Include the root marker in :attr:`actors`. By default True.
    line_kwargs : dict | None, optional
        Keyword arguments for the vedo ``Lines`` constructor. Recognised style keys
        (``c``/``lw``/``alpha`` and aliases) update the stored style; anything else
        is stored and forwarded verbatim on every build. By default None.
    root_kwargs : dict | None, optional
        Keyword arguments for the root marker constructor, handled as
        *line_kwargs*. By default None.
    random_c : bool, optional
        Pick a random line colour. By default False.
    **style_kwargs
        Further style overrides accepted by :meth:`set_style`, for example
        ``root_colour`` or ``lw``.

    Examples
    --------
    Style before any geometry exists, then build lazily::

        tree.plot3d.colour = "red"
        tree.plot3d.lw = 2
        tree.make_plot3d()          # actors built with the stored style

    Colour edges by a graph property::

        tree.plot3d.colour_by("Path_length", cmap="viridis")
    """

    __slots__ = (
        "_lines",
        "_root",
        "_scalars",
        "_scalar_limits",
        "_style",
        "_tree",
        "_tree_id",
    )

    def __init__(
        self,
        tree: _Tree | None = None,
        show_root: bool = True,
        line_kwargs: dict | None = None,
        root_kwargs: dict | None = None,
        random_c: bool = False,
        build: bool = True,
        **style_kwargs,
    ) -> None:
        """Initialize the plot style, and vedo actors when *tree* is given.

        See the class docstring for parameter descriptions.
        """
        self._style = Plot3DStyle()
        self._lines: Lines | None = None
        self._root: RootActor | None = None
        self._tree: _Tree | None = None
        self._tree_id: Any = None
        self._scalars: np.ndarray | None = None
        self._scalar_limits: tuple[float | None, float | None] = (None, None)

        self.set_style(
            show_root=show_root,
            line_kwargs=line_kwargs,
            root_kwargs=root_kwargs,
            random_c=random_c,
            **style_kwargs,
        )
        if tree is not None:
            self._bind_tree(tree)
            if build:
                self.build(tree, force=True)

    # --- tree binding ---

    @property
    def tree(self) -> _Tree | None:
        """Tree this plot is bound to, or None for an unbound shell."""
        return self._tree

    @property
    def tree_id(self) -> Any:
        """ID of the tree this plot is bound to, or None."""
        return self._tree_id

    def _bind_tree(self, tree: _Tree) -> None:
        """Record the tree so rebuilds and property lookups need no arguments.

        A tree and its own plot reference each other; the resulting cycle is
        reclaimed by the cyclic garbage collector.
        """
        self._tree = tree
        self._tree_id = tree.ID

    def _resolve_tree(self, tree: _Tree | None) -> _Tree:
        if tree is not None:
            return tree
        bound = self.tree
        if bound is None:
            raise ValueError("This plot is not bound to a tree; pass one explicitly.")
        return bound

    # --- state ---

    @property
    def is_built(self) -> bool:
        """Whether vedo actors have been generated."""
        return self._lines is not None

    def clear(self) -> None:
        """Drop vedo actors, keeping the style and tree binding intact."""
        self._lines = None
        self._root = None

    def build(
        self,
        tree: _Tree | None = None,
        *,
        show_root: bool | None = None,
        line_kwargs: dict | None = None,
        root_kwargs: dict | None = None,
        random_c: bool = False,
        force: bool = False,
        **style_kwargs,
    ) -> TreePlot3D:
        """Build vedo actors for *tree* in place, or restyle an existing build.

        When actors already exist and *force* is False the geometry is left alone,
        but any style arguments are still applied — so this is safe to call
        repeatedly as a cheap "ensure built" step.

        Parameters
        ----------
        tree : _Tree | None, optional
            Tree to build from. Defaults to the previously bound tree.
        show_root : bool | None, optional
            Update :attr:`show_root`. By default None (unchanged).
        line_kwargs, root_kwargs : dict | None, optional
            Raw vedo constructor kwargs, handled as in :meth:`set_style`.
        random_c : bool, optional
            Pick a random colour. Only honoured when geometry is actually
            (re)built, so repeated cached builds keep a stable colour.
            By default False.
        force : bool, optional
            Rebuild geometry even when actors already exist. By default False.
        **style_kwargs
            Further style overrides accepted by :meth:`set_style`.

        Returns
        -------
        TreePlot3D
            Self for chaining.
        """
        will_build = force or not self.is_built
        self.set_style(
            show_root=show_root,
            line_kwargs=line_kwargs,
            root_kwargs=root_kwargs,
            random_c=random_c and will_build,
            **style_kwargs,
        )
        if not will_build:
            return self

        tree = self._resolve_tree(tree)
        self._bind_tree(tree)
        starts, stops = tree.get_edge_coordinates()
        self._lines = make_lines(starts, stops, **self._style.line_build_kwargs())
        self._root = self._make_root(tree) if self._style.show_root else None
        self._apply_scalars()
        return self

    def rebuild(
        self,
        tree: _Tree | None = None,
        line_kwargs: dict | None = None,
        root_kwargs: dict | None = None,
        **style_kwargs,
    ) -> TreePlot3D:
        """Regenerate vedo actors from updated tree geometry, in place.

        The stored style (colour, line width, alpha, root size, colour map) is
        re-applied, so appearance is preserved across tree edits unless overridden.

        Parameters
        ----------
        tree : _Tree | None, optional
            Updated tree. Defaults to the previously bound tree.
        line_kwargs, root_kwargs : dict | None, optional
            Style overrides for this and subsequent builds. By default None.
        **style_kwargs
            Further style overrides accepted by :meth:`set_style`.

        Returns
        -------
        TreePlot3D
            Self for chaining.
        """
        return self.build(
            tree,
            line_kwargs=line_kwargs,
            root_kwargs=root_kwargs,
            force=True,
            **style_kwargs,
        )

    refresh = rebuild

    def copy(self, *, build: bool = False) -> TreePlot3D:
        """Return a new plot with the same style and tree binding.

        Vedo actors are never shared: the copy starts unbuilt, or is built fresh
        when *build* is True.
        """
        other = TreePlot3D()
        other._style = self._style.copy()
        other._scalars = None if self._scalars is None else self._scalars.copy()
        other._scalar_limits = self._scalar_limits
        other._tree = self._tree
        other._tree_id = self._tree_id
        if build:
            other.build(force=True)
        return other

    def _make_root(self, tree: _Tree) -> RootActor:
        """Build the root marker.

        Notebook backends (k3d, panel, …) get a :class:`vedo.Sphere` because a
        single-point actor has zero average size and leaks into / breaks k3d export.
        """
        coords = tree.get_node_coordinates(subset=tree.get_root_index())
        return make_point_marker(
            coords,
            as_sphere=is_notebook_vedo_backend(),
            **self._style.root_build_kwargs(),
        )

    def _sync_root_actor(self) -> None:
        """Create or drop the root actor to match :attr:`show_root`.

        The marker is never instantiated when hidden. That matters for k3d: constructing
        a Sphere registers it with the notebook plot even if it is later omitted from
        :attr:`actors`.
        """
        if not self.is_built:
            return
        if self._style.show_root:
            if self._root is None:
                self._root = self._make_root(self._resolve_tree(None))
        else:
            self._root = None

    # --- actors ---

    @property
    def lines(self) -> Lines | None:
        """Vedo ``Lines`` actor for tree edges, or None when unbuilt."""
        return self._lines

    @property
    def root(self) -> RootActor | None:
        """Root marker actor, or None when unbuilt."""
        return self._root

    @property
    def actors(self) -> list[Lines | RootActor]:
        """Vedo actors ready to pass to a Plotter, excluding missing ones."""
        candidates = [self._lines]
        if self._style.show_root:
            candidates.append(self._root)
        return [actor for actor in candidates if actor is not None]

    @property
    def n_segments(self) -> int | None:
        """Number of line segments, or None when unbuilt."""
        return None if self._lines is None else int(self._lines.ncells)

    def assembly(self) -> Assembly:
        """Group the actors into a single vedo :class:`Assembly`."""
        return make_assembly(self.actors)

    def bounds(self) -> tuple[float, ...] | None:
        """Return ``(xmin, xmax, ymin, ymax, zmin, zmax)``, or None when unbuilt."""
        return combined_bounds(self.actors)

    def center(self) -> np.ndarray | None:
        """Return the centre of :meth:`bounds`, or None when unbuilt."""
        box = self.bounds()
        if box is None:
            return None
        arr = np.asarray(box, dtype=float).reshape(3, 2)
        return arr.mean(axis=1)

    # --- style ---

    def get_style(self) -> dict:
        """Return the current style as a plain dict."""
        return self._style.as_dict()

    def set_style(
        self,
        *,
        colour: Any = None,
        color: Any = None,
        c: Any = None,
        lw: float | None = None,
        Line_width: float | None = None,
        line_width: float | None = None,
        linewidth: float | None = None,
        alpha: float | None = None,
        a: float | None = None,
        opacity: float | None = None,
        root_colour: Any = _SENTINEL,
        root_color: Any = _SENTINEL,
        root_c: Any = _SENTINEL,
        rc: Any = _SENTINEL,
        root_alpha: Any = _SENTINEL,
        root_a: Any = _SENTINEL,
        ra: Any = _SENTINEL,
        root_size: float | None = None,
        root_s: float | None = None,
        rs: float | None = None,
        show_root: bool | None = None,
        cmap: str | None = None,
        random_c: bool = False,
        seed: int | None = None,
        line_kwargs: dict | None = None,
        root_kwargs: dict | None = None,
    ) -> TreePlot3D:
        """Update style in bulk, applying to actors when they exist.

        Every argument is optional and omitted arguments leave the current value
        unchanged. Works on an unbuilt plot, in which case values are stored and
        applied by the next :meth:`build`.

        Parameters
        ----------
        colour, color, c : Any, optional
            Line colour, as any vedo colour specifier. Aliases of one another;
            overridden by *random_c*. Setting a colour clears scalar colouring.
        lw, Line_width, line_width, linewidth : float, optional
            Line width.
        alpha, a, opacity : float, optional
            Line opacity in ``[0, 1]``.
        root_colour, root_color, root_c, rc : Any, optional
            Root marker colour. Explicitly passing None makes the root follow the
            line colour again.
        root_alpha, root_a, ra : float, optional
            Root marker opacity. Explicitly passing None makes the root follow the
            line opacity again.
        root_size, root_s, rs : float, optional
            Root marker point size (sphere radius on notebook backends).
        show_root : bool, optional
            Whether the root marker is included in :attr:`actors`.
        cmap : str, optional
            Colour map name used for scalar colouring, see :meth:`colour_by`.
        random_c : bool, optional
            Pick a random RGB colour, taking priority over *colour*.
            By default False.
        seed : int, optional
            Seed for *random_c*, for reproducible colours. By default None.
        line_kwargs, root_kwargs : dict | None, optional
            Raw vedo constructor kwargs. Recognised style keys update the fields
            above; the remainder are stored and forwarded on every build.

        Returns
        -------
        TreePlot3D
            Self for chaining.
        """
        line_style, line_extras = split_style_kwargs(line_kwargs, LINE_KWARG_ALIASES)
        root_style, root_extras = split_style_kwargs(root_kwargs, ROOT_KWARG_ALIASES)

        colour = _coalesce(colour, color, c, line_style.get("colour"))
        if random_c:
            colour = random_colour(seed)
        root_colour = _first_given(
            root_colour,
            root_color,
            root_c,
            rc,
            root_style.get("root_colour", _SENTINEL),
        )
        root_alpha = _first_given(root_alpha, root_a, ra, root_style.get("root_alpha", _SENTINEL))

        lw = _coalesce(lw, Line_width, line_width, linewidth, line_style.get("lw"))
        alpha = _coalesce(alpha, a, opacity, line_style.get("alpha"))
        root_size = _coalesce(root_size, root_s, rs, root_style.get("root_size"))

        if colour is not None:
            self._style.colour = colour
            self._scalars = None
            self._style.cmap = None
        if root_colour is not _SENTINEL:
            self._style.root_colour = root_colour
        if lw is not None:
            self._style.lw = float(lw)
        if alpha is not None:
            self._style.alpha = float(alpha)
        if root_alpha is not _SENTINEL:
            self._style.root_alpha = None if root_alpha is None else float(root_alpha)
        if root_size is not None:
            self._style.root_size = float(root_size)
        if show_root is not None:
            self._style.show_root = bool(show_root)
        if cmap is not None:
            self._style.cmap = cmap
        if line_extras:
            self._style.line_kwargs.update(line_extras)
        if root_extras:
            self._style.root_kwargs.update(root_extras)

        self._sync_root_actor()
        self._apply_style()
        return self

    style = set_style

    def _apply_style(self) -> None:
        """Push the stored style onto existing actors, if any."""
        if self._lines is not None:
            # A scalar colour map owns the line colours; don't overwrite it.
            if self._scalars is None:
                set_actor_colour(self._lines, self._style.colour)
            self._lines.lw(self._style.lw)
            set_actor_alpha(self._lines, self._style.alpha)
        if self._root is not None:
            set_actor_colour(self._root, self._style.effective_root_colour)
            set_actor_alpha(self._root, self._style.effective_root_alpha)
            set_marker_size(self._root, self._style.root_size)

    # --- individual style properties ---

    @property
    def colour(self) -> Any:
        """Line colour. Setting it clears any scalar colouring."""
        return self._style.colour

    @colour.setter
    def colour(self, value: Any) -> None:
        self.set_style(colour=value)

    color = colour
    c = colour

    @property
    def colour_rgb(self) -> tuple[float, float, float]:
        """Line colour normalised to an RGB triple in ``[0, 1]``."""
        return resolve_colour(self._style.colour)

    @property
    def lw(self) -> float:
        """Line width."""
        return self._style.lw

    @lw.setter
    def lw(self, value: float) -> None:
        self.set_style(lw=value)

    linewidth = lw
    Line_width = lw

    @property
    def alpha(self) -> float:
        """Line opacity in ``[0, 1]``."""
        return self._style.alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        self.set_style(alpha=value)

    a = alpha
    opacity = alpha

    @property
    def root_colour(self) -> Any:
        """Root marker colour, falling back to the line colour when unset."""
        return self._style.effective_root_colour

    @root_colour.setter
    def root_colour(self, value: Any) -> None:
        self.set_style(root_colour=value)

    root_color = root_colour
    root_c = root_colour
    rc = root_colour

    @property
    def root_alpha(self) -> float:
        """Root marker opacity, falling back to the line opacity when unset."""
        return self._style.effective_root_alpha

    @root_alpha.setter
    def root_alpha(self, value: float) -> None:
        self.set_style(root_alpha=value)

    root_a = root_alpha
    ra = root_alpha

    @property
    def root_size(self) -> float:
        """Root marker point size (sphere radius on the k3d backend)."""
        if self._root is not None:
            return get_marker_size(self._root)
        return self._style.root_size

    @root_size.setter
    def root_size(self, value: float) -> None:
        self.set_style(root_size=value)

    root_s = root_size
    rs = root_size

    @property
    def show_root(self) -> bool:
        """Whether the root marker is included in :attr:`actors`."""
        return self._style.show_root

    @show_root.setter
    def show_root(self, value: bool) -> None:
        self.set_style(show_root=bool(value))

    @property
    def cmap(self) -> str | None:
        """Colour map name used for scalar colouring, or None."""
        return self._style.cmap

    @cmap.setter
    def cmap(self, value: str) -> None:
        self._style.cmap = value
        self._apply_scalars()

    # --- scalar colouring ---

    @property
    def scalars(self) -> np.ndarray | None:
        """Per-edge scalar values driving the colour map, or None."""
        return self._scalars

    def colour_by(
        self,
        values: str | np.ndarray,
        *,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        reduce: EdgeReduce = "mean",
        tree: _Tree | None = None,
    ) -> TreePlot3D:
        """Colour line segments by scalar values instead of a flat colour.

        Parameters
        ----------
        values : str | ndarray
            Graph property name, or an array of values. Edge-level values are used
            directly; vertex-level values are reduced to edges using *reduce*.
        cmap : str, optional
            Any vedo/matplotlib colour map name. By default "viridis".
        vmin, vmax : float, optional
            Colour map limits. By default the data range.
        reduce : {"mean", "min", "max", "start", "stop"}, optional
            How to turn per-vertex values into per-edge values. By default "mean".
        tree : _Tree | None, optional
            Tree to read a named property from. Defaults to the bound tree.

        Returns
        -------
        TreePlot3D
            Self for chaining.
        """
        self._scalars = self._resolve_edge_values(values, reduce=reduce, tree=tree)
        self._scalar_limits = (vmin, vmax)
        self._style.cmap = cmap
        self._apply_scalars()
        return self

    color_by = colour_by

    def clear_scalars(self) -> TreePlot3D:
        """Drop scalar colouring and restore the flat :attr:`colour`."""
        self._scalars = None
        self._style.cmap = None
        self._apply_style()
        return self

    def _apply_scalars(self) -> None:
        """Apply the stored scalar colour map to the lines actor, if possible."""
        if self._lines is None or self._scalars is None or self._style.cmap is None:
            return
        vmin, vmax = self._scalar_limits
        self._lines.cmap(self._style.cmap, self._scalars, on="cells", vmin=vmin, vmax=vmax)

    def _resolve_edge_values(
        self,
        values: str | np.ndarray,
        *,
        reduce: EdgeReduce,
        tree: _Tree | None,
    ) -> np.ndarray:
        """Return per-edge scalar values from a property name or raw array."""
        if isinstance(values, str):
            tree = self._resolve_tree(tree)
            if tree.has_property(values, "e"):
                arr = np.asarray(tree.get_property(values, "e"), dtype=float)
            elif tree.has_property(values, "v"):
                arr = self._reduce_to_edges(
                    tree, np.asarray(tree.get_property(values, "v"), dtype=float), reduce
                )
            else:
                raise KeyError(f"No vertex or edge property named {values!r} on this tree")
            return arr

        arr = np.asarray(values, dtype=float).ravel()
        tree = self.tree if tree is None else tree
        if tree is not None and arr.size == tree.graph.num_vertices():
            return self._reduce_to_edges(tree, arr, reduce)
        return arr

    @staticmethod
    def _reduce_to_edges(tree: _Tree, vertex_values: np.ndarray, reduce: EdgeReduce) -> np.ndarray:
        """Collapse per-vertex values onto edges using the endpoints of each edge."""
        edges = np.asarray(tree.get_edge_indices())
        pairs = vertex_values[edges]
        reducers = {
            "mean": lambda p: p.mean(axis=1),
            "min": lambda p: p.min(axis=1),
            "max": lambda p: p.max(axis=1),
            "start": lambda p: p[:, 0],
            "stop": lambda p: p[:, 1],
        }
        if reduce not in reducers:
            raise ValueError(f"reduce must be one of {sorted(reducers)}, got {reduce!r}")
        return reducers[reduce](pairs)

    # --- display ---

    def show(self, **plot_kwargs) -> Any:
        """Display this plot on its own, building actors first if needed.

        Parameters
        ----------
        **plot_kwargs
            Forwarded to :meth:`neurosetta.Viewer.show`.

        Returns
        -------
        Any
            The :class:`~neurosetta.ops.plotting.viewer.Viewer` for desktop
            backends, or the inline widget for notebook backends such as k3d.
        """
        from ...config.vedo_settings import is_notebook_vedo_backend
        from .viewer import Viewer

        if not self.is_built:
            self.build()

        view = Viewer()
        view.add(*self.actors)
        shown = view.show(**plot_kwargs)
        if is_notebook_vedo_backend():
            return shown
        shown.close()
        return view

    def __repr__(self) -> str:
        """Return a concise summary of binding, build state, and style."""
        tree = "" if self._tree_id is None else f"tree={self._tree_id}, "
        state = "unbuilt" if not self.is_built else f"segments={self.n_segments}"
        if self._scalars is not None and self._style.cmap is not None:
            colouring = f"cmap={self._style.cmap!r}"
        else:
            colouring = f"colour={self._style.colour!r}"
        return (
            f"TreePlot3D({tree}{state}, {colouring}, "
            f"lw={self._style.lw}, alpha={self._style.alpha})"
        )


def _first_given(*values: Any) -> Any:
    """Return the first value that is not the "argument omitted" sentinel."""
    for value in values:
        if value is not _SENTINEL:
            return value
    return _SENTINEL


def _coalesce(*values: Any) -> Any:
    """Return the first non-None value, or None."""
    for value in values:
        if value is not None:
            return value
    return None
