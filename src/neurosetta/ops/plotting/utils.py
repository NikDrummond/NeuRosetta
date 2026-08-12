"""Vedo-based 3D plot helpers for neuron trees."""

from __future__ import annotations

from typing import Any

from numpy.random import choice
from vedo import Lines, Point

from ...core import _Tree


class TreePlot3D:
    """3D plot representation of a neuron tree using vedo primitives.

    Parameters
    ----------
    tree : _Tree | None
        Tree to plot. If None, creates an empty plot shell.
    show_root : bool, optional
        Include root marker in :attr:`actors`. By default True.
    line_kwargs : dict | None, optional
        Keyword arguments forwarded to the vedo Lines constructor.
        By default None.
    root_kwargs : dict | None, optional
        Keyword arguments forwarded to the vedo Point constructor.
        By default None.
    random_c : bool, optional
        Pick a random line colour. By default False.
    """

    def __init__(
        self,
        tree: _Tree | None,
        show_root: bool = True,
        line_kwargs: dict | None = None,
        root_kwargs: dict | None = None,
        random_c: bool = False,
    ) -> None:
        """Initialize vedo plot actors from a tree.

        See the class docstring for parameter descriptions.
        """
        self._show_root = show_root
        self._lines: Lines | None = None
        self._root: Point | None = None

        if tree is not None:
            c = self._random_c() if random_c else "k"
            line_kwargs = {"c": c, "lw": 1, "alpha": 1.0} | (line_kwargs or {})
            root_kwargs = {"r": 12, "c": c, "alpha": 1.0} | (root_kwargs or {})
            self._lines = self._make_lines(tree, line_kwargs)
            self._root = self._make_root(tree, root_kwargs)

    # Private builders

    @staticmethod
    def _make_lines(tree: _Tree, kwargs: dict) -> Lines:
        """Build a vedo Lines object from tree edge coordinates."""
        starts, stops = tree.get_edge_coordinates()
        return Lines(starts, stops, **kwargs)

    @staticmethod
    def _make_root(tree: _Tree, kwargs: dict) -> Point:
        """Build a vedo Point object at the tree root."""
        r_coords = tree.get_node_coordinates(subset=tree.get_root_index())
        return Point(r_coords, **kwargs)

    @staticmethod
    def _random_c() -> list:
        """Return a random RGB colour as a list of three ints in [0, 255]."""
        return list(choice(range(256), size=3))

    # Read-only properties
    @property
    def lines(self) -> Lines | None:
        """Vedo Lines actor for tree edges, or None if empty."""
        return self._lines

    @property
    def root(self) -> Point | None:
        """Vedo Point actor at the tree root, or None if empty."""
        return self._root

    @property
    def show_root(self) -> bool:
        """Whether the root marker is included in :attr:`actors`."""
        return self._show_root

    @show_root.setter
    def show_root(self, value: bool) -> None:
        """Set whether the root marker is included in :attr:`actors`."""
        self._show_root = value

    @property
    def color(self) -> Any | None:
        """Current line colour, or None if no lines actor exists."""
        return self._lines.color() if self._lines is not None else None

    # Aliases
    colour = color
    c = color

    @property
    def lw(self) -> float | None:
        """Current line width, or None if no lines actor exists."""
        return self._lines.lw() if self._lines is not None else None

    @property
    def alpha(self) -> float | None:
        """Current line opacity, or None if no lines actor exists."""
        return self._lines.alpha() if self._lines is not None else None

    # Alias
    a = alpha

    @property
    def root_size(self) -> float | None:
        """Current root marker radius, or None if no root actor exists."""
        return self._root.ps() if self._root is not None else None

    @property
    def actors(self) -> list[Lines | Point | None]:
        """Flattened list of vedo actors ready to pass to a Plotter."""
        actors = [self._lines]
        if self._show_root:
            actors.append(self._root)
        return actors

    # Style

    def style(
        self,
        *,
        colour: Any | None = None,
        lw: float | None = None,
        alpha: float | None = None,
        root_size: float | None = None,
        show_root: bool | None = None,
        random_c: bool = False,
    ) -> TreePlot3D:
        """Update visual style in bulk.  Returns self for chaining.

        Parameters
        ----------
        colour :
            Any colour specifier accepted by vedo. Overridden by *random_c*.
        lw : float, optional
            Line width. By default unchanged.
        alpha : float, optional
            Opacity in [0, 1]. By default unchanged.
        root_size : float, optional
            Point radius of the root marker. By default unchanged.
        show_root : bool, optional
            Whether to include the root marker in :attr:`actors`.
            By default unchanged.
        random_c : bool, optional
            When True, pick a random RGB colour (takes priority over *colour*).
            By default False.

        Returns
        -------
        TreePlot3D
            Self for chaining.
        """
        if random_c:
            colour = self._random_c()

        if colour is not None:
            self._lines.color(colour)
            self._root.color(colour)
        if lw is not None:
            self._lines.lw(lw)
        if alpha is not None:
            self._lines.alpha(alpha)
            self._root.alpha(alpha)
        if root_size is not None:
            self._root.ps(root_size)
        if show_root is not None:
            self._show_root = show_root

        return self

    # inplace rebuild
    def rebuild(
        self,
        tree: _Tree,
        line_kwargs: dict | None = None,
        root_kwargs: dict | None = None,
    ) -> TreePlot3D:
        """Regenerate vedo objects from an updated tree in place.

        Preserves the current colour, line width, alpha, and root size unless
        *line_kwargs* / *root_kwargs* override them explicitly.

        Parameters
        ----------
        tree : _Tree
            Updated tree.
        line_kwargs : dict | None, optional
            Override kwargs for the vedo Lines constructor. By default None.
        root_kwargs : dict | None, optional
            Override kwargs for the vedo Point constructor. By default None.

        Returns
        -------
        TreePlot3D
            Self for chaining.
        """
        # Snapshot current style before rebuilding
        current_colour = self.color
        current_lw = self.lw
        current_alpha = self.alpha
        current_root_size = self.root_size

        self._lines = self._make_lines(tree, line_kwargs or {})
        self._root = self._make_root(tree, root_kwargs or {})

        # Reapply style, letting explicit kwargs take precedence
        self.style(
            colour=current_colour,
            lw=current_lw,
            alpha=current_alpha,
            root_size=current_root_size,
        )
        return self

    def __repr__(self) -> str:
        """Return a concise summary of the current plot style."""
        return f"TreePlot3D(color={self.color}, lw={self.lw}, alpha={self.alpha})"
