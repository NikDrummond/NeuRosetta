"""3D viewer for neuron trees and forests."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from vedo.plotter import Plotter

from ...config import get_settings, sync_vedo_runtime
from ...core import _Forest, _Tree
from .utils import TreePlot3D

Actor = Any


@dataclass(slots=True)
class Viewer:
    """Interactive 3D viewer for neuron trees and forests using vedo.

    Parameters
    ----------
    size : tuple[int, int], optional
        Window size in pixels (width, height). By default (1200, 800).
    bg : str | Sequence[float], optional
        Background color. By default "white".
    bg2 : str | Sequence[float] | None, optional
        Second background color for gradient. By default None.
    axes : int | bool | str, optional
        Axes style. By default 0 (no axes).
    title : str, optional
        Window title. By default "".
    offscreen : bool, optional
        Whether to render offscreen. By default False.
    interactive : bool, optional
        Whether to run in interactive mode. By default True.
    """

    # ---- user-facing defaults (None → global vedo settings) ----
    size: tuple[int, int] | None = None
    bg: str | Sequence[float] | None = None
    bg2: str | Sequence[float] | None = None
    axes: int | bool | str = 0
    title: str = ""
    offscreen: bool | None = None
    interactive: bool = True

    # ---- internal state ----
    _plotter: Plotter = field(init=False, repr=False)
    _actors: list[Actor] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    ### lifecycle

    def __post_init__(self) -> None:
        """Initialize the vedo Plotter."""
        vedo_cfg = get_settings().vedo
        sync_vedo_runtime(vedo_cfg)
        if self.size is None:
            self.size = vedo_cfg.window_size
        if self.bg is None:
            self.bg = vedo_cfg.default_bg
        if self.offscreen is None:
            self.offscreen = vedo_cfg.offscreen
        self._plotter = Plotter(
            size=self.size,
            bg=self.bg,
            bg2=self.bg2,
            axes=self.axes,
            title=self.title,
            offscreen=self.offscreen,
        )

    def close(self) -> None:
        """Close the plotter window."""
        if not self._closed:
            self._plotter.close()
            self._closed = True

    def clear(self) -> None:
        """Remove all actors from the scene."""
        if not self._actors:
            return

        try:
            self._plotter.remove(self._actors)
        except Exception:
            for actor in self._actors:
                with contextlib.suppress(Exception):
                    self._plotter.remove(actor)

        self._actors.clear()

    def reset_camera(self) -> None:
        """Reset the camera to default position."""
        self._plotter.reset_camera()

    ### context manager

    def __enter__(self) -> Viewer:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """Exit the context manager and close the viewer."""
        self.close()

    ### scene construction

    def add(self, *actors: Actor) -> Viewer:
        """Add one or more actors to the scene.

        Parameters
        ----------
        *actors : Actor
            Vedo actors to add.

        Returns
        -------
        Viewer
            Self for method chaining.
        """
        if not actors:
            return self

        payload = list(actors) if len(actors) > 1 else actors[0]
        self._plotter.add(payload)
        self._actors.extend(actors)
        return self

    def show(
        self,
        *actors: Actor,
        interactive: bool | None = None,
        resetcam: bool = False,
        zoom: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Show the scene.

        Parameters
        ----------
        *actors : Actor
            Additional actors to add before showing.
        interactive : bool | None, optional
            Override interactive mode. By default uses viewer's setting.
        resetcam : bool, optional
            If True, reset camera before showing. By default False.
        zoom : float | None, optional
            Camera zoom factor. By default None.
        **kwargs : Any
            Additional keyword arguments passed to vedo show.

        Returns
        -------
        The vedo Plotter show return value.
        """
        if actors:
            self.add(*actors)

        if resetcam:
            self.reset_camera()

        if zoom is not None:
            self._plotter.camera.Zoom(float(zoom))

        return self._plotter.show(
            interactive=self.interactive if interactive is None else interactive,
            **kwargs,
        )

    def screenshot(self, path: str, scale: float = 1.0) -> str:
        """Save a screenshot to file.

        Parameters
        ----------
        path : str
            Output file path.
        scale : float, optional
            Scale factor for screenshot. By default 1.0.

        Returns
        -------
        str
            The saved file path.
        """
        self._plotter.screenshot(path, scale=scale)
        return path

    @property
    def plotter(self) -> Plotter:
        """Explicit access to the underlying vedo Plotter."""
        return self._plotter

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying vedo Plotter."""
        return getattr(self._plotter, name)

    ### Add neuron

    def add_neuron(
        self,
        tree: _Tree,
        show_root: bool = True,
        cache: bool = True,
        line_kwargs: dict | None = None,
        root_kwargs: dict | None = None,
        force_refresh: bool = False,
    ) -> None:
        """Add a neuron tree to the viewer.

        Parameters
        ----------
        tree : _Tree
            The tree to plot.
        show_root : bool, optional
            Whether to show the root (soma) marker. By default True.
        cache : bool, optional
            Cache the TreePlot3D object on the tree as `tree._3d_plot`.
            By default True.
        line_kwargs : dict | None, optional
            Keyword arguments passed to the vedo Lines constructor. By default None.
        root_kwargs : dict | None, optional
            Keyword arguments passed to the vedo Point constructor. By default None.
        force_refresh : bool, optional
            Force a rebuild of the plot objects even if a cached version exists.
            By default False.
        """
        # Use cached plot if available and no refresh forced
        if not force_refresh and tree.plot3d is not None:
            plot = tree.plot3d
        else:
            plot = TreePlot3D(tree=tree, line_kwargs=line_kwargs, root_kwargs=root_kwargs)

        if cache:
            tree.plot3d = plot

        # set soma
        plot.show_root = show_root

        self.add(plot.actors)

    def add_forest(
        self,
        forest: _Forest,
        force_refresh: bool = False,
        **build_kwargs,
    ) -> None:
        """Add all neurons in a forest to the viewer.

        Parameters
        ----------
        forest : _Forest
            Forest containing trees to plot.
        force_refresh : bool, optional
            Force rebuild of 3D objects. By default False.
        **build_kwargs : dict
            Additional keyword arguments passed to forest.build_3d().
        """
        forest.build_3d(force_refresh=force_refresh, **build_kwargs, show_progress=None)

        # Batch all actors into a single add call
        all_actors = []
        for tree in forest:
            if tree.plot3d is not None:
                all_actors.extend(tree.plot3d.actors)

        self.add(all_actors)
