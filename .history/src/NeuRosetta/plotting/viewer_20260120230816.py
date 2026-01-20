from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from vedo.plotter import Plotter
from vedo import settings

settings.default_backend = "vtk"
settings.use_parallel_projection = True

from ..core import _Tree

Actor = Any 


@dataclass(slots=True)
class Viewer:
    # ---- user-facing defaults ----
    size: tuple[int, int] = (1200, 800)
    bg: str | Sequence[float] = "white"
    bg2: str | Sequence[float] | None = None
    axes: int | bool | str = 0
    title: str = ""
    offscreen: bool = False
    interactive: bool = True

    # ---- internal state ----
    _plotter: Plotter = field(init=False, repr=False)
    _actors: list[Actor] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    ### lifecycle

    def __post_init__(self) -> None:
        self._plotter = Plotter(
            size=self.size,
            bg=self.bg,
            bg2=self.bg2,
            axes=self.axes,
            title=self.title,
            offscreen=self.offscreen,
        )

    def close(self) -> None:
        if not self._closed:
            self._plotter.close()
            self._closed = True

    def clear(self) -> None:
        if not self._actors:
            return

        try:
            self._plotter.remove(self._actors)
        except Exception:
            for actor in self._actors:
                try:
                    self._plotter.remove(actor)
                except Exception:
                    pass

        self._actors.clear()

    def reset_camera(self) -> None:
        self._plotter.reset_camera()

    ### context manager

    def __enter__(self) -> Viewer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    ### scene construction

    def add(self, *actors: Actor) -> Viewer:
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
    ):
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
        self._plotter.screenshot(path, scale=scale)
        return path

    @property
    def plotter(self) -> Plotter:
        """Explicit access to the underlying vedo Plotter."""
        return self._plotter

    def __getattr__(self, name: str):
        """
        Delegate everything else to Plotter.
        Keeps wrapper thin without re-implementing vedo's API.
        """
        return getattr(self._plotter, name)

    ### Add neuron
    def add_neuron()