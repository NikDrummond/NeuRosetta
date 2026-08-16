"""vedo rendering settings stored by NeuRosetta (not auto-applied on configure)."""

from __future__ import annotations

from dataclasses import dataclass

from .defaults import (
    DEFAULT_VEDO_BACKEND,
    DEFAULT_VEDO_BG,
    DEFAULT_VEDO_USE_PARALLEL_PROJECTION,
    DEFAULT_VEDO_WINDOW_SIZE,
)


@dataclass
class VedoSettings:
    backend: str = DEFAULT_VEDO_BACKEND
    use_parallel_projection: bool = DEFAULT_VEDO_USE_PARALLEL_PROJECTION
    offscreen: bool = False
    window_size: tuple[int, int] = DEFAULT_VEDO_WINDOW_SIZE
    default_bg: str = DEFAULT_VEDO_BG
    multi_samples: int | None = None


def is_notebook_vedo_backend(backend: str | None = None) -> bool:
    """Return True when vedo renders inline in notebooks rather than a desktop window."""
    from .settings import get_settings

    name = backend if backend is not None else get_settings().vedo.backend
    return name in {"k3d", "panel"} or name.startswith(("ipyvtk", "trame"))


def sync_vedo_runtime(settings: VedoSettings | None = None) -> None:
    """Push NeuRosetta vedo settings to ``vedo.settings`` before plotting."""
    from vedo import settings as vedo_runtime

    from .settings import get_settings

    cfg = settings if settings is not None else get_settings().vedo
    vedo_runtime.default_backend = cfg.backend
    vedo_runtime.use_parallel_projection = cfg.use_parallel_projection
    if cfg.multi_samples is not None:
        vedo_runtime.multi_samples = cfg.multi_samples
