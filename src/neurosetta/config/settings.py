"""Global NeuRosetta settings (mutable singleton)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from .defaults import DEFAULT_UNITS, PARALLEL_SCOPE_DEFAULTS
from .env import ensure_runtime_env, load_env_into_settings
from .openmp import OpenMPSchedule, OpenMPSettings, apply_openmp
from .vedo_settings import VedoSettings

ParallelScope = Literal["io", "forest", "default"]

_settings: Settings | None = None
_context_stack: list[dict] = []
_env_loaded = False


@dataclass
class ParallelSettings:
    io: bool = PARALLEL_SCOPE_DEFAULTS["io"]
    forest: bool = PARALLEL_SCOPE_DEFAULTS["forest"]
    default: bool = PARALLEL_SCOPE_DEFAULTS["default"]
    max_workers: int | None = None
    show_progress: bool = False


@dataclass
class Settings:
    parallel: ParallelSettings = field(default_factory=ParallelSettings)
    openmp: OpenMPSettings = field(default_factory=OpenMPSettings)
    vedo: VedoSettings = field(default_factory=VedoSettings)
    default_units: str = DEFAULT_UNITS


def get_settings() -> Settings:
    """Return the global settings singleton."""
    global _settings, _env_loaded
    ensure_runtime_env()
    if _settings is None:
        _settings = Settings()
    if not _env_loaded:
        load_env_into_settings(_settings)
        apply_openmp(_settings.openmp)
        _env_loaded = True
    return _settings


def configure(
    *,
    parallel: bool | None = None,
    parallel_io: bool | None = None,
    parallel_forest: bool | None = None,
    parallel_default: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool | None = None,
    default_units: str | None = None,
    openmp_num_threads: int | None = None,
    openmp_schedule: OpenMPSchedule | None = None,
    openmp_schedule_chunk: int | None = None,
    openmp_thresh: int | None = None,
    vedo_backend: str | None = None,
    vedo_parallel_projection: bool | None = None,
    vedo_offscreen: bool | None = None,
    vedo_window_size: tuple[int, int] | None = None,
    vedo_bg: str | None = None,
    vedo_multi_samples: int | None = None,
    apply_openmp_now: bool = True,
) -> Settings:
    """Update global settings in place and return the singleton."""
    s = get_settings()
    if parallel is not None:
        s.parallel.io = parallel
        s.parallel.forest = parallel
        s.parallel.default = parallel
    if parallel_io is not None:
        s.parallel.io = parallel_io
    if parallel_forest is not None:
        s.parallel.forest = parallel_forest
    if parallel_default is not None:
        s.parallel.default = parallel_default
    if max_workers is not None:
        s.parallel.max_workers = max_workers
    if show_progress is not None:
        s.parallel.show_progress = show_progress
    if default_units is not None:
        s.default_units = default_units
    if openmp_num_threads is not None:
        s.openmp.num_threads = openmp_num_threads
    if openmp_schedule is not None:
        s.openmp.schedule = openmp_schedule
    if openmp_schedule_chunk is not None:
        s.openmp.schedule_chunk = openmp_schedule_chunk
    if openmp_thresh is not None:
        s.openmp.thresh = openmp_thresh
    if vedo_backend is not None:
        s.vedo.backend = vedo_backend
    if vedo_parallel_projection is not None:
        s.vedo.use_parallel_projection = vedo_parallel_projection
    if vedo_offscreen is not None:
        s.vedo.offscreen = vedo_offscreen
    if vedo_window_size is not None:
        s.vedo.window_size = vedo_window_size
    if vedo_bg is not None:
        s.vedo.default_bg = vedo_bg
    if vedo_multi_samples is not None:
        s.vedo.multi_samples = vedo_multi_samples
    if apply_openmp_now:
        apply_openmp(s.openmp)
    return s


def _current_context() -> dict:
    merged: dict = {}
    for layer in _context_stack:
        merged.update(layer)
    return merged


def _context_overrides(
    *,
    parallel: bool | None = None,
    parallel_io: bool | None = None,
    parallel_forest: bool | None = None,
    parallel_default: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool | None = None,
) -> dict:
    overrides: dict = {}
    if parallel is not None:
        overrides["parallel_io"] = parallel
        overrides["parallel_forest"] = parallel
        overrides["parallel_default"] = parallel
    if parallel_io is not None:
        overrides["parallel_io"] = parallel_io
    if parallel_forest is not None:
        overrides["parallel_forest"] = parallel_forest
    if parallel_default is not None:
        overrides["parallel_default"] = parallel_default
    if max_workers is not None:
        overrides["max_workers"] = max_workers
    if show_progress is not None:
        overrides["show_progress"] = show_progress
    return overrides


@contextmanager
def settings(
    *,
    parallel: bool | None = None,
    parallel_io: bool | None = None,
    parallel_forest: bool | None = None,
    parallel_default: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool | None = None,
):
    """Temporarily override settings for the duration of the ``with`` block."""
    overrides = _context_overrides(
        parallel=parallel,
        parallel_io=parallel_io,
        parallel_forest=parallel_forest,
        parallel_default=parallel_default,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    _context_stack.append(overrides)
    try:
        yield get_settings()
    finally:
        _context_stack.pop()


def _reset_settings_for_tests() -> None:
    """Replace the singleton with fresh defaults (tests only)."""
    global _settings, _env_loaded
    _settings = Settings()
    _env_loaded = True
    _context_stack.clear()
