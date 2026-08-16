"""Environment-variable configuration for NeuRosetta."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .defaults import ENV_PREFIX, OMP_WAIT_POLICY

if TYPE_CHECKING:
    from .settings import Settings

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def ensure_runtime_env() -> None:
    """Set process environment defaults used by NeuRosetta dependencies."""
    os.environ.setdefault("OMP_WAIT_POLICY", OMP_WAIT_POLICY)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    raise ValueError(f"Invalid boolean environment value: {value!r}")


def _env(name: str) -> str | None:
    return os.environ.get(f"{ENV_PREFIX}{name}")


def load_env_into_settings(settings: Settings) -> None:
    """Apply ``NEUROSETTA_*`` environment variables to *settings*."""
    if (value := _env("PARALLEL")) is not None:
        flag = _parse_bool(value)
        settings.parallel.io = flag
        settings.parallel.forest = flag
        settings.parallel.default = flag
    if (value := _env("PARALLEL_IO")) is not None:
        settings.parallel.io = _parse_bool(value)
    if (value := _env("PARALLEL_FOREST")) is not None:
        settings.parallel.forest = _parse_bool(value)
    if (value := _env("PARALLEL_DEFAULT")) is not None:
        settings.parallel.default = _parse_bool(value)
    if (value := _env("MAX_WORKERS")) is not None:
        settings.parallel.max_workers = int(value)
    if (value := _env("SHOW_PROGRESS")) is not None:
        settings.parallel.show_progress = _parse_bool(value)
    if (value := _env("DEFAULT_UNITS")) is not None:
        settings.default_units = value
    if (value := _env("GT_OPENMP_THREADS")) is not None:
        settings.openmp.num_threads = int(value)
    if (value := _env("GT_OPENMP_SCHEDULE")) is not None:
        settings.openmp.schedule = value
    if (value := _env("GT_OPENMP_SCHEDULE_CHUNK")) is not None:
        settings.openmp.schedule_chunk = int(value)
    if (value := _env("GT_OPENMP_THRESH")) is not None:
        settings.openmp.thresh = int(value)
    if (value := _env("VEDO_BACKEND")) is not None:
        settings.vedo.backend = value
    if (value := _env("VEDO_PARALLEL_PROJECTION")) is not None:
        settings.vedo.use_parallel_projection = _parse_bool(value)
    if (value := _env("VEDO_OFFSCREEN")) is not None:
        settings.vedo.offscreen = _parse_bool(value)
