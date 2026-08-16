"""Global configuration for NeuRosetta."""

from .env import ensure_runtime_env
from .openmp import OpenMPSettings, apply_openmp, openmp_context, openmp_enabled
from .resolve import resolve_max_workers, resolve_parallel, resolve_show_progress
from .settings import (
    ParallelScope,
    ParallelSettings,
    Settings,
    configure,
    get_settings,
    settings,
)
from .vedo_settings import VedoSettings, sync_vedo_runtime

ensure_runtime_env()

__all__ = [
    "OpenMPSettings",
    "ParallelScope",
    "ParallelSettings",
    "Settings",
    "VedoSettings",
    "apply_openmp",
    "configure",
    "get_settings",
    "openmp_context",
    "openmp_enabled",
    "resolve_max_workers",
    "resolve_parallel",
    "resolve_show_progress",
    "settings",
    "sync_vedo_runtime",
]
