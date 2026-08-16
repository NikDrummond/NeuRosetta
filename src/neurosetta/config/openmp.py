"""graph_tool OpenMP runtime settings."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal

OpenMPSchedule = Literal["static", "dynamic", "guided", "auto"]


@dataclass
class OpenMPSettings:
    num_threads: int | None = None
    schedule: OpenMPSchedule | None = None
    schedule_chunk: int = 0
    thresh: int | None = None


def apply_openmp(settings: OpenMPSettings | None = None) -> None:
    """Push NeuRosetta OpenMP settings to graph_tool."""
    from graph_tool import openmp as gt_openmp

    from .settings import get_settings

    cfg = settings if settings is not None else get_settings().openmp
    if cfg.num_threads is not None:
        gt_openmp.openmp_set_num_threads(cfg.num_threads)
    if cfg.schedule is not None:
        gt_openmp.openmp_set_schedule(cfg.schedule, cfg.schedule_chunk)
    if cfg.thresh is not None:
        gt_openmp.openmp_set_thresh(cfg.thresh)


def openmp_context(
    *,
    num_threads: int | None = None,
    schedule: OpenMPSchedule | None = None,
    schedule_chunk: int = 0,
    thresh: int | None = None,
) -> AbstractContextManager[None]:
    """Temporary graph_tool OpenMP overrides."""
    from graph_tool import openmp as gt_openmp

    return gt_openmp.openmp_context(
        nthreads=num_threads,
        schedule=schedule,
        chunk=schedule_chunk,
        thresh=thresh,
    )


def openmp_enabled() -> bool:
    """Return whether graph_tool was compiled with OpenMP support."""
    from graph_tool import openmp as gt_openmp

    return gt_openmp.openmp_enabled()
