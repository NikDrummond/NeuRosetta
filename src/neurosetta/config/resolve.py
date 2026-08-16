"""Resolve effective settings from explicit values, context, and globals."""

from __future__ import annotations

from .settings import ParallelScope, _current_context, get_settings


def resolve_parallel(*, explicit: bool | None, scope: ParallelScope) -> bool:
    """Return whether to run work in parallel for *scope*."""
    if explicit is not None:
        return explicit

    ctx = _current_context()
    scope_key = f"parallel_{scope}"
    if scope_key in ctx:
        return ctx[scope_key]

    return getattr(get_settings().parallel, scope)


def resolve_max_workers(*, explicit: int | None) -> int | None:
    """Return the thread-pool worker limit."""
    if explicit is not None:
        return explicit

    ctx = _current_context()
    if "max_workers" in ctx:
        return ctx["max_workers"]

    return get_settings().parallel.max_workers


def resolve_show_progress(*, explicit: bool | None) -> bool:
    """Return whether to show a tqdm progress bar."""
    if explicit is not None:
        return explicit

    ctx = _current_context()
    if "show_progress" in ctx:
        return ctx["show_progress"]

    return get_settings().parallel.show_progress
