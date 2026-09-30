"""Helpers for resolving mesh overlay flags on Tree / Forest plots."""

from __future__ import annotations


def resolve_mesh_overlay(
    mesh: bool | None = None,
    show_mesh: bool | None = None,
) -> bool:
    """Resolve ``mesh=`` / ``show_mesh=`` into whether to overlay ``tree.mesh``.

    Both default to ``None`` (do not overlay). ``True`` / ``False`` must not
    conflict when both are provided. ``show_mesh`` is an alias of ``mesh``.
    """
    if mesh is None and show_mesh is None:
        return False
    if mesh is not None and show_mesh is not None and bool(mesh) != bool(show_mesh):
        raise ValueError(f"Conflicting mesh overlay flags: mesh={mesh!r}, show_mesh={show_mesh!r}")
    flag = show_mesh if mesh is None else mesh
    return bool(flag)
