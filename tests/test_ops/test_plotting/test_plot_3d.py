"""Tests that plot_3d / Tree.show_3d use the tree's plot3d handle."""

from __future__ import annotations

import numpy as np
import pytest

from neurosetta.config import configure, get_settings
from neurosetta.ops.plotting import plot_3d
from neurosetta.testing import make_synthetic_tree
from neurosetta.utils.vedo_utils import resolve_colour


@pytest.fixture(autouse=True)
def _offscreen():
    previous = get_settings().vedo.offscreen
    configure(vedo_offscreen=True)
    yield
    configure(vedo_offscreen=previous)


@pytest.fixture
def tree():
    return make_synthetic_tree(n=12, seed=0)


def _show(tree, **kwargs):
    return plot_3d(tree, plot_kwargs={"interactive": False}, **kwargs)


def test_show_3d_builds_the_tree_plot(tree):
    view = _show(tree)
    assert tree.plot3d.is_built
    assert tree.plot3d.tree is tree
    assert len(view._actors) == 2
    view.close()


def test_show_3d_honours_stored_style(tree):
    tree.plot3d.colour = "red"
    tree.plot3d.lw = 3
    tree.plot3d.show_root = False
    _show(tree)
    assert np.allclose(tree.plot3d.lines.color(), resolve_colour("red"))
    assert tree.plot3d.lw == 3
    assert tree.plot3d.actors == [tree.plot3d.lines]


def test_show_3d_does_not_inject_legacy_k4_defaults(tree):
    tree.plot3d.colour = "red"
    _show(tree)
    assert tree.plot3d.colour == "red"
    assert np.allclose(tree.plot3d.lines.color(), resolve_colour("red"))


def test_show_3d_kwargs_override_and_persist(tree):
    tree.plot3d.colour = "red"
    _show(tree, colour="blue", lw=4)
    assert np.allclose(tree.plot3d.lines.color(), resolve_colour("blue"))
    assert tree.plot3d.lw == 4


def test_show_3d_without_cache_does_not_mutate_the_shell(tree):
    tree.plot3d.colour = "red"
    tree.plot3d.lw = 3
    view = _show(tree, cache=False, lw=5)
    assert not tree.plot3d.is_built
    assert tree.plot3d.lw == 3
    assert tree.plot3d.colour == "red"
    assert np.allclose(view._actors[0].color(), resolve_colour("red"))
    assert view._actors[0].lw() == 5
    view.close()


def test_tree_show_3d_is_plot_3d(tree):
    assert type(tree).show_3d is plot_3d
