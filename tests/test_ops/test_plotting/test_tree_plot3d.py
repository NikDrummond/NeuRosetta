"""Tests for the lazy, styleable TreePlot3D attached to every tree."""

from __future__ import annotations

import numpy as np
import pytest
from vedo import Sphere

from neurosetta.config import configure
from neurosetta.ops.plotting import TreePlot3D
from neurosetta.testing import make_synthetic_forest, make_synthetic_tree
from neurosetta.utils.vedo_utils import resolve_colour


@pytest.fixture
def tree():
    return make_synthetic_tree(n=20, seed=0)


@pytest.fixture
def built(tree):
    tree.make_plot3d()
    return tree.plot3d


### shell state


def test_new_tree_has_bound_unbuilt_plot(tree):
    assert isinstance(tree.plot3d, TreePlot3D)
    assert not tree.plot3d.is_built
    assert tree.plot3d.lines is None
    assert tree.plot3d.root is None
    assert tree.plot3d.actors == []
    assert tree.plot3d.tree is tree
    assert tree.plot3d.tree_id == tree.ID


def test_copied_tree_gets_its_own_plot(tree):
    tree.make_plot3d()
    other = tree.copy()
    assert other.plot3d is not tree.plot3d
    assert not other.plot3d.is_built
    assert other.plot3d.tree is other


def test_make_plot3d_caches_and_reuses(tree):
    first = tree.make_plot3d()
    assert first is tree.plot3d
    assert first.is_built
    assert tree.make_plot3d() is first
    assert tree.make_plot3d().lines is first.lines


def test_make_plot3d_without_cache_leaves_shell_unbuilt(tree):
    plot = tree.make_plot3d(cache=False)
    assert plot is not tree.plot3d
    assert plot.is_built
    assert not tree.plot3d.is_built


def test_make_plot3d_without_cache_inherits_stored_style(tree):
    tree.plot3d.colour = "red"
    tree.plot3d.lw = 3
    plot = tree.make_plot3d(cache=False, lw=5)
    assert plot is not tree.plot3d
    assert not tree.plot3d.is_built
    assert tree.plot3d.lw == 3
    assert np.allclose(plot.lines.color(), resolve_colour("red"))
    assert plot.lw == 5


def test_force_refresh_replaces_geometry(built):
    lines = built.lines
    built.build(force=True)
    assert built.lines is not lines


def test_clear_keeps_style_and_binding(built):
    built.lw = 4
    built.clear()
    assert not built.is_built
    assert built.lw == 4
    assert built.tree is not None
    built.build()
    assert built.is_built
    assert built.lines.lw() == 4


### styling before and after build


def test_style_set_before_build_is_applied_on_build(tree):
    plot = tree.plot3d
    plot.colour = "red"
    plot.lw = 3
    plot.alpha = 0.5
    plot.root_size = 20
    assert not plot.is_built

    tree.make_plot3d()
    assert np.allclose(plot.lines.color(), resolve_colour("red"))
    assert plot.lines.lw() == 3
    assert plot.lines.alpha() == pytest.approx(0.5)
    assert plot.root_size == pytest.approx(20)


def test_style_setters_reach_actors_when_built(built):
    built.colour = "blue"
    built.lw = 2
    built.alpha = 0.25
    assert np.allclose(built.lines.color(), resolve_colour("blue"))
    assert built.lines.lw() == 2
    assert built.lines.alpha() == pytest.approx(0.25)


@pytest.mark.parametrize("alias", ["colour", "color", "c"])
def test_colour_aliases_are_interchangeable(built, alias):
    setattr(built, alias, "green")
    assert np.allclose(built.colour_rgb, resolve_colour("green"))


def test_lw_and_alpha_aliases(built):
    built.linewidth = 5
    built.opacity = 0.4
    assert built.lw == 5
    assert built.a == pytest.approx(0.4)
    built.Line_width = 2
    built.a = 0.2
    assert built.lw == 2
    assert built.alpha == pytest.approx(0.2)


def test_root_style_aliases(built):
    built.style(root_c="yellow", root_a=0.3, root_s=9)
    assert np.allclose(built.root.color(), resolve_colour("yellow"))
    assert built.root_alpha == pytest.approx(0.3)
    assert built.root_size == pytest.approx(9)

    built.rc = "blue"
    built.ra = 0.7
    built.rs = 11
    assert np.allclose(built.root.color(), resolve_colour("blue"))
    assert built.root_a == pytest.approx(0.7)
    assert built.root_s == pytest.approx(11)


def test_style_kwargs_aliases(built):
    built.style(Line_width=4, a=0.6, rc="green", ra=0.1, rs=6)
    assert built.lw == 4
    assert built.alpha == pytest.approx(0.6)
    assert np.allclose(built.root.color(), resolve_colour("green"))
    assert built.root_alpha == pytest.approx(0.1)
    assert built.root_size == pytest.approx(6)


def test_bulk_style_is_chainable(built):
    assert built.style(colour="orange", lw=2, alpha=0.9, root_size=8) is built
    assert built.get_style()["colour"] == "orange"
    assert built.root_size == pytest.approx(8)


def test_show_root_controls_actor_list(built):
    assert len(built.actors) == 2
    built.show_root = False
    assert built.root is None
    assert built.actors == [built.lines]
    built.show_root = True
    assert built.root is not None
    assert len(built.actors) == 2


def test_root_follows_line_style_until_overridden(built):
    built.set_style(colour="blue", alpha=0.4)
    assert np.allclose(built.root.color(), built.colour_rgb)
    assert built.root_alpha == pytest.approx(0.4)

    built.root_colour = "yellow"
    assert np.allclose(built.root.color(), resolve_colour("yellow"))

    built.set_style(root_colour=None, root_alpha=None)
    assert np.allclose(built.root.color(), built.colour_rgb)
    assert built.root_alpha == pytest.approx(0.4)


def test_raw_vedo_kwargs_split_into_style_and_extras(tree):
    plot = tree.make_plot3d(line_kwargs={"c": "purple", "lw": 5, "res": 4}, root_kwargs={"r": 7})
    assert plot.lw == 5
    assert plot.root_size == pytest.approx(7)
    assert plot.get_style()["line_kwargs"] == {"res": 4}

    plot.rebuild()
    assert plot.lw == 5
    assert plot.root_size == pytest.approx(7)


def test_random_colour_is_stable_across_cached_builds(tree):
    tree.make_plot3d(random_c=True)
    colour = tuple(tree.plot3d.lines.color())
    tree.make_plot3d(random_c=True)
    assert tuple(tree.plot3d.lines.color()) == colour


def test_random_colour_is_seedable(tree):
    a = tree.make_plot3d(random_c=True, seed=3).colour
    b = make_synthetic_tree(n=5, seed=1).make_plot3d(random_c=True, seed=3).colour
    assert a == b


### rebuild


def test_rebuild_preserves_style_without_tree_argument(built):
    built.style(colour="orange", lw=2, alpha=0.9)
    built.rebuild()
    assert np.allclose(built.lines.color(), resolve_colour("orange"))
    assert built.lines.lw() == 2
    assert built.lines.alpha() == pytest.approx(0.9)


def test_rebuild_picks_up_new_geometry(tree):
    plot = tree.make_plot3d()
    segments = plot.n_segments
    tree.graph.remove_edge(next(tree.graph.edges()))
    plot.rebuild()
    assert plot.n_segments == segments - 1


def test_unbound_plot_raises_a_clear_error():
    with pytest.raises(ValueError, match="not bound to a tree"):
        TreePlot3D().build()


### scalar colouring


def test_colour_by_edge_property(built):
    built.colour_by("Euclidean_length", cmap="plasma")
    assert built.cmap == "plasma"
    assert built.scalars.shape == (built.n_segments,)


def test_colour_by_reduces_vertex_property_to_edges(built):
    built.colour_by("radius", reduce="max")
    assert built.scalars.shape == (built.n_segments,)


def test_colour_by_works_before_build(tree):
    tree.plot3d.colour_by("Euclidean_length")
    assert not tree.plot3d.is_built
    tree.make_plot3d()
    assert tree.plot3d.cmap == "viridis"
    assert tree.plot3d.scalars.shape == (tree.plot3d.n_segments,)


def test_scalar_colouring_survives_restyle_and_rebuild(built):
    built.colour_by("Euclidean_length")
    built.lw = 4
    assert built.cmap == "viridis"
    built.rebuild()
    assert built.cmap == "viridis"
    assert built.scalars is not None


def test_setting_a_colour_clears_scalar_colouring(built):
    built.colour_by("Euclidean_length")
    built.colour = "red"
    assert built.cmap is None
    assert built.scalars is None


def test_clear_scalars_restores_flat_colour(built):
    built.colour_by("Euclidean_length")
    built.clear_scalars()
    assert built.cmap is None
    assert built.scalars is None


def test_colour_by_unknown_property_raises(built):
    with pytest.raises(KeyError):
        built.colour_by("not_a_property")


def test_colour_by_invalid_reduce_raises(built):
    with pytest.raises(ValueError, match="reduce must be one of"):
        built.colour_by("radius", reduce="median")


### geometry helpers


def test_geometry_helpers(built):
    assert built.n_segments == built.lines.ncells
    assert len(built.bounds()) == 6
    assert built.center().shape == (3,)
    assert len(built.assembly().unpack()) == 2


def test_geometry_helpers_are_none_when_unbuilt(tree):
    assert tree.plot3d.n_segments is None
    assert tree.plot3d.bounds() is None
    assert tree.plot3d.center() is None


def test_copy_shares_style_but_not_actors(built):
    built.lw = 3
    unbuilt = built.copy()
    assert not unbuilt.is_built
    assert unbuilt.get_style() == built.get_style()

    rebuilt = built.copy(build=True)
    assert rebuilt.is_built
    assert rebuilt.lines is not built.lines


def test_repr_reports_build_state(tree):
    assert "unbuilt" in repr(tree.plot3d)
    tree.make_plot3d()
    assert f"tree={tree.ID}" in repr(tree.plot3d)
    assert "segments=" in repr(tree.plot3d)


### backend handling


def test_k3d_backend_uses_a_sphere_root_marker(tree):
    configure(vedo_backend="k3d")
    try:
        plot = tree.make_plot3d(root_size=5)
        assert isinstance(plot.root, Sphere)
        assert plot.root_size == pytest.approx(5)
        plot.root_size = 9
        assert float(plot.root.radius) == pytest.approx(9)
    finally:
        configure(vedo_backend="vtk")


def test_k3d_show_root_false_does_not_create_a_root_actor(tree):
    configure(vedo_backend="k3d")
    try:
        plot = tree.make_plot3d(show_root=False)
        assert plot.root is None
        assert plot.actors == [plot.lines]
        assert not any(isinstance(actor, Sphere) for actor in plot.actors)
    finally:
        configure(vedo_backend="vtk")


### forest integration


def test_forest_build_3d_builds_every_plot():
    forest = make_synthetic_forest(n_trees=3, n=10, seed=1)
    forest.build_3d()
    assert all(t.plot3d.is_built for t in forest)
    assert all(t.plot3d.show_root is False for t in forest)


def test_forest_build_3d_keeps_random_colours_stable_unless_refreshed():
    forest = make_synthetic_forest(n_trees=3, n=10, seed=1)
    forest.build_3d()
    colours = [tuple(t.plot3d.lines.color()) for t in forest]

    forest.build_3d()
    assert [tuple(t.plot3d.lines.color()) for t in forest] == colours

    forest.build_3d(force_refresh=True)
    assert [tuple(t.plot3d.lines.color()) for t in forest] != colours
