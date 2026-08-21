"""GUI renderer behaviour around TreePlot3D."""

from __future__ import annotations

import numpy as np
import pytest
from vedo import Plotter

from neurosetta.config import configure, get_settings
from neurosetta.gui.config.constants import RENDERING_CONSTANTS
from neurosetta.gui.rendering.renderer import NeuronRenderer
from neurosetta.testing import make_synthetic_tree
from neurosetta.utils.vedo_utils import resolve_colour


@pytest.fixture
def renderer():
    previous = get_settings().vedo.offscreen
    configure(vedo_offscreen=True, vedo_backend="vtk")
    plotter = Plotter(offscreen=True, size=(200, 200))
    try:
        yield NeuronRenderer(plotter)
    finally:
        configure(vedo_offscreen=previous, vedo_backend="vtk")


@pytest.fixture
def tree():
    return make_synthetic_tree(n=12, seed=0)


def test_render_neuron_uses_stored_colour(renderer, tree):
    renderer.neuron_color = "red"
    renderer.render_neuron(tree)
    assert np.allclose(renderer.current_lines.color(), resolve_colour("red"))
    assert np.allclose(renderer.soma.color(), resolve_colour("red"))


def test_set_neuron_color_survives_rerender(renderer, tree):
    renderer.render_neuron(tree)
    renderer.set_neuron_color("blue")
    renderer.render_neuron(tree)
    assert renderer.neuron_color == "blue"
    assert np.allclose(renderer.current_lines.color(), resolve_colour("blue"))
    assert np.allclose(renderer.soma.color(), resolve_colour("blue"))


def test_default_render_colour_matches_constant(renderer, tree):
    renderer.render_neuron(tree)
    expected = resolve_colour(RENDERING_CONSTANTS["DEFAULT_NEURON_COLOR"])
    assert np.allclose(renderer.current_lines.color(), expected)
