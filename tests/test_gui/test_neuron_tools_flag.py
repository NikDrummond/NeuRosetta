"""Tests for GUI neuron flag handling via NeuronTools and ApplicationCore."""

from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from neurosetta.api import Tree
from neurosetta.gui.core.application import NeuroGUIApplication
from neurosetta.gui.tools.neuron_tools import NeuronTools
from neurosetta.io import load, save


@pytest.fixture
def neuron_tools() -> NeuronTools:
    return NeuronTools()


def test_flag_roundtrip_via_neuron_tools(simple_tree, neuron_tools):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    neuron_tools.set_neuron(tree)

    assert neuron_tools.get_flag_state() is False

    neuron_tools.update_flag_state(True)
    assert neuron_tools.get_flag_state() is True
    assert tree.metadata["Flag"] is True

    neuron_tools.update_flag_state(False)
    assert neuron_tools.get_flag_state() is False
    assert tree.metadata["Flag"] is False


def test_get_flag_state_without_neuron(neuron_tools):
    assert neuron_tools.get_flag_state() is False


def test_direct_flag_mutation_blocked_after_gui_set(simple_tree, neuron_tools):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    neuron_tools.set_neuron(tree)
    neuron_tools.update_flag_state(True)

    with pytest.raises(KeyError, match="protected"):
        tree.metadata["Flag"] = False


def test_flag_persists_through_nr_save_load(simple_tree, neuron_tools):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    neuron_tools.set_neuron(tree)
    neuron_tools.update_flag_state(True)

    with TemporaryDirectory() as tmpdir:
        save(tree, tmpdir)
        loaded = load(f"{tmpdir}/1.nr")

    neuron_tools.set_neuron(loaded)
    assert neuron_tools.get_flag_state() is True


def test_application_core_syncs_flag_cache(simple_tree):
    app = NeuroGUIApplication(MagicMock(), MagicMock())
    tree = Tree(ID=1, metadata={"units": "nm", "Flag": True}, graph=simple_tree)

    app.current_neuron = tree
    app.neuron_tools.set_neuron(tree)
    app.flag_state = app.neuron_tools.get_flag_state()

    assert app.get_flag_state() is True

    app.set_flag_state(False)
    assert app.get_flag_state() is False
    assert app.neuron_tools.get_flag_state() is False
    assert tree.metadata["Flag"] is False


def test_application_set_flag_noop_without_neuron():
    app = NeuroGUIApplication(MagicMock(), MagicMock())

    app.set_flag_state(True)

    assert app.get_flag_state() is False
