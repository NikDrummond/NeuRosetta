import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
from NeuRosetta.classes import Tree
from NeuRosetta.io_utils.swc_utils import _base_meta

def test_nr_read_write(simple_tree):

    # test neuron
    tree = Tree(ID = 1, metadata = _base_meta(), graph = simple_tree)