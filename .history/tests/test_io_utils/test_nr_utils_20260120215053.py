import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
from NeuRosetta.classes import Tree
from NeuRosetta.io_utils.swc_utils import _base_meta
from NeuRosetta.io_utils import load

def test_nr_read_write(simple_tree):

    # test neuron
    tree = Tree(ID = 1, metadata = _base_meta(), graph = simple_tree)

    # write
    with TemporaryDirectory() as tmpdir:
        
        path = Path(tmpdir)
        tree.save()

