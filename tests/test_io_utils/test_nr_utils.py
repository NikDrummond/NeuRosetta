import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
from NeuRosetta.api import Tree
from NeuRosetta.io.swc_utils import _base_meta
from NeuRosetta.io import load

def test_nr_read_write(simple_tree):

    # test neuron
    tree = Tree(ID = 1, metadata = _base_meta(), graph = simple_tree)
    
    # write
    with TemporaryDirectory() as tmpdir:
        
        path = Path(tmpdir)
        saved_file = tree.save(path)

        result = load(saved_file)
    
    # assert metadata and ID
    assert result.ID == 1, "ID not loaded"
    assert result.metadata['ID'] == 1, "Metadata ID not loaded"
    assert result.metadata['units'] == 'Undefined', "Metadata Units not Loaded"
    assert result.metadata['isReduced'] == False, "Metadata isReduced not Loaded"


