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
        saved_file = tree.save(path)

        result = load(saved_file)
    
    # assert metadata and ID
    assert result.ID == 1, "ID loaded"
    assert result.metadata['ID'] == 1
    assert result.metadata['units'] == 'undefined'
    assert result.metadata['isReduced'] == False
    assert result.metadata['file_path'] == path


