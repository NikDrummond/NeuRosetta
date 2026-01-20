# tests/test_io_utils/test_swc_utils.py
import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
import pandas as pd
from NeuRosetta.io_utils.swc_utils import _check_swc_columns, _table_from_swc, write_swc, import_swc
from NeuRosetta.classes import Tree_graph

def test_check_swc_columns_valid():
    df = pd.DataFrame({
        "node_id": [1, 2],
        "type": [1, 1],
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "z": [0.0, 1.0],
        "radius": [0.5, 0.5],
        "parent_id": [-1, 1]
    })
    _check_swc_columns(df)  # Should not raise

def test_check_swc_columns_missing():
    df = pd.DataFrame({"node_id": [1, 2], "type": [1, 1]})
    with pytest.raises(ValueError) as exc_info:
        _check_swc_columns(df)
    assert "Missing required columns" in str(exc_info.value)

def test_table_from_swc(tmp_path):
    # Create a temporary SWC file
    swc_content = "1 1 0.0 0.0 0.0 0.5 -1\n2 1 1.0 1.0 1.0 0.5 1\n"
    swc_file = tmp_path / "test.swc"
    swc_file.write_text(swc_content)
    
    df = _table_from_swc(str(swc_file))
    assert len(df) == 2
    assert list(df.columns) == ["node_id", "type", "x", "y", "z", "radius", "parent_id"]

def test_swc_read_write(simple_tree):

    # test neuron
    tree = Tree_graph(ID = 1, metadata = {}, graph = simple_tree)

    with TemporaryDirectory() as tmpdir:

        path = Path(tmpdir) / "1.swc"

        # Write then read
        write_swc(path, tree)
        result = import_swc(path)

        # asserts
        

