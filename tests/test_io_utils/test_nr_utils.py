from tempfile import TemporaryDirectory
from pathlib import Path
from NeuRosetta.api import Tree
from NeuRosetta.io.io_utils import _base_meta
from NeuRosetta.io import load, save


def test_nr_read_write(simple_tree):
    tree = Tree(ID=1, metadata=_base_meta(), graph=simple_tree)

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        save(tree, path)
        saved_file = path / "1.nr"
        result = load(saved_file)

    assert result.ID == 1, "ID not loaded"
    assert "ID" not in result.metadata
    assert result.metadata["units"] == "dimensionless", "Metadata Units not Loaded"
    assert result.metadata["isReduced"] is False, "Metadata isReduced not Loaded"
    # gp is source of truth
    assert int(result.graph.gp["ID"]) == 1
    assert result.metadata is result.graph.gp["metadata"]


def test_load_set_units_overrides_metadata(simple_tree):
    tree = Tree(ID=1, metadata=_base_meta(), graph=simple_tree)

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        save(tree, path)
        result = load(path / "1.nr", set_units="nm")

    assert result.metadata["units"] == "nanometer"
