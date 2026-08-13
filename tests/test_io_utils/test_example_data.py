from pathlib import Path

from neurosetta.api import Forest, Tree
from neurosetta.io.example_data import example_data_dir, load_example_data


def test_example_data_dir_nr():
    data_dir = example_data_dir()
    assert data_dir.is_dir()
    assert data_dir.name == "nr"
    assert list(data_dir.glob("*.nr"))


def test_example_data_dir_swc():
    data_dir = example_data_dir(format="swc")
    assert data_dir.is_dir()
    assert data_dir.name == "swc"
    assert list(data_dir.glob("*.swc"))


def test_load_example_data_all():
    result = load_example_data()
    assert isinstance(result, Forest)
    assert len(result) == 4


def test_load_example_data_single():
    tree = load_example_data(720575940596125868)
    assert isinstance(tree, Tree)
    assert tree.ID == 720575940596125868


def test_example_data_dir_matches_docs_data():
    repo_nr = Path(__file__).resolve().parents[2] / "docs" / "data" / "nr"
    assert sorted(p.name for p in example_data_dir().glob("*.nr")) == sorted(
        p.name for p in repo_nr.glob("*.nr")
    )

    repo_swc = Path(__file__).resolve().parents[2] / "docs" / "data" / "swc"
    assert sorted(p.name for p in example_data_dir(format="swc").glob("*.swc")) == sorted(
        p.name for p in repo_swc.glob("*.swc")
    )
