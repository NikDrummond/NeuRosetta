"""Tests for GUI file manager folder scanning."""

from pathlib import Path

from neurosetta.gui.file_io.file_manager import FileManager


def test_scan_folder_for_files_filters_extensions(tmp_path: Path):
    (tmp_path / "a.nr").write_text("x")
    (tmp_path / "b.swc").write_text("x")
    (tmp_path / "c.csv").write_text("x")
    (tmp_path / "d.txt").write_text("x")
    (tmp_path / ".hidden.nr").write_text("x")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.nr").write_text("x")

    files = FileManager().scan_folder_for_files(str(tmp_path))

    assert [Path(p).name for p in files] == ["a.nr", "b.swc"]


def test_scan_folder_for_files_case_insensitive(tmp_path: Path):
    (tmp_path / "Upper.NR").write_text("x")
    (tmp_path / "mixed.SwC").write_text("x")

    files = FileManager().scan_folder_for_files(str(tmp_path))

    assert {Path(p).name.lower() for p in files} == {"upper.nr", "mixed.swc"}
