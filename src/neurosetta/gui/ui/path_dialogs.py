"""Path / directory picker dialogs tuned for large directories.

Uses native OS file dialogs for browsing (non-native QFileDialog segfaults with
the VTK/vedo render widget). For huge directories, paste a path instead of browsing.
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import AppSettings


def get_open_file_path(
    parent: QWidget,
    caption: str,
    start_dir: str,
    name_filter: str,
) -> str:
    """Open a native file dialog and return the selected path (or "")."""
    filename, _ = QFileDialog.getOpenFileName(
        parent,
        caption,
        start_dir or "",
        name_filter,
    )
    return filename or ""


def get_save_file_path(
    parent: QWidget,
    caption: str,
    start_dir: str,
    name_filter: str,
) -> str:
    """Open a native save dialog and return the selected path (or "")."""
    filename, _ = QFileDialog.getSaveFileName(
        parent,
        caption,
        start_dir or "",
        name_filter,
    )
    return filename or ""


def browse_existing_directory(parent: QWidget, caption: str, start_dir: str) -> str:
    """Browse for a directory using the native OS dialog."""
    return (
        QFileDialog.getExistingDirectory(
            parent,
            caption,
            start_dir or "",
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        or ""
    )


class DirectoryPickerDialog(QDialog):
    """Directory picker with path paste, recent paths, and optional browse.

    Paste/type a path to skip listing huge directories in the file dialog.
    """

    def __init__(self, parent: QWidget, title: str, settings: AppSettings):
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(560, 180)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Directory path (paste to skip browsing large folders):"))
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("/path/to/folder")
        start = settings.last_directory
        if start:
            self.path_edit.setText(start)
        path_row.addWidget(self.path_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        recent = [p for p in settings.recent_directories() if os.path.isdir(p)]
        layout.addWidget(QLabel("Recent:"))
        self.recent_combo = QComboBox()
        self.recent_combo.setEditable(False)
        self.recent_combo.addItem("(select recent…)", "")
        for path in recent:
            self.recent_combo.addItem(path, path)
        self.recent_combo.currentIndexChanged.connect(self._on_recent_chosen)
        layout.addWidget(self.recent_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.path_edit.returnPressed.connect(self._accept_if_valid)
        self.path_edit.setFocus()

    def _browse(self) -> None:
        start = self.path_edit.text().strip() or self.settings.last_directory
        chosen = browse_existing_directory(self, self.windowTitle(), start)
        if chosen:
            self.path_edit.setText(chosen)

    def _on_recent_chosen(self, index: int) -> None:
        path = self.recent_combo.itemData(index)
        if path:
            self.path_edit.setText(str(path))

    def _accept_if_valid(self) -> None:
        path = self.selected_path()
        if not path:
            QMessageBox.warning(self, "Missing path", "Enter or browse to a directory.")
            self.path_edit.setFocus()
            return
        # Expand ~ for convenience
        path = os.path.expanduser(path)
        self.path_edit.setText(path)
        if not os.path.isdir(path):
            QMessageBox.warning(
                self,
                "Invalid directory",
                f"Not a directory:\n{path}",
            )
            self.path_edit.selectAll()
            self.path_edit.setFocus()
            return
        self.accept()

    def selected_path(self) -> str:
        """Return the stripped directory path from the line edit."""
        return os.path.expanduser(self.path_edit.text().strip())
