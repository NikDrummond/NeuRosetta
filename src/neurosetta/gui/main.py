"""Main entry point for the modular Neurosetta GUI application."""
import os
os.environ["NO_AT_BRIDGE"] = "1"

import sys
from PySide6.QtWidgets import QApplication
from .ui import MainWindow


def start_GUI():
    """Main application entry point."""

    app = QApplication.instance()
    if app is None:
        # Create Qt application
        app = QApplication(sys.argv)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())
    # app.exec_()
