"""Logging utilities for the Neurosetta GUI application."""

import logging
from PySide6.QtWidgets import QTextEdit
from PySide6.QtCore import QTimer
from ..config import LOGGING_CONFIG


class QTextEditLogger(logging.Handler):
    """Custom logging handler that outputs to a QTextEdit widget."""
    
    def __init__(self, text_edit: QTextEdit):
        """Initialize the logger handler.
        
        Args:
            text_edit: QTextEdit widget to display log messages
        """
        super().__init__()
        self.widget = text_edit
        
        # Set up formatter
        formatter = logging.Formatter(
            LOGGING_CONFIG['FORMAT'],
            datefmt=LOGGING_CONFIG['DATE_FORMAT']
        )
        self.setFormatter(formatter)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the QTextEdit widget.
        
        Args:
            record: The log record to emit
        """
        msg = self.format(record)
        
        def append_message():
            """Append message to the text widget and scroll to bottom."""
            self.widget.append(msg)
            scrollbar = self.widget.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        
        # Use QTimer to ensure GUI thread safety
        QTimer.singleShot(0, append_message)


def setup_logging(text_edit: QTextEdit = None) -> None:
    """Set up application logging.
    
    Args:
        text_edit: Optional QTextEdit widget for GUI logging
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, LOGGING_CONFIG['LEVEL']))
    
    # Add text edit handler if provided
    if text_edit:
        handler = QTextEditLogger(text_edit)
        logger.addHandler(handler)
    
    # Add console handler for development
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        LOGGING_CONFIG['FORMAT'],
        datefmt=LOGGING_CONFIG['DATE_FORMAT']
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
