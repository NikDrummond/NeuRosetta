"""Scale bar overlay widget for the 3D viewer."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt
from ..config import AppSettings


class ScaleOverlay(QWidget):
    """Overlay widget that displays a scale bar on the 3D viewer."""
    
    def __init__(self, parent=None):
        """Initialize the scale overlay.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.settings = AppSettings()
        self._setup_ui()
        self._setup_styling()
    
    def _setup_ui(self) -> None:
        """Set up the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Scale bar line
        self.bar_line = QFrame(self)
        self.bar_line.setFrameShape(QFrame.HLine)
        self.bar_line.setLineWidth(2)
        layout.addWidget(self.bar_line, alignment=Qt.AlignCenter)
        
        # Scale label
        self.scale_label = QLabel(self)
        layout.addWidget(self.scale_label, alignment=Qt.AlignCenter)
        
        # Initially hidden
        self.setVisible(False)
    
    def _setup_styling(self) -> None:
        """Set up widget styling."""
        # Make overlay transparent for mouse events
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        # Ensure overlay stays on top
        self.raise_()
    
    def update_scale_display(self, pixel_width: int) -> None:
        """Update the scale bar display.
        
        Args:
            pixel_width: Width of the scale bar in pixels
        """
        # Set the bar width
        self.bar_line.setFixedWidth(max(1, pixel_width))
        
        # Update the label text
        self.scale_label.setText(self.settings.get_scale_display_text())
    
    def set_visible(self, visible: bool) -> None:
        """Set the visibility of the scale overlay.
        
        Args:
            visible: Whether the overlay should be visible
        """
        self.setVisible(visible)
    
    def position_at_bottom_center(self, parent_width: int, parent_height: int, margin: int = 10) -> None:
        """Position the overlay at the bottom center of the parent widget.
        
        Args:
            parent_width: Width of the parent widget
            parent_height: Height of the parent widget
            margin: Margin from the bottom edge
        """
        size_hint = self.sizeHint()
        x = (parent_width - size_hint.width()) // 2
        y = parent_height - size_hint.height() - margin
        self.move(x, y)
