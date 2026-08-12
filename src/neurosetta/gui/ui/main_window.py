"""Main window for the Neurosetta GUI application."""

import sys
import warnings
import logging
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox, QColorDialog,  
    QPushButton, QLabel, QInputDialog, QTextEdit, QDockWidget,
    QDialog, QRadioButton, QLineEdit, QDialogButtonBox, QCheckBox, QStyle
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QEvent, QTimer
# from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk import vtkPointPicker
import vedo as vd

from ..config import AppSettings, UI_CONSTANTS
from ..core import NeuroGUIApplication
from ..utils.logging_utils import setup_logging
from .scale_overlay import ScaleOverlay

# Suppress warnings
warnings.filterwarnings('ignore')


class JumpToFileDialog(QDialog):
    """Custom dialog for jumping to a specific file by index or name."""
    
    def __init__(self, parent, total_files: int, current_counter: str):
        """Initialize the dialog.
        
        Args:
            parent: Parent widget
            total_files: Total number of files
            current_counter: Current file counter string (e.g., '3/10')
        """
        super().__init__(parent)
        self.setWindowTitle("Jump to File")
        self.setModal(True)
        self.resize(350, 200)
        
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel(f"Current: {current_counter}\nTotal files: {total_files}")
        layout.addWidget(info_label)
        layout.addSpacing(10)
        
        # Radio buttons for selection type
        self.index_radio = QRadioButton(f"Jump by index (1-{total_files})")
        self.index_radio.setChecked(True)  # Default to index
        layout.addWidget(self.index_radio)
        
        self.filename_radio = QRadioButton("Jump by filename (without extension)")
        layout.addWidget(self.filename_radio)
        layout.addSpacing(10)
        
        # Input field
        input_label = QLabel("Enter value:")
        layout.addWidget(input_label)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter index or filename...")
        layout.addWidget(self.input_field)
        layout.addSpacing(10)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # Focus on input field
        self.input_field.setFocus()
    
    def get_input_data(self) -> tuple[str, bool]:
        """Get the input data from the dialog.
        
        Returns:
            Tuple of (input_text, is_index_mode)
        """
        return self.input_field.text().strip(), self.index_radio.isChecked()


class MainWindow(QMainWindow):
    """Main application window for the Neurosetta GUI."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        # Initialize settings
        self.settings = AppSettings()
        
        # Initialize UI
        self._setup_window()
        self._setup_central_widget()
        self._setup_vtk_widget()
        self._setup_side_panel()
        self._setup_overlays()
        self._setup_menus()
        self._setup_logging()
        
        # Initialize application core
        self._setup_application_core()
        
        # Tool buttons (created dynamically)
        self.reroot_action_btn = None
        self.define_subtree_btn = None
        
        # Initial setup
        self._position_overlays()
    
    def _setup_window(self) -> None:
        """Set up the main window properties."""
        self.setWindowTitle(self.settings.window_title)
    
    def _setup_central_widget(self) -> None:
        """Set up the central widget and main layout."""
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QHBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
    
    def _setup_vtk_widget(self) -> None:
        """Set up the VTK rendering widget."""
        self.vtk_widget = QVTKRenderWindowInteractor(self.centralWidget())
        self.plotter = vd.Plotter(qt_widget=self.vtk_widget, bg='white')
        
        # Set up picker and interactor
        self.picker = vtkPointPicker()
        self.picker.SetTolerance(UI_CONSTANTS['PICKER_TOLERANCE'])
        
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetPicker(self.picker)
        interactor.AddObserver('MouseWheelForwardEvent', self._update_scale_overlay)
        interactor.AddObserver('MouseWheelBackwardEvent', self._update_scale_overlay)
        interactor.Initialize()
        
        # Install event filter for custom interactions
        self.vtk_widget.installEventFilter(self)
        
        # Add to layout
        self.main_layout.addWidget(self.vtk_widget, stretch=1)
    
    def _setup_side_panel(self) -> None:
        """Set up the side panel with controls."""
        side_panel = QWidget()
        layout = QVBoxLayout(side_panel)
        layout.setContentsMargins(*UI_CONSTANTS['BUTTON_MARGINS'])
        
        # Current file label
        layout.addWidget(QLabel('Current File:'))
        self.filename_label = QLabel('')
        self.filename_label.setStyleSheet('font-weight: bold;')
        layout.addWidget(self.filename_label)
        
        # File counter label
        self.file_counter_label = QLabel('')
        self.file_counter_label.setStyleSheet('color: #666; font-size: 11px;')
        layout.addWidget(self.file_counter_label)
        
        # Jump to file button
        self.jump_to_file_btn = QPushButton('Jump to File...')
        self.jump_to_file_btn.clicked.connect(self._show_jump_to_file_dialog)
        layout.addWidget(self.jump_to_file_btn)
        
        # Mesh path button
        self.set_mesh_path_btn = QPushButton('Set Mesh Path...')
        self.set_mesh_path_btn.clicked.connect(self._set_mesh_path)
        layout.addWidget(self.set_mesh_path_btn)
        
        # Show mesh checkbox
        self.show_mesh_checkbox = QCheckBox('Show Mesh')
        self.show_mesh_checkbox.stateChanged.connect(self._on_show_mesh_changed)
        layout.addWidget(self.show_mesh_checkbox)
        
        # Show subtree checkbox
        self.show_subtree_checkbox = QCheckBox('Show Subtree')
        self.show_subtree_checkbox.stateChanged.connect(self._on_show_subtree_changed)
        layout.addWidget(self.show_subtree_checkbox)
        
        # Flag neuron checkbox
        self.flag_neuron_checkbox = QCheckBox('Flag Neuron')
        self.flag_neuron_checkbox.stateChanged.connect(self._on_flag_neuron_changed)
        layout.addWidget(self.flag_neuron_checkbox)
        layout.addSpacing(10)
        
        # Tools section
        tools_label = QLabel('Tools:')
        tools_label.setStyleSheet('font-weight: bold;')
        layout.addWidget(tools_label)
        
        # Tool buttons
        self.reroot_btn = QPushButton('Reroot Neuron')
        self.reroot_btn.clicked.connect(self._start_reroot_mode)
        layout.addWidget(self.reroot_btn)
        
        self.subtree_from_point_btn = QPushButton('Subtree from Point')
        self.subtree_from_point_btn.clicked.connect(self._start_subtree_mode)
        layout.addWidget(self.subtree_from_point_btn)
        
        # Removed Show Current Subtree button - replaced with checkbox above
        
        layout.addStretch(1)
        
        # Auto-save checkbox
        self.auto_save_checkbox = QCheckBox('AutoSave')
        self.auto_save_checkbox.stateChanged.connect(self._on_auto_save_changed)
        layout.addWidget(self.auto_save_checkbox)
        
        # Save buttons
        self.save_as_btn = QPushButton('Save As…')
        self.save_as_btn.clicked.connect(self._save_file_as)
        layout.addWidget(self.save_as_btn)
        
        self.save_btn = QPushButton('Save…')
        self.save_btn.clicked.connect(self._save_current_file)
        layout.addWidget(self.save_btn)
        
        side_panel.setFixedWidth(UI_CONSTANTS['SIDE_PANEL_WIDTH'])
        self.main_layout.addWidget(side_panel)
    
    def _setup_overlays(self) -> None:
        """Set up overlay widgets."""
        # Scale overlay
        self.scale_overlay = ScaleOverlay(self.vtk_widget)
        
        # Navigation buttons
        self.prev_btn = QPushButton(self.vtk_widget)
        self.prev_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))        
        self.prev_btn.clicked.connect(self._navigate_previous)
        self.prev_btn.raise_()
        
        self.next_btn = QPushButton(self.vtk_widget)
        self.next_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight))        
        self.next_btn.clicked.connect(self._navigate_next)
        self.next_btn.raise_()
    
    def _setup_menus(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        file_menu.addAction(QAction('Load File…', self, triggered=self._load_file))
        file_menu.addAction(QAction('Load Folder…', self, triggered=self._load_folder))
        file_menu.addAction(QAction('Save As…', self, triggered=self._save_file_as))
        file_menu.addAction(QAction('Save…', self, triggered=self._save_current_file))
        file_menu.addAction(QAction('Exit', self, triggered=self.close))
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        # Logging dock setup
        self.log_dock = QDockWidget("Console Log", self)
        self.log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.log_console = QTextEdit(self)
        self.log_console.setReadOnly(True)
        self.log_dock.setWidget(self.log_console)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.hide()
        
        log_action = QAction("Show Log", self, checkable=True)
        log_action.toggled.connect(lambda v: self.log_dock.setVisible(v))
        tools_menu.addAction(log_action)
        
        tools_menu.addAction(QAction('Show Current Subtree', self, triggered=self._show_subtree))
        tools_menu.addAction(QAction('Reroot Neuron', self, triggered=self._start_reroot_mode))
        tools_menu.addAction(QAction('Subtree from Point', self, triggered=self._start_subtree_mode))
        
        # Viewer menu
        viewer_menu = menubar.addMenu('Viewer')
        
        # Units submenu
        units_menu = viewer_menu.addMenu('Units')
        self.nm_action = QAction('Nanometers', self, checkable=True, checked=True)
        self.um_action = QAction('Micrometers', self, checkable=True)
        units_menu.addActions([self.nm_action, self.um_action])
        self.nm_action.triggered.connect(lambda: self._set_units('nm'))
        self.um_action.triggered.connect(lambda: self._set_units('µm'))
        
        # Scale bar controls
        self.scale_action = QAction('Toggle Scale Bar', self, checkable=True)
        self.scale_action.toggled.connect(self.scale_overlay.set_visible)
        viewer_menu.addAction(self.scale_action)
        viewer_menu.addAction(QAction('Set Scale Bar Size…', self, triggered=self._set_scale_size))
        
        # Color control
        viewer_menu.addAction(QAction('Set Neuron Colour…', self, triggered=self._set_neuron_color))
    
    def _setup_logging(self) -> None:
        """Set up application logging."""
        setup_logging(self.log_console)
    
    def _setup_application_core(self) -> None:
        """Set up the application core logic."""
        self.app_core = NeuroGUIApplication(self.plotter, self.picker)
        
        # Set up callbacks
        self.app_core.set_selection_changed_callback(self._on_selection_changed)
        self.app_core.set_file_loaded_callback(self._on_file_loaded)
    
    def eventFilter(self, obj, event) -> bool:
        """Handle custom events for the VTK widget.
        
        Args:
            obj: Object that received the event
            event: The event
            
        Returns:
            True if event was handled, False otherwise
        """
        if obj is self.vtk_widget:
            if event.type() == QEvent.Resize:
                self._position_overlays()
            
            elif event.type() == QEvent.MouseMove:
                self.app_core.point_selector.handle_hover(event, self.vtk_widget.height())
                return False
            
            elif (event.type() == QEvent.MouseButtonPress and 
                  event.modifiers() & Qt.ShiftModifier):
                return self.app_core.point_selector.handle_click(event, self.vtk_widget.height())
        
        return super().eventFilter(obj, event)
    
    def _position_overlays(self) -> None:
        """Position overlay widgets."""
        width = self.vtk_widget.width()
        height = self.vtk_widget.height()
        margin = UI_CONSTANTS['OVERLAY_MARGIN']
        
        # Position scale overlay
        self.scale_overlay.position_at_bottom_center(width, height, margin)
        
        # Position navigation buttons
        prev_size = self.prev_btn.sizeHint()
        self.prev_btn.move(margin, height - prev_size.height() - margin)
        
        next_size = self.next_btn.sizeHint()
        self.next_btn.move(width - next_size.width() - margin, height - next_size.height() - margin)
        
        # Position dynamic tool buttons
        if self.reroot_action_btn and self.reroot_action_btn.isVisible():
            self._position_reroot_btn()
        if self.define_subtree_btn and self.define_subtree_btn.isVisible():
            self._position_define_subtree_btn()
    
    def _update_scale_overlay(self, *args) -> None:
        """Update the scale bar overlay display."""
        renderer = self.plotter.renderer
        camera = renderer.GetActiveCamera()
        focal_point = camera.GetFocalPoint()
        
        # Calculate pixel width of scale bar
        scale_length = self.settings.scale_length_nm
        p1 = (*focal_point, 1)
        p2 = (focal_point[0] + scale_length, focal_point[1], focal_point[2], 1)
        
        renderer.SetWorldPoint(*p1)
        renderer.WorldToDisplay()
        d1 = renderer.GetDisplayPoint()
        
        renderer.SetWorldPoint(*p2)
        renderer.WorldToDisplay()
        d2 = renderer.GetDisplayPoint()
        
        pixel_width = abs(d2[0] - d1[0])
        self.scale_overlay.update_scale_display(int(pixel_width))
        self._position_overlays()
    
    # File operations
    def _load_file(self) -> None:
        """Load a single file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open File', '', self.app_core.file_manager.get_file_filter_string()
        )
        if filename:
            if not self.app_core.load_file(filename):
                QMessageBox.critical(self, 'Error', 'Failed to load file')
    
    def _load_folder(self) -> None:
        """Load files from a folder."""
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            if not self.app_core.load_folder(folder):
                QMessageBox.warning(self, 'Warning', 'No supported files found in folder')
    
    def _save_file_as(self) -> None:
        """Save current file with a new name."""
        if not self.app_core.current_neuron:
            QMessageBox.warning(self, 'Warning', 'Nothing to save')
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save As', '', self.app_core.file_manager.get_save_filter_string()
        )
        if filename:
            if not self.app_core.save_file_as(filename):
                QMessageBox.critical(self, 'Error', 'Failed to save file')
    
    def _save_current_file(self) -> None:
        """Save the current file."""
        if not self.app_core.save_current_file():
            QMessageBox.critical(self, 'Error', 'Failed to save file')
    
    # Navigation
    def _navigate_previous(self) -> None:
        """Navigate to previous file."""
        self.app_core.navigate_previous()
    
    def _navigate_next(self) -> None:
        """Navigate to next file."""
        self.app_core.navigate_next()
    
    # Tool operations
    def _start_reroot_mode(self) -> None:
        """Start reroot mode."""
        if not self.app_core.activate_point_selection():
            QMessageBox.warning(self, 'Warning', 'No neuron loaded for rerooting')
            return
        
        # Create reroot action button
        if not self.reroot_action_btn:
            self.reroot_action_btn = QPushButton('Set as Root', self.vtk_widget)
            self.reroot_action_btn.clicked.connect(self._perform_reroot)
            self.reroot_action_btn.raise_()
        
        self.reroot_action_btn.show()
        self.reroot_action_btn.setEnabled(False)
        self._position_reroot_btn()
    
    def _start_subtree_mode(self) -> None:
        """Start subtree definition mode."""
        if not self.app_core.activate_point_selection():
            QMessageBox.warning(self, 'Warning', 'No neuron loaded for subtree operation')
            return
        
        # Create define subtree button
        if not self.define_subtree_btn:
            self.define_subtree_btn = QPushButton('Define Subtree', self.vtk_widget)
            self.define_subtree_btn.clicked.connect(self._perform_subtree)
            self.define_subtree_btn.raise_()
        
        self.define_subtree_btn.show()
        self.define_subtree_btn.setEnabled(False)
        self._position_define_subtree_btn()
    
    def _perform_reroot(self) -> None:
        """Perform neuron rerooting."""
        if self.app_core.reroot_neuron():
            self._cleanup_reroot_mode()
        else:
            QMessageBox.warning(self, 'Warning', 'Failed to reroot neuron')
    
    def _perform_subtree(self) -> None:
        """Perform subtree creation."""
        if self.app_core.create_subtree():
            self._cleanup_subtree_mode()
        else:
            QMessageBox.warning(self, 'Warning', 'Failed to create subtree')
    
    def _show_subtree(self) -> None:
        """Show current subtree."""
        self.app_core.show_subtree()
    
    def _cleanup_reroot_mode(self) -> None:
        """Clean up reroot mode UI."""
        if self.reroot_action_btn:
            self.reroot_action_btn.hide()
    
    def _cleanup_subtree_mode(self) -> None:
        """Clean up subtree mode UI."""
        if self.define_subtree_btn:
            self.define_subtree_btn.hide()
    
    def _position_reroot_btn(self) -> None:
        """Position the reroot action button."""
        if not self.reroot_action_btn:
            return
        
        width = self.vtk_widget.width()
        height = self.vtk_widget.height()
        btn_size = self.reroot_action_btn.sizeHint()
        
        x = (width - btn_size.width()) // 2
        y = height - btn_size.height() - UI_CONSTANTS['OVERLAY_MARGIN']
        self.reroot_action_btn.move(x, y)
    
    def _position_define_subtree_btn(self) -> None:
        """Position the define subtree button."""
        if not self.define_subtree_btn:
            return
        
        width = self.vtk_widget.width()
        height = self.vtk_widget.height()
        btn_size = self.define_subtree_btn.sizeHint()
        
        x = (width - btn_size.width()) // 2
        y = height - btn_size.height() - UI_CONSTANTS['OVERLAY_MARGIN']
        self.define_subtree_btn.move(x, y)
    
    # Settings and viewer controls
    def _set_units(self, units: str) -> None:
        """Set display units."""
        self.settings.set_units(units)
        self.nm_action.setChecked(units == 'nm')
        self.um_action.setChecked(units == 'µm')
        self._update_scale_overlay()
    
    def _set_scale_size(self) -> None:
        """Set scale bar size."""
        current_size = self.settings.get_scale_length()
        size, ok = QInputDialog.getInt(
            self, 'Scale Bar Size', 
            f'Enter size in {self.settings.units}:', 
            current_size, 1
        )
        if ok:
            self.settings.set_scale_length(size)
            self._update_scale_overlay()
    
    def _set_neuron_color(self) -> None:
        """Set neuron color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.app_core.set_neuron_color(color.name())
    
    # Callbacks
    def _on_selection_changed(self) -> None:
        """Handle point selection changes."""
        # Update button states
        if self.reroot_action_btn and self.reroot_action_btn.isVisible():
            self.reroot_action_btn.setEnabled(self.app_core.is_selection_valid_for_reroot())
        
        if self.define_subtree_btn and self.define_subtree_btn.isVisible():
            self.define_subtree_btn.setEnabled(self.app_core.is_selection_valid_for_subtree())
    
    def _on_file_loaded(self) -> None:
        """Handle file loaded event."""
        # Update UI elements
        # filename = self.app_core.get_current_filename()
        self.filename_label.setText(self.app_core.get_current_filename())
        self.file_counter_label.setText(self.app_core.get_file_counter())
        
        # Update navigation buttons
        self.prev_btn.setEnabled(self.app_core.can_navigate_previous())
        self.next_btn.setEnabled(self.app_core.can_navigate_next())
        
        # Update flag checkbox state
        if hasattr(self, 'flag_neuron_checkbox'):
            # Block signals to prevent triggering state change
            self.flag_neuron_checkbox.blockSignals(True)
            self.flag_neuron_checkbox.setChecked(self.app_core.get_flag_state())
            self.flag_neuron_checkbox.blockSignals(False)
        
        # Update overlays
        self._update_scale_overlay()
    
    def _show_jump_to_file_dialog(self) -> None:
        """Show dialog to jump to a specific file by index or name."""
        if not self.app_core.files:
            QMessageBox.warning(self, "No Files", "No files are currently loaded.")
            return
        
        # Get current file info for the dialog
        current_counter = self.app_core.get_file_counter()
        total_files = len(self.app_core.files)
        
        # Create and show custom dialog
        dialog = JumpToFileDialog(self, total_files, current_counter)
        
        if dialog.exec() != QDialog.Accepted:
            return
            
        # Get input data
        text, is_index_mode = dialog.get_input_data()
        
        if not text:
            return
        
        # Handle based on selected mode
        if is_index_mode:
            # Index mode - parse as integer
            try:
                index = int(text)
                if self.app_core.jump_to_file_by_index(index):
                    return  # Success
                else:
                    QMessageBox.warning(
                        self, 
                        "Invalid Index", 
                        f"Index {index} is out of range. Valid range: 1-{total_files}"
                    )
            except ValueError:
                QMessageBox.warning(
                    self, 
                    "Invalid Input", 
                    f"'{text}' is not a valid integer index."
                )
        else:
            # Filename mode
            if self.app_core.jump_to_file_by_name(text):
                return  # Success
            else:
                QMessageBox.warning(
                    self, 
                    "File Not Found", 
                    f"No file found with name '{text}' (case-insensitive, without extension)"
                )
    
    def _set_mesh_path(self) -> None:
        """Open dialog to select mesh directory path."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Mesh Directory",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder_path:
            self.app_core.set_mesh_directory(folder_path)
            logging.info(f"Mesh directory set to: {folder_path}")
    
    def _on_show_mesh_changed(self, state: int) -> None:
        """Handle show mesh checkbox state change.
        
        Args:
            state: Checkbox state (0=unchecked, 2=checked)
        """
        show_mesh = state == 2  # Qt.Checked = 2
        self.app_core.set_show_mesh(show_mesh)

    def _on_show_subtree_changed(self, state: int) -> None:
        """Handle show subtree checkbox state change.
        
        Args:
            state: Checkbox state (0=unchecked, 2=checked)
        """
        self.app_core.set_show_subtree(state == 2)  # Qt.Checked = 2
    
    def _on_flag_neuron_changed(self, state: int) -> None:
        """Handle flag neuron checkbox state change.
        
        Args:
            state: Checkbox state (0=unchecked, 2=checked)
        """
        self.app_core.set_flag_state(state == 2)  # Qt.Checked = 2
    
    def _on_auto_save_changed(self, state: int) -> None:
        """Handle auto-save checkbox state change.
        
        Args:
            state: Checkbox state (0=unchecked, 2=checked)
        """
        self.app_core.set_auto_save(state == 2)  # Qt.Checked = 2
