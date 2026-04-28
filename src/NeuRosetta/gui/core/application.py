"""Core application logic that coordinates all components."""

import pathlib
import logging
import os
from typing import List, Optional, Any

import numpy as np

from ..config import AppSettings
from ..file_io import FileManager
from ..rendering import NeuronRenderer, PointSelector
from ..tools import NeuronTools

from ...io import import_mesh


class NeuroGUIApplication:
    """Core application logic that manages the neuron visualization application."""
    
    def __init__(self, plotter, picker):
        """Initialize the application core.
        
        Args:
            plotter: vedo Plotter instance for rendering
            picker: VTK point picker for interaction
        """
        # Initialize components
        self.settings = AppSettings()
        self.file_manager = FileManager()
        self.renderer = NeuronRenderer(plotter)
        self.point_selector = PointSelector(plotter, picker)
        self.neuron_tools = NeuronTools()
        
        # Application state
        self.files: List[str] = []
        self.current_file_index = 0
        self.current_neuron = None
        self.vertex_coords = None
        self.point_coords = None
        self.neuron_indices = None
        self.show_subtree = False
        self.flag_state = False
        self.auto_save = False
        self.mesh_directory: Optional[str] = None
        self.show_mesh: bool = False
        self.current_mesh = None
        
        # Subtree state
        self.show_subtree: bool = False
        
        # Set up callbacks
        self.point_selector.set_selection_changed_callback(self._on_selection_changed)
        
        # UI callbacks (to be set by UI layer)
        self.selection_changed_callback = None
        self.file_loaded_callback = None
    
    def set_selection_changed_callback(self, callback) -> None:
        """Set callback for when point selection changes."""
        self.selection_changed_callback = callback
    
    def set_file_loaded_callback(self, callback) -> None:
        """Set callback for when a file is loaded."""
        self.file_loaded_callback = callback
    
    def load_file(self, filepath: str) -> bool:
        """Load a single file.
        
        Args:
            filepath: Path to the file to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.files = [filepath]
            self.current_file_index = 0
            return self._load_current_file()
        except Exception as e:
            logging.error(f"Failed to load file {filepath}: {e}")
            return False
    
    def load_folder(self, folder_path: str) -> bool:
        """Load all supported files from a folder.
        
        Args:
            folder_path: Path to the folder to scan
            
        Returns:
            True if files were found and loaded, False otherwise
        """
        try:
            self.files = self.file_manager.scan_folder_for_files(folder_path)
            self.current_file_index = 0
            
            if self.files:
                return self._load_current_file()
            else:
                logging.warning(f"No supported files found in {folder_path}")
                return False
                
        except Exception as e:
            logging.error(f"Failed to load folder {folder_path}: {e}")
            return False
    
    def save_current_file(self) -> bool:
        """Save the current neuron to its original file location.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.current_neuron or not self.files:
            logging.warning("Nothing to save")
            return False
        
        try:
            current_file = self.files[self.current_file_index]
            directory = self.file_manager.get_directory_from_path(current_file)
            self.file_manager.save_neuron_to_directory(self.current_neuron, directory)
            return True
        except Exception as e:
            logging.error(f"Failed to save current file: {e}")
            return False
    
    def save_file_as(self, filepath: str) -> bool:
        """Save the current neuron to a new file.
        
        Args:
            filepath: Path where to save the file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.current_neuron:
            logging.warning("Nothing to save")
            return False
        
        try:
            self.file_manager.save_neuron(self.current_neuron, filepath)
            return True
        except Exception as e:
            logging.error(f"Failed to save file as {filepath}: {e}")
            return False
    
    def _save_if_needed(self) -> bool:
        """Save current file if auto-save is enabled and there are unsaved changes.
        
        Returns:
            bool: True if save was successful or not needed, False if save failed
        """
        if not self.auto_save or not self.current_neuron:
            return True
            
        try:
            logging.info("Auto-saving current neuron...")
            return self.save_current_file()
        except Exception as e:
            logging.error(f"Auto-save failed: {e}")
            return False
    
    def navigate_previous(self) -> bool:
        """Navigate to the previous file.
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        if self.current_file_index <= 0:
            return False
            
        if not self._save_if_needed():
            return False
            
        self.current_file_index -= 1
        return self._load_current_file()
    
    def navigate_next(self) -> bool:
        """Navigate to the next file.
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        if self.current_file_index >= len(self.files) - 1:
            return False
            
        if not self._save_if_needed():
            return False
            
        self.current_file_index += 1
        return self._load_current_file()
    
    def can_navigate_previous(self) -> bool:
        """Check if previous navigation is possible."""
        return self.current_file_index > 0
    
    def can_navigate_next(self) -> bool:
        """Check if next navigation is possible."""
        return self.current_file_index < len(self.files) - 1
    
    def get_current_filename(self) -> str:
        """Get the current filename."""
        if self.files and 0 <= self.current_file_index < len(self.files):
            return pathlib.Path(self.files[self.current_file_index]).name
        return ""
    
    def get_file_counter(self) -> str:
        """Get the file counter string (e.g., '1/3' or '1/1')."""
        if self.files:
            current = self.current_file_index + 1  # 1-based indexing for display
            total = len(self.files)
            return f"{current}/{total}"
        return ""
    
    def jump_to_file_by_index(self, index: int) -> bool:
        """Jump to a specific file by 1-based index.
        
        Args:
            index: 1-based file index
            
        Returns:
            True if successful, False otherwise
        """
        if not self.files:
            return False
            
        # Convert to 0-based index
        zero_based_index = index - 1
        
        if 0 <= zero_based_index < len(self.files):
            self.current_file_index = zero_based_index
            return self._load_current_file()
        return False
    
    def jump_to_file_by_name(self, filename: str) -> bool:
        """Jump to a specific file by filename (without extension).
        
        Args:
            filename: Filename without extension
            
        Returns:
            True if successful, False otherwise
        """
        if not self.files:
            return False
            
        # import pathlib
        
        # Search for file with matching name (case-insensitive)
        filename_lower = filename.lower()
        
        for i, filepath in enumerate(self.files):
            file_stem = pathlib.Path(filepath).stem.lower()
            if file_stem == filename_lower:
                self.current_file_index = i
                return self._load_current_file()
                
        return False
    
    def activate_point_selection(self) -> bool:
        """Activate point selection mode.
        
        Returns:
            True if activated successfully, False otherwise
        """
        if self.point_coords is not None:
            self.point_selector.activate(self.point_coords)
            # Hide mesh when entering point selection mode
            self._hide_current_mesh()
            return True
        return False
    
    def deactivate_point_selection(self) -> None:
        """Deactivate point selection mode."""
        self.point_selector.deactivate()
        # Restore mesh visibility when exiting point selection mode
        self._update_mesh_visibility()
    
    def reroot_neuron(self) -> bool:
        """Reroot the neuron at the selected point.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            selected_indices = self.point_selector.get_selected_indices()
            
            if not self.neuron_tools.validate_selection_for_reroot(selected_indices):
                logging.warning("Invalid selection for rerooting")
                return False
            
            # Perform rerooting
            self.current_neuron, self.vertex_coords, self.point_coords = \
                self.neuron_tools.reroot_neuron(selected_indices)
            
            # Update neuron indices
            # import Neurosetta as nr
            # self.neuron_indices = nr.g_lb_inds(self.current_neuron)
            self.neuron_indices = self.current_neuron.core_indices()

            # Deactivate selection and re-render
            self.deactivate_point_selection()
            if self.show_subtree:
                self.renderer.render_subtree(self.current_neuron)
            else:
                self.renderer.render_neuron(self.current_neuron)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to reroot neuron: {e}")
            return False
    
    def create_subtree(self) -> bool:
        """Create a subtree from the selected point.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            selected_indices = self.point_selector.get_selected_indices()
            
            if not self.neuron_tools.validate_selection_for_subtree(selected_indices):
                logging.warning("Invalid selection for subtree creation")
                return False
            
            # Create subtree mask
            self.neuron_tools.create_subtree_mask(selected_indices)
            
            # Deactivate selection and show subtree
            self.deactivate_point_selection()
            self.renderer.render_subtree(self.current_neuron)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to create subtree: {e}")
            return False
    
    def set_show_subtree(self, show: bool) -> None:
        """Set whether to show the subtree visualization.
        
        Args:
            show: Whether to show the subtree
        """
        self.show_subtree = show
        if not show:
            # If turning off, just render the full neuron
            if self.current_neuron:
                self.renderer.render_neuron(self.current_neuron)
        elif self.current_neuron:
            # If turning on, render the subtree
            self.renderer.render_subtree(self.current_neuron)
    
    def set_flag_state(self, state: bool) -> None:
        """Set the flag state for the current neuron.
        
        Args:
            state: The new flag state
        """
        if self.current_neuron:
            self.flag_state = state
            self.neuron_tools.update_flag_state(state)
    
    def set_auto_save(self, enabled: bool) -> None:
        """Enable or disable auto-save functionality.
        
        Args:
            enabled: Whether to enable auto-save
        """
        self.auto_save = enabled
        logging.info(f"Auto-save {'enabled' if enabled else 'disabled'}")
    
    def get_flag_state(self) -> bool:
        """Get the current flag state.
        
        Returns:
            The current flag state, or False if no neuron is loaded
        """
        return getattr(self, 'flag_state', False)
    
    def show_subtree(self) -> None:
        """Show the current subtree visualization (for backward compatibility)."""
        self.set_show_subtree(True)
    
    def set_neuron_color(self, color: str) -> None:
        """Set the neuron visualization color.
        
        Args:
            color: Color name or hex string
        """
        self.renderer.set_neuron_color(color)
    
    def get_selection_count(self) -> int:
        """Get the number of currently selected points."""
        return self.point_selector.get_selection_count()
    
    def is_selection_valid_for_reroot(self) -> bool:
        """Check if current selection is valid for rerooting."""
        selected_indices = self.point_selector.get_selected_indices()
        return self.neuron_tools.validate_selection_for_reroot(selected_indices)
    
    def is_selection_valid_for_subtree(self) -> bool:
        """Check if current selection is valid for subtree operations."""
        selected_indices = self.point_selector.get_selected_indices()
        return self.neuron_tools.validate_selection_for_subtree(selected_indices)
    
    def _load_current_file(self) -> bool:
        """Load the file at the current index.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.files or self.current_file_index >= len(self.files):
            return False
        
        filepath = self.files[self.current_file_index]
        
        try:
            # Load file data
            neuron, vertex_coords, point_coords = self.file_manager.load_file(filepath)
            
            # Update application state
            self.current_neuron = neuron
            self.vertex_coords = vertex_coords
            self.point_coords = point_coords
            
            if neuron is not None:
                # Neuron file

                self.neuron_indices = neuron.core_indices()
                self.neuron_tools.set_neuron(neuron)
                # Initialize and get flag state for the new neuron
                self.neuron_tools.set_flag_state()
                self.flag_state = self.neuron_tools.get_flag_state()

                # render neuron or subtree
                if self.show_subtree:
                    self.renderer.render_subtree(self.current_neuron)
                else:
                    self.renderer.render_neuron(self.current_neuron)
            else:
                # Point cloud file
                self.neuron_indices = None
                self.neuron_tools.set_neuron(None)
                self.flag_state = False

                # always render point clouds for non-neuron files?
                self.renderer.render_point_cloud(self.vertex_coords)

            # Load mesh if enabled
            if self.current_neuron is not None and self.show_mesh and not self.point_selector.is_active:
                self._load_current_mesh()
            
            # Notify UI
            if self.file_loaded_callback:
                self.file_loaded_callback()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to load file {filepath}: {e}")
            return False
    
    def _on_selection_changed(self) -> None:
        """Handle point selection changes."""
        if self.selection_changed_callback:
            self.selection_changed_callback()
    
    def set_mesh_directory(self, directory_path: str) -> None:
        """Set the directory path for mesh files.
        
        Args:
            directory_path: Path to the directory containing mesh files
        """
        self.mesh_directory = directory_path
        # Try to load mesh for current neuron if show_mesh is enabled
        if self.show_mesh and self.files:
            self._load_current_mesh()
    
    def set_show_mesh(self, show_mesh: bool) -> None:
        """Set whether to show mesh alongside neuron.
        
        Args:
            show_mesh: True to show mesh, False to hide
        """
        self.show_mesh = show_mesh
        
        if show_mesh:
            # Load and display mesh if not in point selection mode
            if not self.point_selector.is_active:
                self._load_current_mesh()
        else:
            # Hide current mesh
            self._hide_current_mesh()
    
    def _load_current_mesh(self) -> None:
        """Load mesh file corresponding to current neuron."""
        if not self.mesh_directory or not self.files:
            return
        
        # Don't show mesh when in point selection mode
        if self.point_selector.is_active:
            return
        
        try:
            # import os
            # from pathlib import Path
            # import Neurosetta
            
            # Get current filename without extension
            current_file = self.files[self.current_file_index]
            filename_stem = pathlib.Path(current_file).stem
            
            # Try to find mesh file with same name in mesh directory
            mesh_path = None
            mesh_dir = pathlib.Path(self.mesh_directory)
            
            # Common mesh file extensions supported by vedo
            mesh_extensions = ['.obj', '.ply', '.stl', '.vtk', '.vtp', '.vtu', '.off', '.3ds']
            
            for ext in mesh_extensions:
                potential_path = mesh_dir / f"{filename_stem}{ext}"
                if potential_path.exists():
                    mesh_path = str(potential_path)
                    break
            
            if mesh_path:
                # Load mesh
                mesh = import_mesh(mesh_path).mesh
                mesh.alpha(0.3)
                mesh.c('gray')
                
                # Hide previous mesh if it exists
                self._hide_current_mesh()
                
                # Store and display new mesh
                self.current_mesh = mesh
                self.renderer.add_mesh(mesh)
                
                logging.info(f"Loaded mesh: {mesh_path}")
            else:
                logging.warning(f"No mesh file found for neuron: {filename_stem}")
                
        except Exception as e:
            logging.error(f"Failed to load mesh: {e}")
    
    def _hide_current_mesh(self) -> None:
        """Hide the currently displayed mesh."""
        if self.current_mesh:
            try:
                self.renderer.remove_mesh(self.current_mesh)
                self.current_mesh = None
            except Exception as e:
                logging.error(f"Failed to hide mesh: {e}")
    
    def _update_mesh_visibility(self) -> None:
        """Update mesh visibility based on current state."""
        # Hide mesh if in point selection mode, regardless of show_mesh setting
        if self.point_selector.is_active:
            self._hide_current_mesh()
        elif self.show_mesh:
            self._load_current_mesh()
