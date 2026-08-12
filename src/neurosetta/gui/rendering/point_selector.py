"""Point selection and interaction handling for neuron visualization."""

import logging
import numpy as np
import vedo as vd
from typing import Optional, Callable, List
from vtk import vtkPointPicker
from PySide6.QtCore import QEvent
from ..config import RENDERING_CONSTANTS
from ..utils.helpers import make_pnts


class PointSelector:
    """Handles point selection and interaction in 3D neuron visualizations."""
    
    def __init__(self, plotter: vd.Plotter, picker: vtkPointPicker):
        """Initialize the point selector.
        
        Args:
            plotter: vedo Plotter instance
            picker: VTK point picker for 3D interaction
        """
        self.plotter = plotter
        self.picker = picker
        
        # Point selection state
        self.point_coords = None
        self.point_mask = None
        self.selected_points_in = None
        self.selected_points_out = None
        self.hover_marker = None
        self.is_active = False
        
        # Callbacks
        self.selection_changed_callback: Optional[Callable] = None
    
    def set_selection_changed_callback(self, callback: Callable) -> None:
        """Set callback function to be called when selection changes.
        
        Args:
            callback: Function to call when selection changes
        """
        self.selection_changed_callback = callback
    
    def activate(self, point_coords: np.ndarray) -> None:
        """Activate point selection mode.
        
        Args:
            point_coords: Array of 3D point coordinates for selection
        """
        self.point_coords = point_coords
        self.point_mask = np.zeros(len(point_coords), dtype=bool)
        self.is_active = True
        
        # Create hover marker
        self.hover_marker = vd.Point(
            [0, 0, 0], 
            c=RENDERING_CONSTANTS['HOVER_POINT_COLOR'], 
            r=RENDERING_CONSTANTS['HOVER_POINT_RADIUS'], 
            alpha=RENDERING_CONSTANTS['HOVER_POINT_ALPHA']
        )
        self.plotter.add(self.hover_marker)
        
        # Update point overlays
        self._update_point_overlays()
        self.plotter.render()
        
        logging.info(f"Activated point selection with {len(point_coords)} points")
    
    def deactivate(self) -> None:
        """Deactivate point selection mode."""
        self.is_active = False
        
        # Remove all selection overlays
        self._remove_overlays()
        self.plotter.render()
        
        # Clear state
        self.point_coords = None
        self.point_mask = None
        
        logging.info("Deactivated point selection")
    
    def handle_hover(self, event: QEvent, widget_height: int) -> None:
        """Handle mouse hover events for point highlighting.
        
        Args:
            event: Qt mouse event
            widget_height: Height of the widget for coordinate conversion
        """
        if not self.is_active or self.point_coords is None:
            return
        
        # Convert Qt coordinates to VTK coordinates
        x = event.x()
        y = widget_height - event.y()
        
        # Pick point in 3D space
        self.picker.Pick(x, y, 0, self.plotter.renderer)
        picked_pos = np.array(self.picker.GetPickPosition())
        
        # Find closest point
        distances = np.linalg.norm(self.point_coords - picked_pos, axis=1)
        closest_idx = distances.argmin()
        
        # Update hover marker
        if distances[closest_idx] <= RENDERING_CONSTANTS['HOVER_DISTANCE_THRESHOLD']:
            self.hover_marker.pos(self.point_coords[closest_idx])
            self.hover_marker.alpha(RENDERING_CONSTANTS['HOVER_POINT_ALPHA'])
        else:
            self.hover_marker.alpha(0)  # Hide marker
        
        self.plotter.render()
    
    def handle_click(self, event: QEvent, widget_height: int) -> bool:
        """Handle mouse click events for point selection.
        
        Args:
            event: Qt mouse event
            widget_height: Height of the widget for coordinate conversion
            
        Returns:
            True if event was handled, False otherwise
        """
        if not self.is_active or self.point_coords is None:
            return False
        
        # Convert Qt coordinates to VTK coordinates
        x = event.x()
        y = widget_height - event.y()
        
        # Pick point in 3D space
        self.picker.Pick(x, y, 0, self.plotter.renderer)
        picked_pos = np.array(self.picker.GetPickPosition())
        
        # Find closest point
        distances = np.linalg.norm(self.point_coords - picked_pos, axis=1)
        closest_idx = distances.argmin()
        
        # Toggle selection if within threshold
        if distances[closest_idx] <= RENDERING_CONSTANTS['HOVER_DISTANCE_THRESHOLD']:
            self.point_mask[closest_idx] = not self.point_mask[closest_idx]
            self._update_point_overlays()
            
            # Notify callback of selection change
            if self.selection_changed_callback:
                self.selection_changed_callback()
            
            logging.debug(f"Toggled selection for point {closest_idx}")
            return True
        
        return False
    
    def get_selected_indices(self) -> np.ndarray:
        """Get indices of currently selected points.
        
        Returns:
            Array of selected point indices
        """
        if self.point_mask is None:
            return np.array([], dtype=int)
        return np.where(self.point_mask)[0]
    
    def get_selection_count(self) -> int:
        """Get number of currently selected points.
        
        Returns:
            Number of selected points
        """
        if self.point_mask is None:
            return 0
        return np.sum(self.point_mask)
    
    def clear_selection(self) -> None:
        """Clear all point selections."""
        if self.point_mask is not None:
            self.point_mask.fill(False)
            self._update_point_overlays()
            
            if self.selection_changed_callback:
                self.selection_changed_callback()
    
    def _update_point_overlays(self) -> None:
        """Update the visual representation of selected/unselected points."""
        if self.point_coords is None or self.point_mask is None:
            return
        
        # Remove existing overlays
        if self.selected_points_in:
            self.plotter.remove(self.selected_points_in)
        if self.selected_points_out:
            self.plotter.remove(self.selected_points_out)
        
        # Create new overlays
        self.selected_points_in, self.selected_points_out = make_pnts(
            self.point_coords, self.point_mask
        )
        
        # Add to plotter
        self.plotter.add(self.selected_points_out)
        self.plotter.add(self.selected_points_in)
        self.plotter.render()
    
    def _remove_overlays(self) -> None:
        """Remove all selection overlays from the plotter."""
        overlays = [
            self.selected_points_in,
            self.selected_points_out,
            self.hover_marker
        ]
        
        for overlay in overlays:
            if overlay:
                self.plotter.remove(overlay)
        
        # Clear references
        self.selected_points_in = None
        self.selected_points_out = None
        self.hover_marker = None
