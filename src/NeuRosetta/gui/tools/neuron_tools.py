"""Neuron manipulation tools including rerooting and subtree operations."""

import logging
import numpy as np
# import NeuRosetta as nr
from typing import Any, Optional
from ..utils.helpers import get_mask_node_ind, n_pnt_coords


class NeuronTools:
    """Tools for manipulating neuron structures."""
    
    def __init__(self):
        """Initialize the neuron tools."""
        self.current_neuron = None
    
    def set_neuron(self, neuron: Any) -> None:
        """Set the current neuron for operations.
        
        Args:
            neuron: Neurosetta neuron object
        """
        self.current_neuron = neuron
    
    def reroot_neuron(self, selected_indices: np.ndarray) -> tuple[Any, np.ndarray, np.ndarray]:
        """Reroot the neuron at the selected point.
        
        Args:
            selected_indices: Array of selected point indices (should contain exactly one)
            
        Returns:
            Tuple of (updated_neuron, new_vertex_coords, new_point_coords)
            
        Raises:
            ValueError: If no neuron is set or selection is invalid
        """
        if self.current_neuron is None:
            raise ValueError("No neuron set for rerooting")
        
        if len(selected_indices) != 1:
            raise ValueError("Exactly one point must be selected for rerooting")
        
        # Get the neuron node index for the selected point
        point_mask = np.zeros(len(n_pnt_coords(self.current_neuron)), dtype=bool)
        point_mask[selected_indices[0]] = True
        node_indices = get_mask_node_ind(self.current_neuron, point_mask)
        
        if len(node_indices) != 1:
            raise ValueError("Invalid point selection for rerooting")
        
        root_node_idx = node_indices[0]
        
        # Perform rerooting
        nr.reroot_tree(self.current_neuron, root=root_node_idx, inplace=True, prune=False)
        
        # Get updated coordinates
        new_vertex_coords = np.array(self.current_neuron.get_node_coordinates())
        new_point_coords = n_pnt_coords(self.current_neuron)
        
        logging.info(f"Rerooted neuron at node {root_node_idx}")
        
        return self.current_neuron, new_vertex_coords, new_point_coords
    
    def create_subtree_mask(self, selected_indices: np.ndarray) -> None:
        """Create a subtree mask from the selected point.
        
        Args:
            selected_indices: Array of selected point indices (should contain exactly one)
            
        Raises:
            ValueError: If no neuron is set or selection is invalid
        """
        if self.current_neuron is None:
            raise ValueError("No neuron set for subtree operation")
        
        if len(selected_indices) != 1:
            raise ValueError("Exactly one point must be selected for subtree operation")
        
        # Get the neuron node index for the selected point
        point_mask = np.zeros(len(n_pnt_coords(self.current_neuron)), dtype=bool)
        point_mask[selected_indices[0]] = True
        node_indices = get_mask_node_ind(self.current_neuron, point_mask)
        
        if len(node_indices) != 1:
            raise ValueError("Invalid point selection for subtree operation")
        
        subtree_root_idx = node_indices[0]
        
        # Create subtree mask
        nr.g_subtree_mask(self.current_neuron, subtree_root_idx)
        
        logging.info(f"Created subtree mask from node {subtree_root_idx}")
    
    def get_neuron_info(self) -> dict:
        """Get information about the current neuron.
        
        Returns:
            Dictionary with neuron information
        """
        if self.current_neuron is None:
            return {"status": "No neuron loaded"}
        
        try:
            vertex_coords = self.current_neuron.get_node_coordinates()
            root_idx = self.current_neuron.root_index()
            
            return {
                "status": "Loaded",
                "num_vertices": len(vertex_coords),
                "root_index": root_idx,
                "has_subtree_mask": hasattr(self.current_neuron, '_subtree_mask')
            }
        except Exception as e:
            return {"status": f"Error: {e}"}
    
    def validate_selection_for_reroot(self, selected_indices: np.ndarray) -> bool:
        """Validate if the current selection is valid for rerooting.
        
        Args:
            selected_indices: Array of selected point indices
            
        Returns:
            True if selection is valid for rerooting
        """
        return (self.current_neuron is not None and 
                len(selected_indices) == 1)
    
    def validate_selection_for_subtree(self, selected_indices: np.ndarray) -> bool:
        """Validate if the current selection is valid for subtree operations.
        
        Args:
            selected_indices: Array of selected point indices
            
        Returns:
            True if selection is valid for subtree operations
        """
        return (self.current_neuron is not None and 
                len(selected_indices) == 1)
    
    def set_flag_state(self) -> None:
        """ Boolian graph property for neurons to flag"""
        if not self.current_neuron.check_property('flag','g'):
            self.current_neuron.graph.gp['flag'] = self.current_neuron.graph.new_gp('bool',False)
        
    def update_flag_state(self, flag_state: bool) -> None:
        """ Boolian graph property for neurons to flag"
        Args:
            flag_state: Boolean value to set the flag
        Returns:
            True if flagged, otherwise False
        """
    
        self.current_neuron.graph.gp['flag'] = flag_state

    def get_flag_state(self) -> bool:
        """Get the flag state of the current neuron.
        
        Returns:
            True if the neuron is flagged, otherwise False
        """
        if self.current_neuron is None:
            return False
        
        if not self.current_neuron.check_property('flag','g'):
            raise AttributeError("Flag property not found")
        
        return self.current_neuron.graph.gp['flag']