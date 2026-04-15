"""Helper functions for neuron data manipulation and visualization."""

import numpy as np
import vedo as vd
# import NeuRosetta as nr
from typing import Tuple, Any
from ..config import FILE_CONSTANTS



def n_pnt_coords(neuron: Any) -> np.ndarray:
    """Get point coordinates for a neuron.
    
    Args:
        neuron: Neurosetta neuron object
        
    Returns:
        Array of point coordinates
    """
    return neuron.get_node_coordinates()[neuron.get_core_inds()]


def make_pnts(coords: np.ndarray, mask: np.ndarray) -> Tuple[vd.Points, vd.Points]:
    """Create vedo Points objects for selected and unselected points.
    
    Args:
        coords: Array of point coordinates
        mask: Boolean mask indicating selected points
        
    Returns:
        Tuple of (selected_points, unselected_points) as vedo.Points objects
    """
    selected_points = vd.Points(coords[mask], c='b', r=8)
    unselected_points = vd.Points(coords[~mask], c='r', r=8)
    return selected_points, unselected_points


def get_mask_node_ind(neuron: Any, mask: np.ndarray) -> np.ndarray:
    """Get node indices corresponding to masked points.
    
    Args:
        neuron: Neurosetta neuron object
        mask: Boolean mask for points
        
    Returns:
        Array of node indices
    """
    return neuron.core_indices()[mask]


def validate_csv_data(df) -> bool:
    """Validate that CSV data contains required columns.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        True if valid, False otherwise
    """
    required_cols = FILE_CONSTANTS['REQUIRED_CSV_COLUMNS']
    return all(col in df.columns for col in required_cols)


def extract_coordinates_from_csv(df) -> np.ndarray:
    """Extract coordinate array from CSV DataFrame.
    
    Args:
        df: Pandas DataFrame with x, y, z columns
        
    Returns:
        Numpy array of coordinates
    """
    from ..config import FILE_CONSTANTS
    required_cols = FILE_CONSTANTS['REQUIRED_CSV_COLUMNS']
    return df[required_cols].to_numpy()
