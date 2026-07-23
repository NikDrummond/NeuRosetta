"""Helper functions for neuron data manipulation and visualization."""

import numpy as np
import vedo as vd
from typing import Tuple, Any
from ..config import FILE_CONSTANTS


def n_pnt_coords(neuron: Any) -> np.ndarray:
    """Get point coordinates for a neuron's core indices.

    Parameters
    ----------
    neuron : Any
        Neurosetta neuron object with methods `get_node_coordinates` and
        `core_indices`.

    Returns
    -------
    np.ndarray
        Array of point coordinates for core vertices with shape (N, 3).
    """
    return neuron.get_node_coordinates()[neuron.core_indices()]


def make_pnts(coords: np.ndarray, mask: np.ndarray) -> Tuple[vd.Points, vd.Points]:
    """Create vedo Points objects for selected and unselected points.

    Parameters
    ----------
    coords : np.ndarray
        Array of point coordinates with shape (N, 3).
    mask : np.ndarray
        Boolean mask indicating selected points with shape (N,).

    Returns
    -------
    Tuple[vd.Points, vd.Points]
        Tuple of (selected_points, unselected_points) as vedo.Points objects.
        Selected points are blue with radius 8, unselected are red with radius 8.
    """
    selected_points = vd.Points(coords[mask], c="b", r=8)
    unselected_points = vd.Points(coords[~mask], c="r", r=8)
    return selected_points, unselected_points


def get_mask_node_ind(neuron: Any, mask: np.ndarray) -> np.ndarray:
    """Get node indices corresponding to masked points.

    Parameters
    ----------
    neuron : Any
        Neurosetta neuron object with method `core_indices`.
    mask : np.ndarray
        Boolean mask for points with shape (N,).

    Returns
    -------
    np.ndarray
        Array of node indices for masked points.
    """
    return neuron.core_indices()[mask]


def validate_csv_data(df) -> bool:
    """Validate that CSV data contains required columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.

    Returns
    -------
    bool
        True if all required columns are present, False otherwise.
    """
    required_cols = FILE_CONSTANTS["REQUIRED_CSV_COLUMNS"]
    return all(col in df.columns for col in required_cols)


def extract_coordinates_from_csv(df) -> np.ndarray:
    """Extract coordinate array from CSV DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with x, y, z columns.

    Returns
    -------
    np.ndarray
        Numpy array of coordinates with shape (N, 3).
    """
    required_cols = FILE_CONSTANTS["REQUIRED_CSV_COLUMNS"]
    return df[required_cols].to_numpy()