"""File management operations for loading and saving neuron data."""

import os
import pathlib
import logging
from typing import List, Tuple, Optional, Any
import pandas as pd
import numpy as np

from ...io import import_swc, export_swc, save, load

from ..config import FILE_CONSTANTS
from ..utils.helpers import validate_csv_data, extract_coordinates_from_csv


class FileManager:
    """Handles file I/O operations for neuron data."""
    
    def __init__(self):
        self.supported_extensions = FILE_CONSTANTS['SUPPORTED_EXTENSIONS']
        self.file_filters = FILE_CONSTANTS['FILE_FILTERS']
    
    def get_file_filter_string(self) -> str:
        """Get the file filter string for file dialogs."""
        return self.file_filters['all']
    
    def get_save_filter_string(self) -> str:
        """Get the save file filter string for file dialogs."""
        return self.file_filters['neurosetta']
    
    def is_supported_file(self, filepath: str) -> bool:
        """Check if a file is supported based on its extension.
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if the file is supported, False otherwise
        """
        return pathlib.Path(filepath).suffix.lower() in self.supported_extensions
    
    def scan_folder_for_files(self, folder_path: str) -> List[str]:
        """Scan a folder for supported neuron files.
        
        Args:
            folder_path: Path to the folder to scan
            
        Returns:
            List of supported file paths
        """
        folder = pathlib.Path(folder_path)
        files = []
        
        for file_path in folder.glob('*'):
            if file_path.suffix.lower() in ['.nr', '.swc']:  # Only neuron files for folder scan
                files.append(str(file_path))
        
        return sorted(files)
    
    def load_file(self, filepath: str) -> Tuple[Any, np.ndarray, Optional[Any]]:
        """Load a neuron file and return the data.
        
        Args:
            filepath: Path to the file to load
            
        Returns:
            Tuple of (neuron_object, vertex_coordinates, point_coordinates)
            For CSV files, neuron_object will be None
            
        Raises:
            ValueError: If the file format is unsupported or invalid
            FileNotFoundError: If the file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        file_ext = pathlib.Path(filepath).suffix.lower()
        
        if file_ext == '.csv':
            return self._load_csv(filepath)
        elif file_ext in ['.nr', '.swc']:
            return self._load_neuron_file(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _load_csv(self, filepath: str) -> Tuple[None, np.ndarray, None]:
        """Load a CSV file containing point cloud data.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Tuple of (None, coordinates, None)
        """
        try:
            df = pd.read_csv(filepath)
            
            if not validate_csv_data(df):
                required_cols = FILE_CONSTANTS['REQUIRED_CSV_COLUMNS']
                raise ValueError(f"CSV file must contain columns: {required_cols}")
            
            coordinates = extract_coordinates_from_csv(df)
            logging.info(f"Loaded CSV file with {len(coordinates)} points")
            
            return None, coordinates, None
            
        except Exception as e:
            logging.error(f"Failed to load CSV file {filepath}: {e}")
            raise
    
    def _load_neuron_file(self, filepath: str) -> Tuple[Any, np.ndarray, np.ndarray]:
        """Load a neuron file (.nr or .swc).
        
        Args:
            filepath: Path to the neuron file
            
        Returns:
            Tuple of (neuron_object, vertex_coordinates, point_coordinates)
            
        Raises:
            ValueError: If the file format is not supported
        """
        try:
            file_ext = pathlib.Path(filepath).suffix.lower()
            
            # Use appropriate loader based on file extension
            if file_ext == '.nr':
                neuron = load(filepath)
            elif file_ext == '.swc':
                neuron = import_swc(filepath)
            else:
                raise ValueError(f"Unsupported neuron file format: {file_ext}")
                
            vertex_coords = neuron.get_node_coordinates()
            
            # Get point coordinates for interaction
            point_coords = vertex_coords[neuron.core_indices()]
            
            logging.info(f"Loaded {file_ext.upper().lstrip('.')} file with {len(vertex_coords)} vertices")
            
            return neuron, vertex_coords, point_coords
            
        except Exception as e:
            logging.error(f"Failed to load neuron file {filepath}: {e}")
            raise
    
    def save_neuron(self, neuron: Any, filepath: str) -> None:
        """Save a neuron object to file.
        
        Args:
            neuron: Neurosetta neuron object to save
            filepath: Path where to save the file
            
        Raises:
            ValueError: If the neuron object is invalid
        """
        if neuron is None:
            raise ValueError("Cannot save: neuron object is None")
        
        try:
            file_ext = pathlib.Path(filepath).suffix.lower()
            if file_ext == '.nr':
                save(neuron, filepath)
            elif file_ext == '.swc':
                export_swc(neuron, filepath)
            else:
                raise ValueError(f"Unsupported save format: {file_ext}")
            logging.info(f"Saved neuron to: {filepath}")
            
        except Exception as e:
            logging.error(f"Failed to save neuron to {filepath}: {e}")
            raise
    
    def save_neuron_to_directory(self, neuron: Any, directory: str) -> None:
        """Save a neuron object to a directory using Neurosetta's directory save.
        
        Args:
            neuron: Neurosetta neuron object to save
            directory: Directory path where to save
            
        Raises:
            ValueError: If the neuron object is invalid
        """
        if neuron is None:
            raise ValueError("Cannot save: neuron object is None")
        
        # Ensure directory path ends with separator
        if not directory.endswith('/'):
            directory += '/'
        
        try:
            save(neuron, directory)
            logging.info(f"Saved neuron to directory: {directory}")
            
        except Exception as e:
            logging.error(f"Failed to save neuron to directory {directory}: {e}")
            raise
    
    def get_directory_from_path(self, filepath: str) -> str:
        """Get the directory path from a file path.
        
        Args:
            filepath: Full path to a file
            
        Returns:
            Directory path
        """
        return os.path.dirname(filepath)
