"""Application settings and configuration management."""

import os
from typing import Dict, Any
from .constants import ENV_CONFIG, SCALE_CONSTANTS, UI_CONSTANTS


class AppSettings:
    """Manages application settings and environment configuration."""
    
    def __init__(self):
        self.scale_length_nm = SCALE_CONSTANTS['DEFAULT_SCALE_LENGTH_NM']
        self.units = 'nm'
        self.window_title = UI_CONSTANTS['WINDOW_TITLE']
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Configure environment variables to suppress warnings and ensure proper library loading."""
        for key, value in ENV_CONFIG.items():
            if key == 'LD_LIBRARY_PATH':
                # Append to existing LD_LIBRARY_PATH if it exists
                existing = os.environ.get(key, '')
                os.environ[key] = f"{value}:{existing}" if existing else value
            else:
                os.environ[key] = value
    
    def set_units(self, units: str) -> None:
        """Set the display units for measurements."""
        if units in SCALE_CONSTANTS['UNITS']:
            self.units = units
        else:
            raise ValueError(f"Unsupported units: {units}")
    
    def set_scale_length(self, length: int, units: str = None) -> None:
        """Set the scale bar length in the specified units."""
        if units is None:
            units = self.units
        
        if units == 'nm':
            self.scale_length_nm = length
        elif units == 'µm':
            self.scale_length_nm = length * SCALE_CONSTANTS['CONVERSION_FACTOR']
        else:
            raise ValueError(f"Unsupported units: {units}")
    
    def get_scale_length(self, units: str = None) -> int:
        """Get the scale bar length in the specified units."""
        if units is None:
            units = self.units
        
        if units == 'nm':
            return self.scale_length_nm
        elif units == 'µm':
            return self.scale_length_nm // SCALE_CONSTANTS['CONVERSION_FACTOR']
        else:
            raise ValueError(f"Unsupported units: {units}")
    
    def get_scale_display_text(self) -> str:
        """Get the formatted scale bar display text."""
        length = self.get_scale_length(self.units)
        return f"{length} {self.units}"
