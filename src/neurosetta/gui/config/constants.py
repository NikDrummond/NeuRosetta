"""Constants and configuration values for the Neurosetta GUI application."""

from typing import Dict, Any

# UI Constants
UI_CONSTANTS = {
    'WINDOW_TITLE': 'Neurosetta Viewer',
    'SIDE_PANEL_WIDTH': 180,
    'BUTTON_MARGINS': (5, 5, 5, 5),
    'OVERLAY_MARGIN': 10,
    'PICKER_TOLERANCE': 0.005,
    'HOVER_DISTANCE_THRESHOLD': 20,
}

# Rendering Constants
RENDERING_CONSTANTS = {
    'BACKGROUND_COLOR': 'white',
    'DEFAULT_NEURON_COLOR': 'black',
    'SOMA_COLOR': 'red',
    'SOMA_RADIUS': 15,
    'POINT_RADIUS': 8,
    'SELECTED_POINT_RADIUS': 10,
    'HOVER_POINT_RADIUS': 15,
    'HOVER_POINT_COLOR': 'yellow',
    'HOVER_POINT_ALPHA': 0.6,
    'SELECTED_IN_COLOR': 'b',  # blue
    'SELECTED_OUT_COLOR': 'r',  # red
    'POINT_CLOUD_COLOR': 'cyan',
    'LINE_WIDTH': 3,
    'HOVER_DISTANCE_THRESHOLD': 5.0,
}

# File Constants
FILE_CONSTANTS = {
    'SUPPORTED_EXTENSIONS': ['.nr', '.swc', '.csv'],
    'FILE_FILTERS': {
        'neurosetta': 'Neurosetta (*.nr)',
        'swc': 'SWC (*.swc)',
        'csv': 'CSV (*.csv)',
        'all': 'Neurosetta (*.nr);;SWC (*.swc);;CSV (*.csv)'
    },
    'REQUIRED_CSV_COLUMNS': ['x', 'y', 'z'],
}

# Scale Bar Constants
SCALE_CONSTANTS = {
    'DEFAULT_SCALE_LENGTH_NM': 5000,
    'UNITS': {
        'nm': 'Nanometers',
        'µm': 'Micrometers'
    },
    'CONVERSION_FACTOR': 1000,  # nm to µm
}

# Environment Configuration
ENV_CONFIG = {
    'QT_DEBUG_PLUGINS': '0',
    'GI_TYPELIB_PATH': '/usr/lib/x86_64-linux-gnu/girepository-1.0',
    'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu',
    'GV_PLUGIN_PATH': '/usr/lib/graphviz',
}

# Logging Configuration
LOGGING_CONFIG = {
    'FORMAT': '%(asctime)s %(levelname)-5s %(message)s',
    'DATE_FORMAT': '%H:%M:%S',
    'LEVEL': 'DEBUG',
}
