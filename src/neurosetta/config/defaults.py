"""Default values for global NeuRosetta settings."""

DEFAULT_UNITS = "dimensionless"

OMP_WAIT_POLICY = "passive"

# Fallback when no explicit kwarg, context override, or configured scope value applies.
PARALLEL_SCOPE_DEFAULTS = {
    "io": False,
    "forest": True,
    "default": False,
}

DEFAULT_VEDO_BACKEND = "vtk"
DEFAULT_VEDO_USE_PARALLEL_PROJECTION = True
DEFAULT_VEDO_WINDOW_SIZE = (1200, 800)
DEFAULT_VEDO_BG = "white"

ENV_PREFIX = "NEUROSETTA_"
