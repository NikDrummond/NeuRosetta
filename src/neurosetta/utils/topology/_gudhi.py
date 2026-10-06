"""Lazy GUDHI import helpers for optional TDA features."""

from __future__ import annotations

_GUDHI_MSG = (
    "GUDHI is required for persistence image and persistence distance operations. "
    "Install with: pip install 'neurosetta[tda]' or conda install -c conda-forge gudhi"
)


def require_gudhi():
    """Import and return the ``gudhi`` module, or raise a clear error."""
    try:
        import gudhi
    except ImportError as exc:
        raise ImportError(_GUDHI_MSG) from exc
    return gudhi
