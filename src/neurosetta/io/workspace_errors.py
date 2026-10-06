"""Domain errors for NeuRosetta Workspace (``.nrw``) I/O."""

from __future__ import annotations


class WorkspaceError(Exception):
    """Base class for Workspace-related failures."""


class WorkspaceFormatError(WorkspaceError, ValueError):
    """Raised when an archive is not a valid NeuRosetta Workspace."""


class WorkspaceVersionError(WorkspaceError, ValueError):
    """Raised when a Workspace schema version is unsupported."""


class WorkspaceIntegrityError(WorkspaceError, ValueError):
    """Raised when Workspace archive contents fail integrity checks."""


__all__ = [
    "WorkspaceError",
    "WorkspaceFormatError",
    "WorkspaceIntegrityError",
    "WorkspaceVersionError",
]
