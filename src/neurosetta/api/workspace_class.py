"""Persistent analysis Workspace container.

A :class:`Workspace` holds a primary :class:`~neurosetta.api.Forest` plus
session-level metadata, named Forest selections, and explicitly registered
analysis results. Persistence uses the ``.nrw`` archive format via
:func:`neurosetta.io.save_workspace` / :func:`neurosetta.io.load_workspace`.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, cast

from ..core import _Forest, _Tree
from .forest_class import Forest


def _is_tree_like(obj: Any) -> bool:
    return isinstance(obj, _Tree) or (
        hasattr(obj, "ID") and hasattr(obj, "graph") and not isinstance(obj, _Forest)
    )


def _selection_ids_from_value(
    selection: Any,
    *,
    forest: Forest,
) -> list[Hashable]:
    """Resolve a selection argument to ordered Forest member IDs."""
    if isinstance(selection, _Forest):
        ids = list(selection.ids())
    elif _is_tree_like(selection):
        ids = [cast(Any, selection).ID]
    elif isinstance(selection, (str, bytes)):
        # Single hashable ID (str is Sequence[str] — treat as one ID).
        ids = [selection]
    elif isinstance(selection, Sequence):
        ids = []
        for item in selection:
            if _is_tree_like(item):
                ids.append(cast(Any, item).ID)
            else:
                ids.append(item)
    else:
        # Single non-sequence hashable (e.g. int).
        ids = [selection]

    missing = [tree_id for tree_id in ids if tree_id not in forest]
    if missing:
        raise KeyError(
            f"selection contains Tree ID(s) not present in the Workspace Forest: {missing}"
        )
    return ids


class Workspace:
    """Persistent scientific state for an analysis involving a Forest.

    A Workspace is **not** a subclass of Forest. It composes one primary Forest
    with workspace-level metadata, named member selections, and named analysis
    results that are external to per-Tree ``.nr`` persistence.

    Parameters
    ----------
    forest : Forest
        Primary morphology collection owned by this workspace.
    name : str or None, optional
        Human-readable workspace title.
    description : str or None, optional
        Free-text description of the analysis.
    metadata : mapping, optional
        Workspace-level JSON-compatible metadata. This is **not** broadcast
        onto Forest members.

    Notes
    -----
    Persist with :func:`neurosetta.save_workspace` /
    :func:`neurosetta.load_workspace`. Do not pickle a Workspace.

    Tree-bound graph properties and metadata remain Tree state and are stored
    via existing ``.nr`` persistence inside the ``.nrw`` archive. Register
    DataFrames, arrays, and JSON-compatible objects with :meth:`add_result`
    only when they are external to individual Trees.

    Examples
    --------
    >>> from neurosetta.testing import make_synthetic_forest
    >>> import neurosetta as nr
    >>> forest = make_synthetic_forest(2, n=5, seed=0)
    >>> ws = nr.Workspace(forest, name="demo", metadata={"dataset": "synthetic"})
    >>> ws.add_selection("first", forest[:1])
    >>> ws.add_result("params", {"n_components": 3})
    """

    __slots__ = (
        "forest",
        "name",
        "description",
        "_metadata",
        "_selections",
        "_results",
    )

    def __init__(
        self,
        forest: Forest,
        *,
        name: str | None = None,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(forest, _Forest):
            raise TypeError(f"forest must be a Forest, got {type(forest).__name__}")
        self.forest = forest
        self.name = name
        self.description = description
        self._metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        self._selections: dict[str, list[Hashable]] = {}
        self._results: dict[str, Any] = {}

    # --- metadata ---------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Workspace-level metadata (mutable mapping; not Tree metadata)."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: Mapping[str, Any] | None) -> None:
        self._metadata = dict(value) if value is not None else {}

    # --- selections -------------------------------------------------------

    def add_selection(
        self,
        name: str,
        selection: _Forest | _Tree | Hashable | Sequence[Any],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a named subset of the primary Forest.

        Parameters
        ----------
        name : str
            Unique selection name.
        selection : Forest, Tree, hashable ID, or sequence of Trees/IDs
            Members must belong to :attr:`forest` (matched by ``ID``).
        overwrite : bool, optional
            Replace an existing selection of the same name. By default False.

        Raises
        ------
        TypeError
            If *name* is not a ``str``.
        ValueError
            If *name* already exists and ``overwrite`` is False.
        KeyError
            If any requested Tree ID is missing from :attr:`forest`.
        """
        if not isinstance(name, str):
            raise TypeError(f"selection name must be str, got {type(name).__name__}")
        if name in self._selections and not overwrite:
            raise ValueError(f"selection {name!r} already exists; pass overwrite=True to replace")
        self._selections[name] = _selection_ids_from_value(selection, forest=self.forest)

    def get_selection(self, name: str) -> Forest:
        """Return a Forest of members for a named selection (in selection order).

        Parameters
        ----------
        name : str
            Selection name.

        Returns
        -------
        Forest
            Sub-forest referencing the same Tree objects as :attr:`forest`.

        Raises
        ------
        KeyError
            If *name* is not a registered selection, or a stored ID is no
            longer present in :attr:`forest`.
        """
        try:
            ids = self._selections[name]
        except KeyError as exc:
            raise KeyError(f"no selection named {name!r}") from exc
        if not ids:
            return Forest([])  # type: ignore[no-untyped-call]
        return cast(Forest, self.forest.by_id(ids))

    def remove_selection(self, name: str) -> None:
        """Remove a named selection.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        try:
            del self._selections[name]
        except KeyError as exc:
            raise KeyError(f"no selection named {name!r}") from exc

    def list_selections(self) -> list[str]:
        """Return selection names in registration order."""
        return list(self._selections)

    def selection_ids(self, name: str) -> list[Hashable]:
        """Return a copy of the Tree IDs stored for *name*."""
        try:
            return list(self._selections[name])
        except KeyError as exc:
            raise KeyError(f"no selection named {name!r}") from exc

    # --- results ----------------------------------------------------------

    def add_result(self, name: str, value: Any, *, overwrite: bool = False) -> None:
        """Register a named analysis result.

        Parameters
        ----------
        name : str
            Unique result name.
        value
            JSON-compatible data, a NumPy ``ndarray``, or a pandas
            ``DataFrame``. Unsupported types fail at save time with a clear
            ``TypeError``.
        overwrite : bool, optional
            Replace an existing result of the same name. By default False.

        Raises
        ------
        TypeError
            If *name* is not a ``str``.
        ValueError
            If *name* already exists and ``overwrite`` is False.
        """
        if not isinstance(name, str):
            raise TypeError(f"result name must be str, got {type(name).__name__}")
        if name in self._results and not overwrite:
            raise ValueError(f"result {name!r} already exists; pass overwrite=True to replace")
        self._results[name] = value

    def get_result(self, name: str) -> Any:
        """Return a registered analysis result.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        try:
            return self._results[name]
        except KeyError as exc:
            raise KeyError(f"no result named {name!r}") from exc

    def remove_result(self, name: str) -> None:
        """Remove a named analysis result.

        Raises
        ------
        KeyError
            If *name* is not registered.
        """
        try:
            del self._results[name]
        except KeyError as exc:
            raise KeyError(f"no result named {name!r}") from exc

    def list_results(self) -> list[str]:
        """Return result names in registration order."""
        return list(self._results)

    def __repr__(self) -> str:
        name = self.name if self.name is not None else None
        return (
            f"Workspace(name={name!r}, trees={len(self.forest)}, "
            f"selections={len(self._selections)}, results={len(self._results)})"
        )


__all__ = ["Workspace"]
