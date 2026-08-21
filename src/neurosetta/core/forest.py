"""Container class for multiple trees."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Self, TypeVar

import numpy as np
from tqdm import tqdm

from ..config import resolve_max_workers, resolve_parallel, resolve_show_progress
from .forest_helpers import (
    _accepts_bind,
    _coerce_id_sequence,
    _ensure_bool_filter_result,
    _is_per_tree_iterable,
    _merge_results,
    _tree_matches_metadata_conditions,
    _tree_metadata_matches,
    _validate_filter_predicate,
)

if TYPE_CHECKING:
    from .tree import _Tree

T = TypeVar("T")


class _Forest(Sequence):
    """Ordered, mutable container of Tree (or other Stone) objects."""

    __slots__ = ("_trees", "_id_index")

    def __init__(self, trees: Iterable[_Tree] = ()) -> None:
        self._trees: list[_Tree] = []
        self._id_index: dict[Hashable, int] = {}
        self.extend(trees)

    def __repr__(self) -> str:
        return f"Forest(n={len(self)}, ids={self.ids()})"

    def __len__(self) -> int:
        return len(self._trees)

    def __iter__(self) -> Iterator[_Tree]:
        return iter(self._trees)

    def __contains__(self, item: Hashable | _Tree) -> bool:
        """Return True when *item* is a tree ID present in the forest.

        Tree objects are matched by ``ID``, not object identity.
        """
        tree_id = item.ID if hasattr(item, "ID") else item
        return tree_id in self._id_index

    def copy(self) -> Self:
        """Return a shallow copy of the forest container.

        Tree objects are shared with the original forest.
        """
        return type(self)(self._trees)

    def _trees_at_indices(self, indices: Sequence[int]) -> Self:
        return type(self)(self._trees[i] for i in indices)

    def _trees_at_mask(self, mask: Sequence[bool]) -> Self:
        if len(mask) != len(self._trees):
            raise IndexError(
                f"boolean index did not match forest length; "
                f"mask length {len(mask)} != forest length {len(self._trees)}"
            )
        return type(self)(tree for tree, keep in zip(self._trees, mask, strict=True) if keep)

    def __getitem__(self, idx):
        """Index trees by position, slice, integer list, or boolean mask.

        Boolean lists/tuples must contain only ``bool`` values. Mixed
        bool/int sequences are treated as integer indices.
        """
        if isinstance(idx, slice):
            return type(self)(self._trees[idx])
        if isinstance(idx, (list, tuple)):
            if idx and all(isinstance(i, bool) for i in idx):
                return self._trees_at_mask(idx)
            return self._trees_at_indices(idx)
        if isinstance(idx, np.ndarray):
            if idx.dtype.kind == "b":
                return self._trees_at_mask(idx.tolist())
            if idx.dtype.kind in "iu":
                return self._trees_at_indices(idx.tolist())
        return self._trees[idx]

    def __add__(self, other: Self) -> Self:
        result = self.copy()
        result.extend(other)
        return result

    def _check_id(self, tree: _Tree) -> None:
        if tree.ID in self._id_index:
            raise ValueError(f"Duplicate Tree ID: {tree.ID}")

    def _rebuild_index(self, start: int = 0) -> None:
        for i in range(start, len(self._trees)):
            self._id_index[self._trees[i].ID] = i

    def append(self, tree: _Tree) -> None:
        """Append a tree to the end of the forest.

        Parameters
        ----------
        tree
            Tree to add. Its ``ID`` must not already be present.

        Raises
        ------
        ValueError
            If a tree with the same ``ID`` is already in the forest.

        Notes
        -----
        Do not mutate ``tree.ID`` while the tree remains in a forest; the
        internal ID index will become stale.
        """
        self._check_id(tree)
        self._id_index[tree.ID] = len(self._trees)
        self._trees.append(tree)

    def extend(self, trees: Iterable[_Tree]) -> None:
        """Append multiple trees from an iterable.

        The iterable is materialised once before any trees are added, so
        generators are safe to pass.

        Parameters
        ----------
        trees : iterable of Tree
            Trees to append. Each ``ID`` must not already be present in the
            forest or duplicated earlier in ``trees``.

        Raises
        ------
        ValueError
            If any tree ``ID`` is already in the forest.
        """
        trees = list(trees)
        for i, tree in enumerate(trees):
            if tree.ID in self._id_index:
                raise ValueError(
                    f"Duplicate Tree ID {tree.ID} at position {i} in the supplied iterable "
                    f"(already at index {self._id_index[tree.ID]} in this forest)"
                )
        for tree in trees:
            self._id_index[tree.ID] = len(self._trees)
            self._trees.append(tree)

    def insert(self, index: int, tree: _Tree) -> None:
        """Insert a tree at ``index``, shifting later trees right.

        Negative indices count from the end, matching list semantics.

        Parameters
        ----------
        index : int
            Insertion position.
        tree
            Tree to insert. Its ``ID`` must not already be present.

        Raises
        ------
        ValueError
            If a tree with the same ``ID`` is already in the forest.
        """
        if index < 0:
            index = max(0, len(self._trees) + index)
        self._check_id(tree)
        self._trees.insert(index, tree)
        self._rebuild_index(index)

    def remove(self, tree: _Tree) -> None:
        """Remove the tree whose ``ID`` matches ``tree.ID``.

        Parameters
        ----------
        tree
            Tree whose ``ID`` should be removed from the forest.

        Raises
        ------
        ValueError
            If no tree with ``tree.ID`` is in the forest.
        """
        if tree.ID not in self._id_index:
            raise ValueError(f"Tree with ID {tree.ID} is not in this forest")
        self.pop(self._id_index[tree.ID])

    def remove_id(self, ID: Hashable | Sequence[Hashable]) -> None:
        """Remove the tree or trees whose ``ID`` matches ``ID``.

        Parameters
        ----------
        ID : hashable or sequence of hashable
            A single tree ``ID``, or a list/tuple/array of IDs to remove.

        Raises
        ------
        KeyError
            If any requested ID is not present in the forest.
        """
        ids = _coerce_id_sequence(ID)
        if ids is None:
            ids = [ID]

        for tree_id in ids:
            if tree_id not in self._id_index:
                raise KeyError(f"No tree with ID {tree_id}")

        for index in sorted({self._id_index[tree_id] for tree_id in ids}, reverse=True):
            self.pop(index)

    def pop(self, index: int = -1) -> _Tree:
        """Remove and return the tree at ``index``.

        Parameters
        ----------
        index : int, optional
            Position of the tree to remove. By default ``-1`` (last tree).

        Returns
        -------
        Tree
            The removed tree.

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        if index < 0:
            index = len(self._trees) + index
        tree = self._trees.pop(index)
        del self._id_index[tree.ID]
        self._rebuild_index(index)
        return tree

    def clear(self) -> None:
        """Remove all trees and reset the internal ID index."""
        self._trees.clear()
        self._id_index.clear()

    def __iadd__(self, other: Self) -> Self:
        self.extend(other)
        return self

    def _tree_at_id(self, ID: Hashable) -> _Tree:
        if ID not in self._id_index:
            raise KeyError(f"No tree with ID {ID}")
        return self._trees[self._id_index[ID]]

    def by_id(self, ID: Hashable | Sequence[Hashable]) -> _Tree | Self:
        """Return one tree or a sub-forest selected by ID.

        Parameters
        ----------
        ID : hashable or sequence of hashable
            A single tree ``ID``, or a list/tuple/array of IDs. For a
            sequence, returns a new forest containing the matching trees
            in request order (duplicates are allowed).

        Returns
        -------
        Tree or Forest
            A single tree when ``ID`` is scalar; otherwise a new forest.

        Raises
        ------
        KeyError
            If any requested ID is not present in the forest.
        """
        ids = _coerce_id_sequence(ID)
        if ids is None:
            return self._tree_at_id(ID)
        return type(self)(self._tree_at_id(i) for i in ids)

    def ids(self) -> list[Hashable]:
        """Return tree IDs in forest order."""
        return [tree.ID for tree in self._trees]

    def list_properties(self, level: str = "all") -> list:
        """List properties for each member (sequential)."""
        return [t.list_properties(level) for t in self._trees]

    # --- user metadata (per-tree, broadcast scalars / length-n iterables) ---

    def get_meta(self, key: str, default=None) -> list[Any]:
        """Return metadata values for ``key`` from each tree."""
        return [t.get_meta(key, default) for t in self._trees]

    def has_meta(self, key: str) -> list[bool]:
        """Return whether each tree defines ``key`` in metadata."""
        return [t.has_meta(key) for t in self._trees]

    def set_meta(self, key: str, value) -> None:
        """Set a user metadata entry on every tree.

        ``value`` is broadcast to all trees unless it is an iterable with
        length ``len(self)`` (same rules as :meth:`apply`).
        """
        n = len(self)
        if n == 0:
            return
        values = list(value) if _is_per_tree_iterable(value, n) else [value] * n
        for tree, item in zip(self._trees, values, strict=True):
            tree.set_meta(key, item)

    def del_meta(self, key: str) -> None:
        """Delete a user metadata entry from each tree that defines it."""
        for tree in self._trees:
            if tree.has_meta(key):
                tree.del_meta(key)

    def list_meta(self, *, include_protected: bool = False) -> list[str]:
        """Sorted union of metadata keys present in any tree."""
        keys: set[str] = set()
        for tree in self._trees:
            keys.update(tree.list_meta(include_protected=include_protected))
        return sorted(keys)

    def meta_summary(self, *, include_protected: bool = False) -> dict[str, int]:
        """Count trees that carry each metadata key."""
        counts: dict[str, int] = {}
        for tree in self._trees:
            for key, n in tree.meta_summary(include_protected=include_protected).items():
                counts[key] = counts.get(key, 0) + n
        return dict(sorted(counts.items()))

    def meta_indices(self, key: str, value) -> list[int]:
        """Return indices of trees whose metadata value for ``key`` matches.

        ``value`` may be a scalar (equality) or a list/tuple/set (membership).
        Trees missing ``key`` never match.
        """
        return [i for i, tree in enumerate(self._trees) if _tree_metadata_matches(tree, key, value)]

    def map(self, fn: Callable[[_Tree], Any]) -> list[Any]:
        """Sequential map over trees (convenience alias for apply with parallel=False)."""
        return self.apply(fn, parallel=False)

    # --- filtering ---

    def filter(
        self,
        predicate: Callable[[_Tree], bool] | None = None,
        **conditions,
    ) -> Self:
        """Return a new forest containing trees matching metadata conditions.

        Two filtering modes are supported (mutually exclusive):

        1. Keyword equality — all conditions must match (AND logic).
        2. Callable predicate — custom logic over each :class:`~neurosetta.api.tree_class.Tree`.

        Missing metadata keys are treated as non-matching; no ``KeyError`` is
        raised.

        Parameters
        ----------
        predicate : Callable[[Tree], bool], optional
            User-defined filter function called as ``predicate(tree)`` for each
            tree.             Must accept exactly one required argument (the tree) and return
            ``True`` to keep the tree or ``False`` to drop it. Additional
            parameters may use defaults. Pass as the sole positional argument,
            e.g. ``forest.filter(my_filter_fn)``.
        **conditions
            Metadata key/value pairs for equality filtering. Values may be
            scalars (exact match) or list/tuple/set (membership test, OR
            within that key). Ignored when *predicate* is provided.

        Returns
        -------
        Forest
            New forest containing only matching trees.

        Raises
        ------
        ValueError
            If both *predicate* and keyword conditions are supplied, or if
            neither is supplied.
        TypeError
            If *predicate* does not accept exactly one argument or does not
            return a ``bool``.

        Notes
        -----
        Keyword filtering examples::

            forest.filter(Neuron_type='T4', Neuron_subtype='a')
            forest.filter(Neuron_type=['T4', 'T5'])

        Custom filter function examples::

            def t4_or_long(tree):
                return tree.get_meta('Neuron_type') == 'T4' or tree.get_total_cable_length() > 500

            forest.filter(t4_or_long)
            forest.filter(lambda tree: tree.count_nodes() > 100)
        """
        if predicate is not None and conditions:
            raise ValueError("Provide either a predicate or keyword conditions, not both.")

        if predicate is not None:
            _validate_filter_predicate(predicate)
            return type(self)(t for t in self._trees if _ensure_bool_filter_result(predicate(t)))

        if not conditions:
            raise ValueError("Provide a predicate or at least one keyword condition.")

        return type(self)(
            t for t in self._trees if _tree_matches_metadata_conditions(t, conditions)
        )

    def apply(
        self,
        fn: Callable[..., T],
        *args,
        parallel: bool | None = None,
        max_workers: int | None = None,
        show_progress: bool | None = None,
        bind: bool = False,
        merge_axis: int | None = None,
        **kwargs,
    ) -> list[T] | None:
        """Apply a function to every tree in the forest.

        Positional and keyword arguments are broadcast to all trees unless
        given as length-*n* iterables (one value per tree). Short numeric
        vectors such as ``(1.0, 0.0, 0.0)`` are always broadcast as a
        single geometric value, even when ``len(self) == 3``. When *fn*
        accepts a ``bind`` parameter and ``bind=True``, results are written
        in place and ``None`` is returned.

        Parameters
        ----------
        fn : Callable[..., T]
            Function called as ``fn(tree, *args, **kwargs)`` for each tree.
        *args
            Positional arguments broadcast to all trees, or iterables of length
            ``len(self)`` supplying per-tree values.
        parallel : bool or None, optional
            Run calls in a thread pool. When None, use the global forest-scoped
            setting (default True). Thread parallelism helps most for I/O-bound
            or native-code workloads; it may not speed up pure Python graph ops.
        max_workers : int or None, optional
            Maximum worker threads when *parallel* is True. When None, use the
            global setting (executor default when unset).
        show_progress : bool or None, optional
            Show a tqdm progress bar. When None, use the global setting
            (default False).
        bind : bool, optional
            Forward ``bind=True`` to *fn* when supported, suppressing the
            return value. By default False.
        merge_axis : int | None, optional
            Axis along which to concatenate ndarray results, or ``0`` to
            flatten list results. When None, return a plain list of per-tree
            results. Single coordinate vectors (one point per tree) are
            stacked to ``(N, d)`` for ``axis=0`` or ``(d, N)`` for
            ``axis=1`` / ``axis=-1`` instead of being flattened. By default
            None.
        **kwargs
            Keyword arguments broadcast to all trees, or iterables of length
            ``len(self)`` supplying per-tree values.

        Returns
        -------
        list | ndarray | None
            Per-tree results as a list when *merge_axis* is None; concatenated
            ndarray or flattened list when *merge_axis* is set; ``None`` when
            all results are ``None`` or when ``bind=True`` and *fn* accepts
            ``bind``. An empty forest returns ``[]``.

        Raises
        ------
        ValueError
            If a per-tree iterable argument does not have length ``len(self)``.
        TypeError
            If *merge_axis* is set but result types cannot be merged.

        Warns
        -----
        UserWarning
            If ``bind=True`` but *fn* does not accept a ``bind`` parameter.
        """
        n = len(self)
        if n == 0:
            return []

        norm_args: list[list] = []
        for a in args:
            if _is_per_tree_iterable(a, n):
                norm_args.append(list(a))
            else:
                norm_args.append([a] * n)

        norm_kwargs: dict[str, list] = {}
        for k, v in kwargs.items():
            if _is_per_tree_iterable(v, n):
                norm_kwargs[k] = list(v)
            else:
                norm_kwargs[k] = [v] * n

        accepts_bind = _accepts_bind(fn)
        if bind and not accepts_bind:
            warnings.warn(
                f"{getattr(fn, '__name__', repr(fn))!r} does not accept 'bind'; "
                "results will be returned normally.",
                stacklevel=2,
            )
            bind = False

        def call(i: int):
            call_kwargs = {k: v[i] for k, v in norm_kwargs.items()}
            if accepts_bind:
                call_kwargs["bind"] = bind
            return fn(self._trees[i], *(arg[i] for arg in norm_args), **call_kwargs)

        indices = range(n)

        parallel = resolve_parallel(explicit=parallel, scope="forest")
        max_workers = resolve_max_workers(explicit=max_workers)
        show_progress = resolve_show_progress(explicit=show_progress)

        if not parallel:
            it = tqdm(indices, total=n, desc="Processing trees") if show_progress else indices
            results = [call(i) for i in it]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(call, i) for i in indices]
                it = tqdm(futures, total=n, desc="Processing trees") if show_progress else futures
                results = [fut.result() for fut in it]

        if bind and accepts_bind:
            return None
        if all(r is None for r in results):
            return None
        return _merge_results(results, merge_axis)

    def build_3d(
        self,
        force_refresh: bool = False,
        parallel: bool | None = None,
        show_root: bool = False,
        random_c: bool = True,
        **kwargs,
    ):
        """Build and cache TreePlot3D for all trees."""

        self.apply(
            lambda t, **kw: t.make_plot3d(**kw),
            parallel=parallel,
            cache=True,
            random_c=random_c,
            show_root=show_root,
            force_refresh=force_refresh,
            **kwargs,
        )
