"""Helpers for :class:`~neurosetta.core.forest._Forest` batch and filter operations."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable, Iterable, Sequence
from functools import lru_cache
from itertools import chain
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy import concatenate, ndarray, stack

if TYPE_CHECKING:
    from .tree import _Tree

T = TypeVar("T")


def _is_numeric_scalar(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating))


def _is_fixed_size_vector(x, sizes: tuple[int, ...] = (2, 3)) -> bool:
    """Return True for short numeric tuples/lists/1d arrays used as geometry."""
    if isinstance(x, np.ndarray):
        return x.ndim == 1 and x.shape[0] in sizes

    if isinstance(x, (tuple, list)):
        seq = x
    else:
        return False

    if len(seq) not in sizes:
        return False

    return all(_is_numeric_scalar(v) for v in seq)


def _is_per_tree_iterable(x, n: int) -> bool:
    """Return True when *x* should be split into one value per tree."""
    if isinstance(x, (str, bytes)):
        return False

    if _is_fixed_size_vector(x):
        return False

    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return False
        return x.ndim >= 1 and len(x) == n

    if not isinstance(x, Iterable):
        return False

    try:
        length = len(x)
    except TypeError:
        return False

    return length == n


def _coerce_id_sequence(ID: Hashable | Sequence[Hashable]) -> list[Hashable] | None:
    """Return a list of IDs when *ID* is a sequence; otherwise ``None``."""
    if isinstance(ID, (str, bytes)):
        return None
    if isinstance(ID, (list, tuple)):
        return list(ID)
    if isinstance(ID, np.ndarray):
        return ID.tolist()
    return None


def _meta_value_matches(metadata_value, condition) -> bool:
    """Return True when a metadata field matches a scalar or membership condition."""
    if isinstance(condition, (list, tuple, set)):
        return metadata_value in condition
    return metadata_value == condition


def _tree_metadata_matches(tree: _Tree, key: str, value) -> bool:
    """Return True when *tree* carries *key* and its value matches *value*."""
    return key in tree.metadata and _meta_value_matches(tree.metadata[key], value)


def _tree_matches_metadata_conditions(tree: _Tree, conditions: dict[str, Any]) -> bool:
    """Return True when *tree* satisfies all keyword metadata conditions."""
    return all(_tree_metadata_matches(tree, key, value) for key, value in conditions.items())


def _validate_filter_predicate(fn: Callable) -> None:
    """Ensure *fn* can be called as ``fn(tree)`` with one positional argument."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return
    positional_params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    required_params = [p for p in positional_params if p.default is inspect.Parameter.empty]
    if len(required_params) != 1:
        raise TypeError(
            "filter predicate must accept exactly one required argument (tree); "
            f"got {len(required_params)} required parameters"
        )


def _ensure_bool_filter_result(result) -> bool:
    """Validate and return a filter predicate result."""
    if not isinstance(result, bool):
        raise TypeError(f"filter predicate must return bool, got {type(result).__name__}")
    return result


@lru_cache(maxsize=256)
def _accepts_bind(fn: Callable) -> bool:
    """Cached check for whether fn has a 'bind' parameter."""
    try:
        return "bind" in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False


def _normalize_single_coord(arr: Any, sizes: tuple[int, ...] = (2, 3)) -> ndarray | None:
    """Return a 1D coordinate vector, or None if not a single-point result."""
    if not isinstance(arr, ndarray):
        return None

    if arr.ndim == 1 and arr.shape[0] in sizes:
        return arr

    if arr.ndim == 2:
        if arr.shape[0] == 1 and arr.shape[1] in sizes:
            return arr[0]
        if arr.shape[1] == 1 and arr.shape[0] in sizes:
            return arr[:, 0]

    return None


def _merge_results(results: list[Any], axis: int | None = None) -> Any:
    if axis is None:
        return results

    items = [r for r in results if r is not None]
    if not items:
        return None

    first = items[0]

    if isinstance(first, list):
        if axis != 0:
            raise ValueError("Lists can only be merged along axis=0.")
        return list(chain.from_iterable(items))

    if isinstance(first, ndarray):
        norms = [_normalize_single_coord(r) for r in items]
        if all(n is not None for n in norms):
            stacked = stack(norms, axis=0)
            if axis == 0:
                return stacked
            if axis in (1, -1):
                return stacked.T
            raise ValueError(
                f"Single coordinate vectors can only be merged along axis 0, 1, or -1; got {axis}."
            )

        return concatenate(items, axis=axis)

    raise TypeError(f"Don't know how to merge {type(first)!r}")
