"""Container class for multiple trees"""

from typing import Iterable, Iterator, Callable, TypeVar, Any, Hashable
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import inspect
from functools import lru_cache
from itertools import chain
import warnings
from numpy import concatenate, ndarray

from tqdm import tqdm

T = TypeVar("T")


def _is_iterable(x) -> bool:
    """
    Returns True for iterables that should be broadcast per-tree.
    Excludes strings and bytes (scalar-like), and numpy arrays with ndim < 1
    or scalar arrays (ndim == 0), which behave like single values.
    """
    if isinstance(x, (str, bytes)):
        return False
    # guard against numpy scalar arrays (ndim=0) which are iterable but represent a single value
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.ndim >= 1
    except ImportError:
        pass
    return isinstance(x, Iterable)


@lru_cache(maxsize=256)
def _accepts_bind(fn: Callable) -> bool:
    """Cached check for whether fn has a 'bind' parameter."""
    try:
        return "bind" in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        # some built-ins don't support inspect.signature
        return False

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
        return concatenate(items, axis=axis)

    raise TypeError(f"Don't know how to merge {type(first)!r}")

class _Forest(Sequence):
    """Ordered, mutable container of Tree (or other Stone) objects."""

    __slots__ = ("_trees", "_id_index")

    def __init__(self, trees: Iterable[Any] = ()) -> None:
        self._trees: list[Any] = []
        self._id_index: dict[Hashable, int] = {}
        self.extend(trees)

    def __repr__(self) -> str:
        return f"Forest(n={len(self)}, ids={self.ids()})"

    def __len__(self) -> int:
        return len(self._trees)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._trees)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return type(self)(self._trees[idx])
        return self._trees[idx]

    def __add__(self, other: "_Forest") -> "_Forest":
        result = type(self)(self)
        result.extend(other)
        return result

    def _check_id(self, tree: Any) -> None:
        if tree.ID in self._id_index:
            raise ValueError(f"Duplicate Tree ID: {tree.ID}")

    def _rebuild_index(self, start: int = 0) -> None:
        for i in range(start, len(self._trees)):
            self._id_index[self._trees[i].ID] = i

    def append(self, tree: Any) -> None:
        self._check_id(tree)
        self._id_index[tree.ID] = len(self._trees)
        self._trees.append(tree)

    def extend(self, trees: Iterable[Any]) -> None:
        trees = list(trees)  # materialise once (fixes generator bug too)
        for i, tree in enumerate(trees):
            if tree.ID in self._id_index:
                raise ValueError(
                    f"Duplicate Tree ID {tree.ID} at position {i} in the supplied iterable "
                    f"(already at index {self._id_index[tree.ID]} in this forest)"
                )
        for tree in trees:
            self._id_index[tree.ID] = len(self._trees)
            self._trees.append(tree)

    def insert(self, index: int, tree: Any) -> None:
        if index < 0:
            index = max(0, len(self._trees) + index)
        self._check_id(tree)
        self._trees.insert(index, tree)
        self._rebuild_index(index)

    def remove(self, tree: Any) -> None:
        if tree.ID not in self._id_index:
            raise ValueError(f"Tree with ID {tree.ID} is not in this forest")
        self.pop(self._id_index[tree.ID])

    def remove_id(self, ID: Hashable) -> None:
        if ID not in self._id_index:
            raise KeyError(f"No tree with ID {ID}")
        self.pop(self._id_index[ID])

    def pop(self, index: int = -1) -> Any:
        if index < 0:
            index = len(self._trees) + index
        tree = self._trees.pop(index)
        del self._id_index[tree.ID]
        self._rebuild_index(index)
        return tree

    def clear(self) -> None:
        self._trees.clear()
        self._id_index.clear()

    def __iadd__(self, other: "_Forest"):
        self.extend(other)
        return self

    def by_id(self, ID: Hashable) -> Any:
        if ID not in self._id_index:
            raise KeyError(f"No tree with ID {ID}")
        return self._trees[self._id_index[ID]]

    def ids(self) -> list[Hashable]:
        return [tree.ID for tree in self._trees]

    def list_properties(self, level: str = "all") -> list:
        """List properties for each member (sequential)."""
        return [t.list_properties(level) for t in self._trees]

    def map(self, fn: Callable[[Any], Any]) -> list[Any]:
        """Sequential map over trees (convenience alias for apply with parallel=False)."""
        return self.apply(fn, parallel=False)

    ### filtering
    def filter(
        self,
        predicate: Callable[[dict], bool] | None = None,
        **conditions,
    ) -> "_Forest":
        """Return a new forest containing trees matching metadata conditions.

        Two filtering modes are supported (mutually exclusive):

        1. Keyword equality — all conditions must match (AND logic).
        2. Callable predicate — custom logic over each tree's metadata dict.

        Missing metadata keys are treated as non-matching; no ``KeyError`` is
        raised.

        Parameters
        ----------
        predicate : Callable[[dict], bool], optional
            Callable receiving a tree metadata dict and returning whether to
            keep the tree. By default None.
        **conditions
            Metadata key/value pairs for equality filtering. Values may be
            scalars (exact match) or list/tuple/set (membership test, OR
            within that key). Ignored when *predicate* is provided.

        Returns
        -------
        _Forest
            New forest containing only matching trees.

        Raises
        ------
        ValueError
            If both *predicate* and keyword conditions are supplied.

        Notes
        -----
        Keyword filtering examples::

            forest.filter(Neuron_type='T4', Neuron_subtype='a')
            forest.filter(Neuron_type=['T4', 'T5'])

        Predicate filtering example::

            forest.filter(lambda m: m.get('Neuron_type') == 'T4' or m.get('score', 0) > 5)
        """
        if predicate is not None and conditions:
            raise ValueError("Provide either a predicate or keyword conditions, not both.")

        if predicate is not None:
            return type(self)(t for t in self._trees if predicate(t.metadata))

        def _matches(metadata_value, condition) -> bool:
            if isinstance(condition, (list, tuple, set)):
                return metadata_value in condition
            return metadata_value == condition

        return type(self)(
            t
            for t in self._trees
            if all(_matches(t.metadata.get(k), v) for k, v in conditions.items())
        )

    def apply(
        self,
        fn: Callable[..., T],
        *args,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = False,
        bind: bool = False,
        merge_axis: int | None = None,
        **kwargs,
    ) -> list[T] | None:
        """Apply a function to every tree in the forest.

        Positional and keyword arguments are broadcast to all trees unless
        given as length-*n* iterables (one value per tree). When *fn* accepts
        a ``bind`` parameter and ``bind=True``, results are written in place
        and ``None`` is returned.

        Parameters
        ----------
        fn : Callable[..., T]
            Function called as ``fn(tree, *args, **kwargs)`` for each tree.
        *args
            Positional arguments broadcast to all trees, or iterables of length
            ``len(self)`` supplying per-tree values.
        parallel : bool, optional
            Run calls in a thread pool. By default True.
        max_workers : int | None, optional
            Maximum worker threads when *parallel* is True. By default None
            (executor default).
        show_progress : bool, optional
            Show a tqdm progress bar. By default False.
        bind : bool, optional
            Forward ``bind=True`` to *fn* when supported, suppressing the
            return value. By default False.
        merge_axis : int | None, optional
            Axis along which to concatenate ndarray results, or ``0`` to
            flatten list results. When None, return a plain list of per-tree
            results. By default None.
        **kwargs
            Keyword arguments broadcast to all trees, or iterables of length
            ``len(self)`` supplying per-tree values.

        Returns
        -------
        list | ndarray | None
            Per-tree results as a list when *merge_axis* is None; concatenated
            ndarray or flattened list when *merge_axis* is set; ``None`` when
            all results are ``None`` or when ``bind=True`` and *fn* accepts
            ``bind``.

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

        # --- normalize positional args
        norm_args: list[list] = []
        for a in args:
            if _is_iterable(a):
                if len(a) != n:
                    raise ValueError("Iterable argument length mismatch")
                norm_args.append(list(a))
            else:
                norm_args.append([a] * n)

        # --- normalize keyword args
        norm_kwargs: dict[str, list] = {}
        for k, v in kwargs.items():
            if _is_iterable(v):
                if len(v) != n:
                    raise ValueError(f"Iterable kwarg '{k}' length mismatch")
                norm_kwargs[k] = list(v)
            else:
                norm_kwargs[k] = [v] * n

        # --- check if fn accepts 'bind'
        accepts_bind = _accepts_bind(fn)
        if bind and not accepts_bind:
            warnings.warn(
                f"{getattr(fn, '__name__', repr(fn))!r} does not accept 'bind'; "
                "results will be returned normally.",
                stacklevel=2,
            )
            bind = False

        # --- per-tree call
        def call(i: int):
            call_kwargs = {k: v[i] for k, v in norm_kwargs.items()}
            if accepts_bind:
                call_kwargs["bind"] = bind
            return fn(self._trees[i], *(arg[i] for arg in norm_args), **call_kwargs)

        indices = range(n)

        # --- sequential
        if not parallel:
            it = tqdm(indices, total=n, desc="Processing trees") if show_progress else indices
            results = [call(i) for i in it]

        # --- parallel
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(call, i) for i in indices]
                it = tqdm(futures, total=n, desc="Processing trees") if show_progress else futures
                results = [fut.result() for fut in it]

        # --- return
        if bind and accepts_bind:
            return None
        if all(r is None for r in results):
            return None
        return _merge_results(results, merge_axis)

    def build_3d(
        self,
        force_refresh: bool = False,
        parallel: bool = False,
        show_root: bool = False,
        random_c: bool = True,
        **kwargs,
    ):
        """Build and cache TreePlot3D for all trees."""

        if force_refresh:
            for tree in self:
                tree.plot3d = None

        self.apply(
            lambda t, **kw: t.make_plot3d(**kw),
            parallel=parallel,
            cache=True,
            random_c=random_c,
            show_root=show_root,
            **kwargs,
        )
