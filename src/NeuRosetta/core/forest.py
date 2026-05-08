"""Container class for multiple trees"""

from typing import Iterable, Iterator, Callable, TypeVar, Any
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import inspect
from functools import lru_cache
import warnings


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


class _Forest(Sequence):
    """Ordered, mutable container of Tree objects"""

    def __init__(self, trees: Iterable[Any] = ()) -> None:
        self._trees: list[Any] = []
        self._id_index: dict[int, int] = {}
        self.extend(trees)

    ### improve this, placeholder for now
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

    # def extend(self, trees: Iterable[Any]) -> None:
    #     trees = list(trees)
    #     for tree in trees:
    #         self._check_id(tree)
    #     for tree in trees:
    #         self._id_index[tree.ID] = len(self._trees)
    #         self._trees.append(tree)
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

    def remove_id(self, ID: int) -> None:
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

    def by_id(self, ID: int) -> Any:
        if ID not in self._id_index:
            raise KeyError(f"No tree with ID {ID}")
        return self._trees[self._id_index[ID]]

    def ids(self) -> list[int]:
        return [tree.ID for tree in self._trees]

    def map(self, fn: Callable[[Any], Any]) -> list[Any]:
        """Sequential map over trees (convenience alias for apply with parallel=False)."""
        return self.apply(fn, parallel=False)

    ### filtering
    def filter(
        self,
        predicate: Callable[[dict], bool] | None = None,
        **conditions,
    ) -> "_Forest":
        """
        Return a new _Forest containing only trees whose metadata satisfies the given conditions.

        Two modes (mutually exclusive):

        1. Keyword equality — all conditions must match (AND logic):
            forest.filter(Neuron_type='T4', Neuron_subtype='a')

        2. Callable predicate — full control over the logic:
            forest.filter(lambda m: m.get('Neuron_type') == 'T4' or m.get('score', 0) > 5)

        Missing metadata keys are treated as non-matching (no KeyError raised).
        """
        if predicate is not None and conditions:
            raise ValueError(
                "Provide either a predicate or keyword conditions, not both."
            )
        if predicate is not None:
            return type(self)(t for t in self._trees if predicate(t.metadata))
        return type(self)(
            t
            for t in self._trees
            if all(t.metadata.get(k) == v for k, v in conditions.items())
        )

    ### parallel mapping

    def apply(
        self,
        fn: Callable[..., T],
        *args,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = False,
        bind: bool = False,
        **kwargs,
    ) -> list[T] | None:
        """apply to all neurons in paralllel"""
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
            bind = False  # reset so we don't silently discard results below

        # --- per-tree call
        def call(i: int):
            call_kwargs = {k: v[i] for k, v in norm_kwargs.items()}
            if accepts_bind:
                call_kwargs["bind"] = bind
            return fn(self._trees[i], *(arg[i] for arg in norm_args), **call_kwargs)

        indices = range(n)

        if not parallel:
            it = (
                tqdm(indices, total=n, desc="Processing trees")
                if show_progress
                else indices
            )
            if bind and accepts_bind:
                for i in it:
                    call(i)
                return None
            return [call(i) for i in it]

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = ex.map(call, indices)
            if show_progress:
                results = tqdm(results, total=n, desc="Processing trees")
            if bind and accepts_bind:
                for _ in results:
                    pass
                return None
            return list(results)
