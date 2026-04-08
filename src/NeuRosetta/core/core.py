from typing import Iterable, Iterator, Callable, TypeVar, Any, List
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import inspect
# from concurrent.futures import ProcessPoolExecutor, as_completed

from graph_tool.all import Graph
from tqdm import tqdm

T = TypeVar("T")


def _is_iterable(x):
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))


class _Stone(object):
    """Core single item class"""

    # constructor
    def __init__(self, ID: int, metadata: dict) -> None:
        self.ID = ID
        self.metadata = metadata


class _Tree(_Stone):
    """Underlying tree graph class"""

    # constructor
    def __init__(self, ID: int, metadata: dict, graph: Graph) -> None:
        super().__init__(ID, metadata)
        self.graph = graph


class _Forest(Sequence):
    """Ordered, mutable container of Tree objects"""

    def __init__(self, trees: Iterable[Any] = ()) -> None:
        self._trees: list[Any] = []
        self._id_index: dict[int, int] = {}
        self.extend(trees)

    def __len__(self) -> int:
        return len(self._trees)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._trees)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return _Forest(self._trees[idx])
        return self._trees[idx]

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
        for tree in trees:
            self._check_id(tree)
        for tree in trees:
            self._id_index[tree.ID] = len(self._trees)
            self._trees.append(tree)

    def insert(self, index: int, tree: Any) -> None:
        self._check_id(tree)
        self._trees.insert(index, tree)
        self._rebuild_index(index)

    def remove(self, tree: Any) -> None:
        self.pop(self._id_index[tree.ID])

    def remove_id(self, ID: int) -> None:
        self.pop(self._id_index[ID])

    def pop(self, index: int = -1) -> Any:
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
        return self._trees[self._id_index[ID]]

    def ids(self) -> list[int]:
        return [tree.ID for tree in self._trees]

    def map(self, fn: Callable[[Any], Any]) -> list[Any]:
        """Sequential map over trees"""
        return [fn(tree) for tree in self._trees]

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
        """
        Unified apply:
        - If bind=True AND fn accepts 'bind', results are not collected.
        - Otherwise returns list of results.

        Supports broadcasting of args/kwargs and optional parallel execution.
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
        sig = inspect.signature(fn)
        accepts_bind = "bind" in sig.parameters

        # --- per-tree call
        def call(i: int):
            call_kwargs = {k: v[i] for k, v in norm_kwargs.items()}

            if accepts_bind:
                call_kwargs["bind"] = bind

            return fn(
                self._trees[i],
                *(arg[i] for arg in norm_args),
                **call_kwargs,
            )

        indices = range(n)

        # --- sequential
        if not parallel:
            it = indices
            if show_progress:
                it = tqdm(it, total=n, desc="Processing trees")

            if bind and accepts_bind:
                for i in it:
                    call(i)
                return None
            else:
                return [call(i) for i in it]

        # --- parallel
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = ex.map(call, indices)

            if show_progress:
                results = tqdm(results, total=n, desc="Processing trees")

            if bind and accepts_bind:
                # exhaust iterator without collecting
                for _ in results:
                    pass
                return None
            else:
                return list(results)
