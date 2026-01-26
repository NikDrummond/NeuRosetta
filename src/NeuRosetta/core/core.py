from typing import Iterable, Iterator, Callable, TypeVar, Any
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from graph_tool.all import Graph
from tqdm import tqdm

T = TypeVar("T")

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

    def apply(self, method_name: str, *args, **kwargs) -> list[Any]:
        """Sequentially call a method by name on each tree"""
        return self.map(lambda t: getattr(t, method_name)(*args, **kwargs))

    ### parallel mapping

    def _map(
        self,
        fn: Callable[[Any], T],
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = False,
    ) -> list[T]:
        """
        Generic map over trees with optional threading and optional progress bar.
        """
        if not parallel:
            if show_progress:
                return [fn(tree) for tree in tqdm(self._trees, desc="Processing trees")]
            else:
                return [fn(tree) for tree in self._trees]

        # Parallel mode
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = ex.map(fn, self._trees)
            if show_progress:
                results = tqdm(results, total=len(self._trees), desc="Processing trees")
            return list(results)