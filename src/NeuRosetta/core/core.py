from graph_tool.all import Graph
from collections.abc import Iterable, Iterator, Sequence
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor

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

    def __init__(self, trees: Iterable[_Tree] = ()) -> None:
        self._trees: list[_Tree] = []
        self._id_index: dict[int, int] = {}
        self.extend(trees)

    def __len__(self) -> int:
        return len(self._trees)

    def __iter__(self) -> Iterator[_Tree]:
        return iter(self._trees)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return _Forest(self._trees[idx])
        return self._trees[idx]

    def _check_id(self, tree: _Tree) -> None:
        if tree.ID in self._id_index:
            raise ValueError(f"Duplicate Tree ID: {tree.ID}")

    def _rebuild_index(self, start: int = 0) -> None:
        for i in range(start, len(self._trees)):
            self._id_index[self._trees[i].ID] = i

    def append(self, tree: _Tree) -> None:
        self._check_id(tree)
        self._id_index[tree.ID] = len(self._trees)
        self._trees.append(tree)

    def extend(self, trees: Iterable[_Tree]) -> None:
        for tree in trees:
            self._check_id(tree)

        for tree in trees:
            self._id_index[tree.ID] = len(self._trees)
            self._trees.append(tree)

    def insert(self, index: int, tree: _Tree) -> None:
        self._check_id(tree)
        self._trees.insert(index, tree)
        self._rebuild_index(index)

    def remove(self, tree: _Tree) -> None:
        self.pop(self._id_index[tree.ID])

    def remove_id(self, ID: int) -> None:
        self.pop(self._id_index[ID])

    def pop(self, index: int = -1) -> _Tree:
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

    def by_id(self, ID: int) -> _Tree:
        return self._trees[self._id_index[ID]]

    def ids(self) -> list[int]:
        return [tree.ID for tree in self._trees]

    def map(self, fn: Callable[[_Tree], Any]) -> list[Any]:
        return [fn(tree) for tree in self._trees]

    def apply(self, method_name: str, *args, **kwargs):
        return self.map(lambda t: getattr(t, method_name)(*args, **kwargs))

    def map_threads(
        self,
        fn: Callable[[_Tree], Any],
        max_workers: int | None = None,
    ):
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(fn, self._trees))
