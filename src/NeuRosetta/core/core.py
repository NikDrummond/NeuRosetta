from typing import Iterable, Iterator, Callable, TypeVar, Any, List
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
# from concurrent.futures import ProcessPoolExecutor, as_completed

from graph_tool.all import Graph, openmp_set_num_threads
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

    def apply(self, method_name: str, *args, **kwargs) -> list[Any]:
        """Sequentially call a method by name on each tree"""
        return self.map(lambda t: getattr(t, method_name)(*args, **kwargs))

    ### parallel mapping

    def apply_fn(
        self,
        fn: Callable[..., T],
        *args,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = False,
        **kwargs,
    ) -> list[T]:
        """
        Apply an arbitrary function to each tree with argument broadcasting.
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

        # --- per-tree callables
        def call(i: int) -> T:
            return fn(
                self._trees[i],
                *(arg[i] for arg in norm_args),
                **{k: v[i] for k, v in norm_kwargs.items()},
            )

        indices = range(n)

        if not parallel:
            it = indices
            if show_progress:
                it = tqdm(it, total=n, desc="Processing trees")
            return [call(i) for i in it]

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = ex.map(call, indices)
            if show_progress:
                results = tqdm(results, total=n, desc="Processing trees")
            return list(results)

    def foreach(
        self,
        fn: Callable[..., Any],
        *args,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = False,
        **kwargs,
    ) -> None:
        """
        Apply a function to each tree for side effects only.

        Unlike apply_fn, this does NOT collect return values.
        """

        _ = self.apply_fn(
            fn,
            *args,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
            **kwargs,
        )
        return None

    # def apply_fn(
    #     self,
    #     fn: Callable[..., Any],
    #     *args,
    #     parallel: bool = True,
    #     max_workers: int | None = None,
    #     batch_size: int = 10,
    #     show_progress: bool = False,
    #     graph_tool_threads: int | None = None,
    #     **kwargs,
    # ) -> List[Any]:
    #     """
    #     Apply a function to each neuron graph.

    #     Each neuron is processed independently, using multiprocessing if requested.
    #     Supports batching to reduce memory usage.

    #     Args:
    #         fn: function of signature fn(tree, *args, **kwargs)
    #         parallel: use multiprocessing if True
    #         max_workers: number of processes to spawn
    #         batch_size: number of neurons per process chunk
    #         show_progress: show tqdm progress bar
    #         graph_tool_threads: if using graph_tool, limit internal threads per neuron
    #     """

    #     n = len(self)

    #     # --- normalize positional args
    #     norm_args: list[list] = []
    #     for a in args:
    #         if hasattr(a, "__len__") and not isinstance(a, (str, bytes)):
    #             if len(a) != n:
    #                 raise ValueError("Iterable argument length mismatch")
    #             norm_args.append(list(a))
    #         else:
    #             norm_args.append([a] * n)

    #     # --- normalize keyword args
    #     norm_kwargs: dict[str, list] = {}
    #     for k, v in kwargs.items():
    #         if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
    #             if len(v) != n:
    #                 raise ValueError(f"Iterable kwarg '{k}' length mismatch")
    #             norm_kwargs[k] = list(v)
    #         else:
    #             norm_kwargs[k] = [v] * n

    #     # --- per-tree call function
    #     def call(i: int) -> Any:
    #         # optional: limit graph_tool threads
    #         if graph_tool_threads is not None:
                
    #             openmp_set_num_threads(graph_tool_threads)

    #         try:
    #             return fn(
    #                 self._trees[i],
    #                 *(arg[i] for arg in norm_args),
    #                 **{k: v[i] for k, v in norm_kwargs.items()},
    #             )
    #         except Exception as e:
    #             raise RuntimeError(f"Error processing tree ID {self._trees[i].ID}") from e

    #     if not parallel or n <= batch_size:
    #         # sequential fallback
    #         it = range(n)
    #         if show_progress:
    #             it = tqdm(it, total=n, desc="Processing neurons")
    #         return [call(i) for i in it]

    #     # --- multiprocessing with batching
    #     results: list[Any] = [None] * n  # preallocate

    #     indices = list(range(n))
    #     batches = [
    #         indices[i:i + batch_size] for i in range(0, n, batch_size)
    #     ]

    #     with ProcessPoolExecutor(max_workers=max_workers) as executor:
    #         futures = {
    #             executor.submit(
    #                 lambda batch: [call(i) for i in batch], batch
    #             ): batch
    #             for batch in batches
    #         }

    #         if show_progress:
    #             pbar = tqdm(total=n, desc="Processing neurons")

    #         for fut in as_completed(futures):
    #             batch = futures[fut]
    #             res_batch = fut.result()
    #             for idx, val in zip(batch, res_batch):
    #                 results[idx] = val
    #             if show_progress:
    #                 pbar.update(len(batch))

    #         if show_progress:
    #             pbar.close()

    #     return results

    # def foreach(
    #     self,
    #     fn: Callable[..., Any],
    #     *args,
    #     parallel: bool = True,
    #     max_workers: int | None = None,
    #     batch_size: int = 10,
    #     show_progress: bool = False,
    #     graph_tool_threads: int | None = None,
    #     **kwargs,
    # ) -> None:
    #     """Apply function to each neuron graph for side effects only."""
    #     _ = self.apply_fn(
    #         fn,
    #         *args,
    #         parallel=parallel,
    #         max_workers=max_workers,
    #         batch_size=batch_size,
    #         show_progress=show_progress,
    #         graph_tool_threads=graph_tool_threads,
    #         **kwargs,
    #     )
    #     return None