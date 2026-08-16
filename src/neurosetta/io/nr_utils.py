from pathlib import Path
from typing import overload

from graph_tool.all import load_graph

from ..core import _Forest, _Tree
from .io_utils import _apply_import_units, _bind_core, _foreach_with_progress, _map_with_progress


@overload
def load(
    fpath: str | Path,
    *,
    tree_id: int | None = None,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> _Tree: ...


@overload
def load(
    fpath: str | Path,
    *,
    tree_id: int | None = None,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> _Forest: ...


def load(
    fpath: str | Path,
    *,
    tree_id: int | None = None,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
):
    """
    Load one or more trees from disk.

    Load serialized tree graphs stored in `.nr` files. If a
    single file is provided, a single ``Tree`` instance is returned. If a
    directory is provided, behaviour depends on ``tree_id``: when set, only
    ``<tree_id>.nr`` is loaded; when one ``.nr`` file exists, that tree is
    loaded; when multiple exist, all are loaded as a ``Forest``.

    Parameters
    ----------
    fpath : str or pathlib.Path
        Path to a `.nr` file or to a directory containing `.nr` files.
    tree_id : int or None, optional
        Identifier of the tree to load when ``fpath`` points to a directory.
        If provided, the file `<tree_id>.nr` is loaded directly.
        Default is None.
    set_units : str or None, optional
        Override ``metadata["units"]`` after loading, without rescaling geometry.
        Default is None (keep units stored in the file).
    voxel_size : float or None, optional
        Voxel edge length when overriding units to ``"voxel"``.
    voxel_unit : str or None, optional
        Spatial unit for the voxel edge length when overriding to voxels.
    parallel : bool, optional
        If True, load multiple `.nr` files in parallel when loading a
        directory containing more than one tree. Default is False.
    max_workers : int or None, optional
        Maximum number of worker processes or threads to use when
        ``parallel=True``. If None, the default executor configuration
        is used. Default is None.
    show_progress : bool, optional
        If True, display a progress indicator while loading multiple trees.
        Default is False.

    Returns
    -------
    Tree or Forest
        A single ``Tree`` instance if exactly one tree is loaded, otherwise
        a ``Forest`` containing all loaded trees.

    Raises
    ------
    FileNotFoundError
        If ``fpath`` does not exist, if no `.nr` files are found in a directory,
        or if the requested ``tree_id`` file does not exist.

    Notes
    -----
    Each loaded graph has its source file path recorded in
    ``tree.metadata["file_path"]`` (backed by ``graph.gp["metadata"]``).

    Examples
    --------
    Load a single tree from a file::

        tree = load("42.nr")

    Load all trees from a directory::

        forest = load("trees/")

    Load a specific tree from a directory::

        tree = load("trees/", tree_id=42)
    """

    from ..api import Forest, Tree
    from ..core.metadata import set_core_meta

    p = Path(fpath)

    def _load_one(path: Path) -> _Tree:
        g = load_graph(str(path), fmt="gt")
        set_core_meta(g.gp["metadata"], "file_path", str(path))
        tree = Tree.from_graph(g)
        _apply_import_units(
            tree,
            set_units,
            voxel_size=voxel_size,
            voxel_unit=voxel_unit,
        )
        return tree

    if p.is_file():
        return _load_one(p)

    if not p.is_dir():
        raise FileNotFoundError(f"Path not found: {p}")

    candidates = sorted(p.glob("*.nr"))

    if tree_id is not None:
        return _load_one(p / f"{tree_id}.nr")

    if not candidates:
        raise FileNotFoundError(f"No .nr files found in directory: {p}")

    if len(candidates) == 1:
        return _load_one(candidates[0])

    trees = _map_with_progress(
        _load_one,
        candidates,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
        desc="Loading .nr files",
    )

    return Forest(trees)


@overload
def save(
    tree: _Tree,
    fpath: str | Path | None = None,
) -> Path: ...


@overload
def save(
    tree: _Forest,
    fpath: str | Path,
    *,
    parallel: bool | None = None,
    max_workers: int | None = None,
    progress: bool = False,
) -> list[Path]: ...


def save(
    tree: _Tree | _Forest,
    fpath: str | Path | None = None,
    *,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
):
    """
    Save a tree or forest to disk in `.nr` format.

    This function serializes a ``Tree`` or all trees in a ``Forest`` to `.nr`
    files. A single tree is saved as one file, while a forest is saved as one
    file per tree, named `<tree_id>.nr`.

    Parameters
    ----------
    tree : Tree or Forest
        The tree or forest to save.
    fpath : str or pathlib.Path or None, optional
        Output path. For a single ``Tree``, ``None`` writes ``<tree_id>.nr`` in
        the current working directory; a directory writes inside it; a file path
        is used directly. For a ``Forest``, use a directory (or ``None`` for
        the current working directory); each tree is saved as ``<tree_id>.nr``.
        Default is None.
    parallel : bool, optional
        If True, save multiple trees in parallel when ``tree`` is a ``Forest``.
        Default is False.
    max_workers : int or None, optional
        Maximum number of worker processes or threads to use when
        ``parallel=True``. If None, the default executor configuration
        is used. Default is None.
    show_progress : bool, optional
        If True, display a progress indicator while saving multiple trees.
        Default is False.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If attempting to save a ``Forest`` to a single file path (i.e. when
        ``fpath`` has a file suffix).

    Notes
    -----
    Prior to saving, each tree is bound to its core representation to ensure
    that all required graph properties are present.

    Examples
    --------
    Save a single tree to the current directory::

        save(tree)

    Save a single tree to a specific file::

        save(tree, "output/42.nr")

    Save a forest to a directory::

        save(forest, "output_trees/", parallel=True, progress=True)
    """

    def _save_one(t, base: Path) -> Path:
        _bind_core(t)
        out = base / f"{t.ID}.nr"
        t.graph.save(str(out), fmt="gt")
        return out

    # ---- Single Tree ----
    if not isinstance(tree, _Forest):
        _bind_core(tree)

        if fpath is None:
            out = Path.cwd() / f"{tree.ID}.nr"
        else:
            p = Path(fpath)
            if p.exists() and p.is_dir():
                out = p / f"{tree.ID}.nr"
            else:
                if p.suffix:
                    out = p
                else:
                    p.mkdir(parents=True, exist_ok=True)
                    out = p / f"{tree.ID}.nr"

        tree.graph.save(str(out), fmt="gt")
        return

    # ---- Forest ----
    base = Path.cwd() if fpath is None else Path(fpath)

    if base.suffix:
        raise ValueError("Cannot save a Forest to a single .nr file")

    base.mkdir(parents=True, exist_ok=True)

    trees = list(tree)
    out_paths: list[Path] = []

    def _wrapped_save(t):
        p = _save_one(t, base)
        out_paths.append(p)

    _foreach_with_progress(
        _wrapped_save,
        trees,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
        desc="Saving .nr files",
    )

    return
