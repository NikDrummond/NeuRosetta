"""Functions for reading and writing .swc files"""

### Imports
from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from pathlib import Path
from typing import TypeVar, overload

from numpy import savetxt

from ..core import _Forest, _Tree
from ..utils.units import apply_voxel_metadata, is_voxel_units, normalize_units_str
from .io_utils import (
    _apply_import_units,
    _base_meta,
    _foreach_with_progress,
    _graph_from_table,
    _map_with_progress,
    _swc_table,
    _table_from_swc,
    resolve_import_id,
    safe_export_stem,
)
from .swc_meta import (
    parse_swc_header,
    swc_header_for_tree,
    units_from_swc_header,
    warn_if_export_dimensionless,
)

T = TypeVar("T")

IdResolver = Callable[[Path], Hashable]


### Import swc


@overload
def import_swc(
    fpath: str | Path,
    *,
    ID: Hashable | None = None,
    id_map: Mapping[str, Hashable] | None = None,
    id_resolver: IdResolver | None = None,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> _Tree: ...


@overload
def import_swc(
    fpath: str | Path,
    *,
    ID: Hashable | None = None,
    id_map: Mapping[str, Hashable] | None = None,
    id_resolver: IdResolver | None = None,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> _Forest: ...


def import_swc(
    fpath: str | Path,
    *,
    ID: Hashable | None = None,
    id_map: Mapping[str, Hashable] | None = None,
    id_resolver: IdResolver | None = None,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
):
    """
    Import one or more SWC morphology files as trees.

    This function imports neuron morphology data stored in SWC format and
    converts it into neurosetta.Tree objects. If a single `.swc` file is
    provided, a single Tree is returned. If a directory is provided,
    all `.swc` files in that directory are imported and returned as a
    Forest.

    Identity
    --------
    * ``name`` is always set to the file stem (arbitrary strings allowed).
    * ``ID`` (logical neuron identifier) defaults to the same stem as a
      string, with **no** integer coercion. Override with ``ID=`` (single
      file), ``id_map`` (stem → ID), or ``id_resolver(path)``.

    Parameters
    ----------
    fpath : str or pathlib.Path
        Path to a `.swc` file or to a directory containing `.swc` files.
    ID : hashable or None, optional
        Explicit logical ID for a **single-file** import. Ignored for
        directory imports (use ``id_map`` / ``id_resolver``). Default None.
    id_map : mapping or None, optional
        Map file stems to logical IDs (useful for batch imports).
    id_resolver : callable or None, optional
        ``(path: Path) -> hashable`` used when ``ID`` / ``id_map`` do not
        supply an ID.
    set_units : str or None, optional
        Spatial units of coordinates in the imported file(s), e.g. ``"nm"``,
        ``"micron"``, or ``"voxel"``. When provided, ``metadata["units"]`` is
        set without rescaling geometry. Default is None (dimensionless).
    voxel_size : float or None, optional
        Edge length of one voxel when ``set_units="voxel"``. Required together
        with ``voxel_unit`` for voxel coordinates.
    voxel_unit : str or None, optional
        Spatial unit for the voxel edge length, e.g. ``"nm"``.
    parallel : bool, optional
        If True, import multiple SWC files in parallel when loading from a
        directory. Default is False.
    max_workers : int or None, optional
        Maximum number of worker processes or threads to use when
        ``parallel=True``. If None, the default executor configuration
        is used. Default is None.
    show_progress : bool, optional
        If True, display a progress indicator while importing multiple SWC
        files. Default is False.

    Returns
    -------
    Tree or Forest
        A single ``Tree`` instance if exactly one SWC file is imported,
        otherwise a ``Forest`` containing all imported trees.

    Raises
    ------
    FileNotFoundError
        If ``fpath`` does not exist or if no `.swc` files are found in the
        specified directory.
    ValueError
        If ``ID=`` is supplied for a multi-file directory import.

    Examples
    --------
    Import a single SWC file (stem becomes both name and ID)::

        tree = import_swc("cell_A.swc")
        assert tree.name == tree.ID == "cell_A"

    Explicit logical ID, filename-derived name::

        tree = import_swc("720575940630123456_full.swc", ID="720575940630123456")

    Batch import with independent IDs::

        forest = import_swc(
            "swc/",
            id_resolver=lambda p: p.stem.split("_")[0],
        )
    """

    from ..api import Forest, Tree

    p = Path(fpath)

    def _import_one(path: Path, *, explicit_id: Hashable | None = None) -> _Tree:
        df = _table_from_swc(str(path))
        graph = _graph_from_table(df)

        tree_id = resolve_import_id(
            path,
            ID=explicit_id,
            id_map=id_map,
            id_resolver=id_resolver,
        )
        name = path.stem
        header_meta = parse_swc_header(path)
        meta = _base_meta()
        if header_units := units_from_swc_header(header_meta):
            meta["units"] = normalize_units_str(header_units)
        if (
            is_voxel_units(meta.get("units"))
            and "voxel_size" in header_meta
            and "voxel_unit" in header_meta
        ):
            apply_voxel_metadata(
                meta,
                header_meta["voxel_size"],
                header_meta["voxel_unit"],
            )
        meta["file_path"] = str(path)
        meta["isReduced"] = False

        tree = Tree(ID=tree_id, metadata=meta, graph=graph, name=name)
        _apply_import_units(
            tree,
            set_units,
            voxel_size=voxel_size,
            voxel_unit=voxel_unit,
        )
        return tree

    if p.is_file():
        return _import_one(p, explicit_id=ID)

    if not p.is_dir():
        raise FileNotFoundError(f"Path not found: {p}")

    if ID is not None:
        raise ValueError(
            "ID= is only valid for single-file import; use id_map= or id_resolver= for directories"
        )

    swcs = sorted(p.glob("*.swc"))

    if not swcs:
        raise FileNotFoundError(f"No .swc files found in directory: {p}")

    if len(swcs) == 1:
        return _import_one(swcs[0])

    trees = _map_with_progress(
        _import_one,
        swcs,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
        desc="Importing .swc files",
    )

    return Forest(trees)


### Export swc


@overload
def export_swc(
    tree: _Tree,
    fpath: str | Path,
    *,
    header: str | None = None,
) -> None: ...


@overload
def export_swc(
    tree: _Forest,
    fpath: str | Path,
    *,
    header: str | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> None: ...


def export_swc(
    tree: _Tree,
    fpath: str | Path,
    *,
    header: str | None = None,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> None:
    """
    Export a tree or forest to SWC morphology files.

    This function serializes a ``Tree`` or all trees in a ``Forest`` to SWC
    format. A single tree is written to one `.swc` file, while a forest is
    written as one file per tree, named ``<tree.name>.swc`` by default.

    Parameters
    ----------
    tree : Tree or Forest
        The tree or forest to export.
    fpath : str or pathlib.Path
        Output file or directory path. For a single ``Tree``, if ``fpath`` is a
        directory the file ``<tree.name>.swc`` is written inside it; if it is a
        file path, that path is used directly (``name`` is not mutated). For a
        ``Forest``, ``fpath`` must be a directory and each tree is written as
        ``<tree.name>.swc`` inside it.
    header : str or None, optional
        Custom header text to include at the top of each SWC file. If None,
        a default header identifying the generator and serialized metadata
        (including units) is used. Default is None.
    parallel : bool, optional
        If True, export multiple trees in parallel when ``tree`` is a
        ``Forest``. Default is False.
    max_workers : int or None, optional
        Maximum number of worker processes or threads to use when
        ``parallel=True``. If None, the default executor configuration
        is used. Default is None.
    show_progress : bool, optional
        If True, display a progress indicator while exporting multiple trees.
        Default is False.

    Returns
    -------
    None
        This function is called for its side effects and does not return a
        value.

    Raises
    ------
    ValueError
        If attempting to export a ``Forest`` to a single SWC file path
        (i.e. when ``fpath`` has a file suffix).

    Examples
    --------
    Export a single tree to a file::

        export_swc(tree, "out/cell_A_full.swc")

    Export a single tree to a directory (uses ``tree.name``)::

        export_swc(tree, "swc_out/")
    """

    p = Path(fpath)

    def _write_one(t):
        df = _swc_table(t)
        warn_if_export_dimensionless(t)
        header_txt = swc_header_for_tree(t, header=header)
        out = p / f"{safe_export_stem(t.name)}.swc"
        savetxt(out, df, header=header_txt)

    # ---- Single Tree ----
    if not isinstance(tree, _Forest):
        out = p / f"{safe_export_stem(tree.name)}.swc" if p.exists() and p.is_dir() else p

        df = _swc_table(tree)
        warn_if_export_dimensionless(tree)
        header_txt = swc_header_for_tree(tree, header=header)
        savetxt(out, df, header=header_txt)
        return

    # ---- Forest ----
    if p.suffix:
        raise ValueError("Cannot write a Forest to a single SWC file")

    p.mkdir(parents=True, exist_ok=True)

    trees = list(tree)

    _foreach_with_progress(
        _write_one,
        trees,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
        desc="Writing SWC files",
    )
