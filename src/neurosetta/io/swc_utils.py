"""Functions for reading and writing .swc files"""

### Imports
from __future__ import annotations

from pathlib import Path
from typing import overload, TypeVar

from numpy import savetxt
from graph_tool.all import Graph
from tqdm import tqdm

from ..core import _Tree, _Forest
from .io_utils import (
    _table_from_swc,
    _graph_from_table,
    _swc_table,
    _base_meta,
    _apply_import_units,
    _foreach_with_progress,
    _map_with_progress
)
from .swc_meta import (
    parse_swc_header,
    swc_header_for_tree,
    units_from_swc_header,
    warn_if_export_dimensionless,
)
from ..utils.units import apply_voxel_metadata, is_voxel_units, normalize_units_str


T = TypeVar("T")

### Import swc


@overload
def import_swc(
    fpath: str | Path,
    *,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> _Tree: ...


@overload
def import_swc(
    fpath: str | Path,
    *,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> _Forest: ...


def import_swc(
    fpath: str | Path,
    *,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool = False,
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

    Parameters
    ----------
    fpath : str or pathlib.Path
        Path to a `.swc` file or to a directory containing `.swc` files.
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

    Notes
    -----
    The tree identifier is inferred from the file stem and converted to
    an integer.     Imported trees are assigned default metadata, including the source file
    path and dimensionless units. Units may be read from SWC ``# Meta:`` header
    comments when present; ``set_units`` overrides header values.

    Examples
    --------
    Import a single SWC file::

        tree = import_swc("42.swc")

    Import with voxel units assigned::

        tree = import_swc("42.swc", set_units="voxel", voxel_size=8, voxel_unit="nm")

    Import all SWC files from a directory::

        forest = import_swc("swc_files/", parallel=True, progress=True)
    """

    from ..api import Tree, Forest

    p = Path(fpath)

    def _import_one(path: Path) -> _Tree:
        df = _table_from_swc(str(path))
        graph = _graph_from_table(df)

        tree_id = int(path.stem)
        header_meta = parse_swc_header(path)
        meta = _base_meta()
        if header_units := units_from_swc_header(header_meta):
            meta["units"] = normalize_units_str(header_units)
        if is_voxel_units(meta.get("units")):
            if "voxel_size" in header_meta and "voxel_unit" in header_meta:
                apply_voxel_metadata(
                    meta,
                    header_meta["voxel_size"],
                    header_meta["voxel_unit"],
                )
        meta["file_path"] = str(path)
        meta["isReduced"] = False

        tree = Tree(ID=tree_id, metadata=meta, graph=graph)
        _apply_import_units(
            tree,
            set_units,
            voxel_size=voxel_size,
            voxel_unit=voxel_unit,
        )
        return tree

    if p.is_file():
        return _import_one(p)

    if not p.is_dir():
        raise FileNotFoundError(f"Path not found: {p}")

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
    parallel: bool = False,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> None: ...


def export_swc(
    tree: _Tree,
    fpath: str | Path,
    *,
    header: str | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> None:

    """
    Export a tree or forest to SWC morphology files.

    This function serializes a ``Tree`` or all trees in a ``Forest`` to SWC
    format. A single tree is written to one `.swc` file, while a forest is
    written as one file per tree, named `<tree_id>.swc`.

    Parameters
    ----------
    tree : Tree or Forest
        The tree or forest to export.
    fpath : str or pathlib.Path
        Output file or directory path. For a single ``Tree``, if ``fpath`` is a
        directory the file ``<tree_id>.swc`` is written inside it; if it is a
        file path, that path is used directly. For a ``Forest``, ``fpath`` must
        be a directory and each tree is written as ``<tree_id>.swc`` inside it.
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

    Notes
    -----
    The SWC table is generated from each tree using the internal table
    conversion utilities. When exporting a forest, the output directory
    is created if it does not already exist.

    Examples
    --------
    Export a single tree to a file::

        export_swc(tree, "42.swc")

    Export a single tree to a directory::

        export_swc(tree, "swc_out/")

    Export a forest in parallel::

        export_swc(forest, "swc_out/", parallel=True, progress=True)
    """

    p = Path(fpath)

    def _write_one(t):
        df = _swc_table(t)
        warn_if_export_dimensionless(t)
        header_txt = swc_header_for_tree(t, header=header)
        out = p / f"{t.ID}.swc"
        savetxt(out, df, header=header_txt)

    # ---- Single Tree ----
    if not isinstance(tree, _Forest):
        if p.exists() and p.is_dir():
            out = p / f"{tree.ID}.swc"
        else:
            out = p

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
