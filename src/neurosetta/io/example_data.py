"""Access bundled tutorial example morphologies from ``docs/data``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, overload

from ..core import _Forest, _Tree
from .nr_utils import load

ExampleFormat = Literal["nr", "swc"]


def example_data_dir(*, format: ExampleFormat = "nr") -> Path:
    """
    Return the path to bundled tutorial example data.

    Example morphologies from the FAFB-FlyWire dataset live under
    ``docs/data/nr/`` and ``docs/data/swc/`` in the repository.

    Parameters
    ----------
    format : {"nr", "swc"}, optional
        Which example file format subdirectory to return. Default is ``"nr"``.

    Returns
    -------
    pathlib.Path
        Directory containing bundled ``*.nr`` or ``*.swc`` example files.

    Raises
    ------
    FileNotFoundError
        If the example data directory cannot be located.
    """
    suffix = format
    roots = (
        Path(__file__).resolve().parent.parent / "example_data",
        Path(__file__).resolve().parents[3] / "docs" / "data",
    )
    for root in roots:
        subdir = root / format
        if subdir.is_dir() and any(subdir.glob(f"*.{suffix}")):
            return subdir

    msg = "Example data not found. Reinstall neurosetta or run from a source checkout."
    raise FileNotFoundError(msg)


@overload
def load_example_data(
    tree_id: int,
    *,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> _Tree: ...


@overload
def load_example_data(
    tree_id: None = None,
    *,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    show_progress: bool = False,
) -> _Forest: ...


def load_example_data(
    tree_id: int | None = None,
    *,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    show_progress: bool = False,
):
    """
    Load bundled tutorial example morphologies from ``docs/data/nr/``.

    By default, all example ``.nr`` files are loaded as a ``Forest``. Pass
    ``tree_id`` to load a single neuron (file stem / FlyWire segment ID).

    Parameters
    ----------
    tree_id : int or None, optional
        FlyWire segment ID of the example neuron to load. When ``None``, load
        all bundled examples. Default is ``None``.
    set_units, voxel_size, voxel_unit, parallel, max_workers, show_progress
        Forwarded to :func:`~neurosetta.io.load`.

    Returns
    -------
    Tree or Forest
        A single example tree or a forest of all bundled examples.

    Examples
    --------
    Load all tutorial examples::

        forest = load_example_data()

    Load one example by segment ID::

        tree = load_example_data(720575940596125868)
    """
    data_dir = example_data_dir(format="nr")
    kwargs = {
        "set_units": set_units,
        "voxel_size": voxel_size,
        "voxel_unit": voxel_unit,
        "parallel": parallel,
        "max_workers": max_workers,
        "show_progress": show_progress,
    }
    if tree_id is not None:
        return load(data_dir / f"{tree_id}.nr", **kwargs)
    return load(data_dir, **kwargs)
