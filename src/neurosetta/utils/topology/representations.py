"""Persistence-image representations via GUDHI."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._gudhi import require_gudhi
from .diagrams import validate_diagram


def _unweighted(_x: Any) -> float:
    """Constant weight for Kanari-style unweighted persistence images."""
    return 1.0


def _as_diagram_list(diagrams: Sequence[np.ndarray]) -> list[NDArray[np.float64]]:
    return [validate_diagram(d) for d in diagrams]


def fit_persistence_image(
    diagrams: Sequence[np.ndarray],
    *,
    bandwidth: float = 1.0,
    resolution: tuple[int, int] = (20, 20),
    im_range: Sequence[float] | None = None,
    weight: Callable[..., float] | None = None,
) -> Any:
    """Fit a GUDHI :class:`~gudhi.representations.PersistenceImage` on a diagram collection.

    Forest / dataset usage must fit **one** transformer so all trees share the
    same image coordinate range. For Kanari TMD reproducibility use constant
    weighting (the default here): ``weight = lambda x: 1.0``.
    """
    require_gudhi()
    from gudhi.representations import PersistenceImage

    diags = _as_diagram_list(diagrams)
    w = _unweighted if weight is None else weight
    kwargs: dict[str, Any] = {
        "bandwidth": bandwidth,
        "weight": w,
        "resolution": list(resolution),
    }
    if im_range is not None:
        kwargs["im_range"] = list(im_range)

    transformer = PersistenceImage(**kwargs)
    transformer.fit(diags)
    return transformer


def persistence_images(
    diagrams: Sequence[np.ndarray],
    *,
    transformer: Any | None = None,
    bandwidth: float = 1.0,
    resolution: tuple[int, int] = (20, 20),
    im_range: Sequence[float] | None = None,
    weight: Callable[..., float] | None = None,
) -> tuple[NDArray[np.float64], Any]:
    """Transform diagrams to persistence images with a shared fitted range.

    Parameters
    ----------
    diagrams : sequence of ndarray
        Persistence diagrams of shape ``(N_i, 2)``.
    transformer : PersistenceImage or None, optional
        Pre-fitted transformer. If None, fit on ``diagrams``.
    bandwidth, resolution, im_range, weight
        Forwarded to :func:`fit_persistence_image` when fitting.

    Returns
    -------
    images : ndarray
        Shape ``(n_diagrams, resolution[0] * resolution[1])`` (GUDHI flat vectors).
    transformer : PersistenceImage
        Fitted transformer (reuse for new trees).
    """
    diags = _as_diagram_list(diagrams)
    if transformer is None:
        transformer = fit_persistence_image(
            diags,
            bandwidth=bandwidth,
            resolution=resolution,
            im_range=im_range,
            weight=weight,
        )
    images = np.asarray(transformer.transform(diags), dtype=np.float64)
    return images, transformer


def reshape_persistence_image(
    image: np.ndarray,
    resolution: tuple[int, int] = (20, 20),
) -> NDArray[np.float64]:
    """Reshape a flat persistence-image vector to ``(resolution[0], resolution[1])``."""
    arr = np.asarray(image, dtype=np.float64).ravel()
    expected = int(resolution[0]) * int(resolution[1])
    if arr.size != expected:
        raise ValueError(
            f"Image length {arr.size} does not match resolution {resolution} ({expected})"
        )
    return arr.reshape(int(resolution[0]), int(resolution[1]))
