"""Tests for persistence-image helpers."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neurosetta.utils.topology import (
    fit_persistence_image,
    persistence_images,
    reshape_persistence_image,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("gudhi") is None,
    reason="GUDHI not installed",
)


def test_persistence_image_common_range_and_shape():
    diags = [
        np.array([[0.0, 1.0], [0.5, 2.0]]),
        np.array([[0.0, 3.0]]),
        np.array([[1.0, 1.2], [2.0, 4.0]]),
    ]
    images, transformer = persistence_images(
        diags,
        bandwidth=1.0,
        resolution=(10, 8),
    )
    assert images.shape == (3, 10 * 8)
    assert hasattr(transformer, "im_range_fixed_")

    # Fit on one diagram alone → different coordinate range
    single = fit_persistence_image(diags[:1], resolution=(10, 8))
    assert not np.allclose(transformer.im_range_fixed_, single.im_range_fixed_)

    img_shared, _ = persistence_images(diags[:1], transformer=transformer)
    img_alone, _ = persistence_images(diags[:1], transformer=single)
    assert not np.allclose(img_shared, img_alone)


def test_reshape_persistence_image():
    vec = np.arange(20, dtype=float)
    img = reshape_persistence_image(vec, resolution=(4, 5))
    assert img.shape == (4, 5)
