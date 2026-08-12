"""Tests for unit conversion factors."""

import pytest

from neurosetta.utils.units.conversion import scale_factor


def test_scale_factor_identity():
    assert scale_factor("nanometer", "nanometer") == 1.0


def test_scale_factor_nm_to_micron():
    assert scale_factor("nm", "micron") == pytest.approx(0.001)


def test_scale_factor_dimensionless_raises():
    with pytest.raises(ValueError, match="dimensionless"):
        scale_factor("dimensionless", "micron")
