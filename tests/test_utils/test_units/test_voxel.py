"""Tests for voxel coordinate units."""

import pytest

from neurosetta.utils.units import is_voxel_units
from neurosetta.utils.units.conversion import scale_factor
from neurosetta.utils.units.equality import units_are_equal
from neurosetta.utils.units.parsing import normalize_units_str
from neurosetta.utils.units.voxel import (
    apply_voxel_metadata,
    clear_voxel_metadata,
    meters_per_coordinate_unit,
    validate_voxel_metadata,
    voxel_spec_from_metadata,
)


def test_is_voxel_units():
    assert is_voxel_units("voxel")
    assert not is_voxel_units("nm")


def test_validate_voxel_metadata():
    meta = {"voxel_size": 8, "voxel_unit": "nm"}
    assert validate_voxel_metadata(meta) == (8.0, "nanometer")


def test_scale_factor_voxel_to_nanometer():
    meta = {"units": "voxel", "voxel_size": 8, "voxel_unit": "nanometer"}
    assert scale_factor("voxel", "nanometer", from_metadata=meta) == pytest.approx(8.0)


def test_scale_factor_nanometer_to_voxel():
    meta = {"units": "voxel", "voxel_size": 8, "voxel_unit": "nanometer"}
    assert scale_factor("nanometer", "voxel", to_metadata=meta) == pytest.approx(0.125)


def test_scale_factor_voxel_to_micron():
    meta = {"units": "voxel", "voxel_size": 8, "voxel_unit": "nm"}
    assert scale_factor("voxel", "micron", from_metadata=meta) == pytest.approx(0.008)


def test_units_are_equal_distinguishes_voxel_sizes():
    meta_a = {"units": "voxel", "voxel_size": 8, "voxel_unit": "nm"}
    meta_b = {"units": "voxel", "voxel_size": 10, "voxel_unit": "nm"}
    assert not units_are_equal("voxel", "voxel", meta_a, meta_b)


def test_apply_voxel_metadata_normalizes_unit():
    meta: dict = {}
    apply_voxel_metadata(meta, 8, "nm")
    assert meta["voxel_size"] == 8.0
    assert meta["voxel_unit"] == "nanometer"
    assert normalize_units_str("voxel") == "voxel"


def test_validate_voxel_metadata_missing_keys_raises():
    with pytest.raises(ValueError, match="Voxel units require"):
        validate_voxel_metadata({})


def test_validate_voxel_metadata_nonpositive_size_raises():
    with pytest.raises(ValueError, match="voxel_size must be positive"):
        validate_voxel_metadata({"voxel_size": 0, "voxel_unit": "nm"})


def test_clear_voxel_metadata():
    meta = {"voxel_size": 8, "voxel_unit": "nanometer", "units": "voxel"}
    clear_voxel_metadata(meta)
    assert "voxel_size" not in meta
    assert "voxel_unit" not in meta


def test_meters_per_coordinate_unit_nanometer():
    meters = meters_per_coordinate_unit("nm")
    assert meters == pytest.approx(1e-9)


def test_meters_per_coordinate_unit_voxel():
    meta = {"voxel_size": 8, "voxel_unit": "nanometer"}
    meters = meters_per_coordinate_unit("voxel", metadata=meta)
    assert meters == pytest.approx(8e-9)


def test_meters_per_coordinate_unit_voxel_requires_metadata():
    with pytest.raises(ValueError, match="require metadata"):
        meters_per_coordinate_unit("voxel")


def test_voxel_spec_from_metadata():
    meta = {"units": "voxel", "voxel_size": 8, "voxel_unit": "nm"}
    assert voxel_spec_from_metadata(meta) == (8.0, "nanometer")


def test_voxel_spec_from_metadata_non_voxel_returns_none():
    assert voxel_spec_from_metadata({"units": "nm"}) is None
