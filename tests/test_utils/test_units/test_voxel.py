"""Tests for voxel coordinate units."""

import pytest

from NeuRosetta.utils.units import is_voxel_units
from NeuRosetta.utils.units.conversion import scale_factor
from NeuRosetta.utils.units.equality import units_are_equal
from NeuRosetta.utils.units.parsing import normalize_units_str
from NeuRosetta.utils.units.voxel import apply_voxel_metadata, validate_voxel_metadata


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
