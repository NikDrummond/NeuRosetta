"""Tests for unit string parsing and normalization."""

import pytest

from NeuRosetta.utils.units.equality import units_are_equal
from NeuRosetta.utils.units.parsing import (
    DEFAULT_UNITS,
    is_dimensionless,
    normalize_units_str,
    parse_units,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, DEFAULT_UNITS),
        ("undefined", DEFAULT_UNITS),
        ("Undefined", DEFAULT_UNITS),
        ("dimensionless", DEFAULT_UNITS),
        ("nm", "nanometer"),
        ("nanometer", "nanometer"),
        ("um", "micron"),
        ("microns", "micron"),
        ("µm", "micron"),
    ],
)
def test_normalize_units_str(raw, expected):
    assert normalize_units_str(raw) == expected


def test_parse_units_unknown_raises():
    with pytest.raises(ValueError, match="Unknown units"):
        parse_units("foobar")


def test_units_are_equal_aliases():
    assert units_are_equal("nm", "nanometer")
    assert not units_are_equal("nm", "micron")


def test_is_dimensionless():
    assert is_dimensionless("dimensionless")
    assert is_dimensionless("undefined")
    assert not is_dimensionless("nm")


def test_parse_units_voxel_raises():
    with pytest.raises(ValueError, match="Voxel units are metadata-backed"):
        parse_units("voxel")
