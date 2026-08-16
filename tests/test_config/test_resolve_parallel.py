import pytest

from neurosetta.config import configure, get_settings, resolve_parallel, settings
from neurosetta.config.settings import _reset_settings_for_tests


@pytest.fixture(autouse=True)
def _fresh_settings():
    _reset_settings_for_tests()
    yield
    _reset_settings_for_tests()


def test_scope_defaults_without_configuration():
    assert resolve_parallel(explicit=None, scope="io") is False
    assert resolve_parallel(explicit=None, scope="forest") is True
    assert resolve_parallel(explicit=None, scope="default") is False


def test_explicit_kwarg_wins_over_global():
    configure(parallel_io=True)
    assert resolve_parallel(explicit=False, scope="io") is False


def test_configure_scoped_parallel():
    configure(parallel_io=True, parallel_forest=False)
    assert resolve_parallel(explicit=None, scope="io") is True
    assert resolve_parallel(explicit=None, scope="forest") is False


def test_configure_parallel_sets_all_scopes():
    configure(parallel=True)
    s = get_settings()
    assert s.parallel.io is True
    assert s.parallel.forest is True
    assert s.parallel.default is True


def test_context_manager_overrides_global():
    configure(parallel_io=False)
    with settings(parallel_io=True):
        assert resolve_parallel(explicit=None, scope="io") is True
    assert resolve_parallel(explicit=None, scope="io") is False


def test_context_scoped_override_does_not_leak():
    with settings(parallel_forest=False):
        assert resolve_parallel(explicit=None, scope="forest") is False
    assert resolve_parallel(explicit=None, scope="forest") is True


def test_default_units_dimensionless():
    assert get_settings().default_units == "dimensionless"
