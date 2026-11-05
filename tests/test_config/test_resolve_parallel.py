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


def test_parallel_default_fills_unset_scopes():
    configure(parallel_default=True)
    assert resolve_parallel(explicit=None, scope="io") is True
    assert resolve_parallel(explicit=None, scope="forest") is True
    assert resolve_parallel(explicit=None, scope="default") is True
    assert get_settings().parallel.io is True
    assert get_settings().parallel.forest is True


def test_parallel_default_does_not_override_explicit_scope():
    configure(parallel_io=False, parallel_default=True)
    assert resolve_parallel(explicit=None, scope="io") is False
    assert resolve_parallel(explicit=None, scope="forest") is True

    _reset_settings_for_tests()
    configure(parallel_default=True, parallel_forest=False)
    assert resolve_parallel(explicit=None, scope="io") is True
    assert resolve_parallel(explicit=None, scope="forest") is False


def test_parallel_default_false_overrides_forest_scoped_default():
    configure(parallel_default=False)
    assert resolve_parallel(explicit=None, scope="io") is False
    assert resolve_parallel(explicit=None, scope="forest") is False


def test_context_parallel_default_overrides_unset_scopes_only():
    configure(parallel_io=True)
    with settings(parallel_default=False):
        assert resolve_parallel(explicit=None, scope="io") is True
        assert resolve_parallel(explicit=None, scope="forest") is False
    assert resolve_parallel(explicit=None, scope="forest") is True

    _reset_settings_for_tests()
    with settings(parallel_default=True):
        assert resolve_parallel(explicit=None, scope="io") is True
    assert resolve_parallel(explicit=None, scope="io") is False


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


def test_io_progress_follows_configure(monkeypatch):
    from neurosetta.io.io_utils import _map_with_progress

    configure(show_progress=True, parallel_default=True)
    seen: dict = {}

    def fake_tqdm(it, total, desc):
        seen["desc"] = desc
        seen["total"] = total
        return it

    class _Pool:
        def __init__(self, max_workers=None):
            seen["workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, item):
            class _Fut:
                def result(self):
                    return fn(item)

            return _Fut()

    monkeypatch.setattr("neurosetta.io.io_utils.tqdm", fake_tqdm)
    monkeypatch.setattr("neurosetta.io.io_utils.ThreadPoolExecutor", _Pool)

    assert _map_with_progress(lambda x: x + 1, [1, 2], desc="Loading .nr files") == [2, 3]
    assert seen == {"desc": "Loading .nr files", "total": 2, "workers": None}

    seen.clear()
    assert _map_with_progress(
        lambda x: x,
        [1],
        show_progress=False,
        parallel=False,
        desc="off",
    ) == [1]
    assert seen == {}
