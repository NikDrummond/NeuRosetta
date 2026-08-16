import os

import pytest

from neurosetta.config import configure, get_settings, sync_vedo_runtime
from neurosetta.config.env import load_env_into_settings
from neurosetta.config.openmp import apply_openmp
from neurosetta.config.settings import Settings, _reset_settings_for_tests


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for key in list(os.environ):
        if key.startswith("NEUROSETTA_"):
            monkeypatch.delenv(key, raising=False)
    _reset_settings_for_tests()
    yield
    _reset_settings_for_tests()


def test_configure_openmp_applies_to_graph_tool():
    configure(openmp_num_threads=3)
    from graph_tool import openmp as gt_openmp

    assert gt_openmp.openmp_get_num_threads() == 3


def test_load_env_parallel_and_openmp(monkeypatch):
    monkeypatch.setenv("NEUROSETTA_PARALLEL_IO", "1")
    monkeypatch.setenv("NEUROSETTA_GT_OPENMP_THREADS", "2")
    monkeypatch.setenv("NEUROSETTA_VEDO_BACKEND", "vtk")

    s = Settings()
    load_env_into_settings(s)
    apply_openmp(s.openmp)

    assert s.parallel.io is True
    assert s.openmp.num_threads == 2
    assert s.vedo.backend == "vtk"


def test_sync_vedo_runtime_uses_config_not_configure():
    configure(vedo_backend="vtk", vedo_parallel_projection=False, apply_openmp_now=False)

    from vedo import settings as vedo_runtime

    vedo_runtime.default_backend = "changed"
    sync_vedo_runtime()

    assert vedo_runtime.default_backend == "vtk"
    assert vedo_runtime.use_parallel_projection is False


def test_vedo_not_applied_on_configure_only():
    from vedo import settings as vedo_runtime

    original_backend = vedo_runtime.default_backend
    configure(vedo_backend="vtk", apply_openmp_now=False)
    assert get_settings().vedo.backend == "vtk"
    assert vedo_runtime.default_backend == original_backend


def test_ensure_runtime_env_sets_omp_wait_policy(monkeypatch):
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    from neurosetta.config.env import ensure_runtime_env

    ensure_runtime_env()
    assert os.environ["OMP_WAIT_POLICY"] == "passive"
