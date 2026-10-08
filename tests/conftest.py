"""Keep disposable application caches out of user directories during tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def native_frame_ownership():
    from native_frames import close_evaluators

    yield
    close_evaluators()


@pytest.fixture(scope="session", autouse=True)
def session_compilation_cache_directory(tmp_path_factory):
    # Module-scoped scenes are created before function-scoped fixtures run.
    previous = os.environ.get("STAVELLUM_CACHE_DIR")
    os.environ["STAVELLUM_CACHE_DIR"] = str(tmp_path_factory.mktemp("compiled-session"))
    yield
    if previous is None:
        os.environ.pop("STAVELLUM_CACHE_DIR", None)
    else:
        os.environ["STAVELLUM_CACHE_DIR"] = previous


@pytest.fixture(autouse=True)
def compilation_cache_directory(session_compilation_cache_directory, tmp_path, monkeypatch):
    monkeypatch.setenv("STAVELLUM_CACHE_DIR", str(tmp_path / "compiled-cache"))
