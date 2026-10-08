"""The diagnostic runner keeps backend isolation and measurement boundaries."""

from __future__ import annotations

import copy
import importlib.util
import io
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtGui import QImage

from stavellum.rendering import _rhi, render

path = Path(__file__).resolve().parents[1] / "scripts/profile_rhi.py"
spec = importlib.util.spec_from_file_location("rhi_profile", path)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


@pytest.mark.parametrize("previous", [None, "original.dll"])
def test_native_library_restores_environment_even_after_failure(monkeypatch, previous):
    name = "STAVELLUM_RHI_DLL"
    if previous is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, previous)
    with pytest.raises(RuntimeError, match="failure"):
        with profile.native_library(Path("baseline.dll")):
            assert os.environ[name] == "baseline.dll"
            raise RuntimeError("failure")
    assert os.environ.get(name) == previous


@pytest.mark.parametrize("archive_name", ["stavellum", "historical_package"])
@pytest.mark.parametrize("layout", ["flat", "subpackages"])
def test_baseline_isolates_historical_assets_and_optional_native_module(tmp_path, monkeypatch, archive_name, layout):
    prefix_path = "rendering/" if layout == "subpackages" else ""
    namespace_path = "rendering." if layout == "subpackages" else ""
    sources = {
        "__init__": b"",
        prefix_path + "render": b"class FrameRenderer:\n    pass\n",
        prefix_path + "_rhi": b"class RhiTarget:\n    pass\n",
        prefix_path + "rhi": (
            f"from {archive_name}.{namespace_path}_rhi import RhiTarget\n"
            f"from {archive_name}.{namespace_path}render import FrameRenderer\n"
            "from importlib.resources import files\n"
            f"RESOURCE = files('{archive_name}').joinpath('baseline-marker.txt').read_text()\n"
            "class RhiFrameRenderer:\n    pass\n"
        ).encode(),
    }
    if prefix_path:
        sources[prefix_path + "__init__"] = b""
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("src/", b"")
        output.writestr(f"src/{archive_name}/", b"")
        output.writestr(f"src/{archive_name}/baseline-marker.txt", b"historical resources")
        for name, source in sources.items():
            output.writestr(f"src/{archive_name}/{name}.py", source)

    def git(arguments, **kwargs):
        if arguments[1] == "rev-parse":
            return SimpleNamespace(stdout="a" * 40 + "\n")
        assert arguments[1] == "archive"
        return SimpleNamespace(stdout=archive.getvalue())

    monkeypatch.setattr(profile._baseline.subprocess, "run", git)
    prefix = "stavellum._benchmark_baseline_"
    previous = {key: value for key, value in sys.modules.items() if key.startswith(prefix)}
    try:
        old, commit = profile.baseline_renderer("old", tmp_path / "old")
        namespace = sys.modules[old.__module__]
        assert namespace.RhiTarget is not _rhi.RhiTarget
        assert namespace.FrameRenderer is not render.FrameRenderer
        assert namespace.RESOURCE == "historical resources"
        assert sys.modules["stavellum.rendering._rhi"] is _rhi
        assert commit == "a" * 40
        ablation, _ = profile.baseline_renderer("old", tmp_path / "ablation", current_native=True)
        assert sys.modules[ablation.__module__].RhiTarget is _rhi.RhiTarget
        assert namespace.RhiTarget is not _rhi.RhiTarget  # Previously loaded baseline stays isolated.
    finally:
        for key in list(sys.modules):
            if key.startswith(prefix):
                del sys.modules[key]
        sys.modules.update(previous)


def test_counter_delta_preserves_missing_measurements_and_excludes_cache_peaks():
    before = {"gpu_upload_bytes": 64, "gpu_execution_seconds": None,
              "command_pack_seconds": .1, "native_begin_frame_seconds": .2,
              "gpu_texture_cache_hits": 2, "gpu_texture_cache_misses": 3,
              "gpu_texture_cache_evictions": 1,
              "tile_cache_report": {"tile_cache_hits": 4, "tile_cache_misses": 2,
                                    "svg_raster_seconds": .3, "cache_peak_bytes": 100}}
    after = {"gpu_upload_bytes": 80, "gpu_execution_seconds": None,
             "command_pack_seconds": .15, "native_begin_frame_seconds": .25,
             "gpu_texture_cache_hits": 9, "gpu_texture_cache_misses": 4,
             "gpu_texture_cache_evictions": 2,
             "tile_cache_report": {"tile_cache_hits": 10, "tile_cache_misses": 2,
                                   "svg_raster_seconds": .3, "cache_peak_bytes": 200}}
    preserved = copy.deepcopy((before, after))
    delta = profile.counter_delta(before, after)
    assert delta["gpu_upload_bytes"] == 16
    assert delta["command_pack_seconds"] == pytest.approx(.05)
    assert delta["native_begin_frame_seconds"] == pytest.approx(.05)
    assert delta["gpu_texture_cache_hits"] == 7
    assert delta["gpu_texture_cache_misses"] == delta["gpu_texture_cache_evictions"] == 1
    assert "gpu_execution_seconds" not in delta
    assert delta["tile_cache_report"] == {"tile_cache_hits": 6, "tile_cache_misses": 0, "svg_raster_seconds": 0}
    assert (before, after) == preserved


class FakeRenderer:
    scene = SimpleNamespace(settings=SimpleNamespace(fps=60))

    def __init__(self, failure_index=None):
        self.failure_index = failure_index
        self.count = 0
        self.seen = set()
        self.streams = []

    def backend_report(self):
        return {"render_seconds": self.count / 1000, "tile_cache_hits": self.count - len(self.seen),
                "tile_cache_misses": len(self.seen), "gpu_upload_bytes": 8 * len(self.seen)}

    def export_frames(self, times, cancel):
        owner = self

        class Stream:
            pixel_format = "bgra"

            def __init__(self):
                self.times = enumerate(times)
                self.closed = False

            def __next__(self):
                index, seconds = next(self.times)
                if cancel() or index == owner.failure_index:
                    raise InterruptedError("cancelled")
                owner.count += 1
                owner.seen.add(seconds)
                image = QImage(2, 1, QImage.Format.Format_ARGB32)
                image.fill(0xff000000 | owner.count)
                return index, image

            def close(self):
                self.closed = True

        stream = Stream()
        self.streams.append(stream)
        return stream


def test_profile_retains_owned_cold_snapshots_and_measures_only_warm_counter_delta(monkeypatch):
    def no_encoder(*args, **kwargs):
        raise AssertionError("Render-only diagnostics must not start an encoder")

    monkeypatch.setattr(profile._baseline.subprocess, "run", no_encoder)
    renderer = FakeRenderer()
    result, snapshots = profile.profile(renderer, 0, frames=3, warm_frames=4)
    assert len(result["frame_samples_ms"]) == 3
    assert result["backend_report"]["tile_cache_report"]["tile_cache_misses"] == 3
    assert result["fixed_warm"]["counter_delta"]["tile_cache_report"]["tile_cache_misses"] == 0
    assert result["fixed_warm"]["counter_delta"]["gpu_upload_bytes"] == 0
    assert len(result["fixed_warm"]["frame_samples_ms"]) == 4
    assert renderer.count == 3 + 1 + 4
    assert all(stream.closed for stream in renderer.streams)
    assert [snapshots[index].pixel(0, 0) & 0xff for index in sorted(snapshots)] == [1, 2, 3]


def test_consume_closes_stream_after_cancellation():
    renderer = FakeRenderer(failure_index=1)
    with pytest.raises(InterruptedError, match="cancelled"):
        profile.consume(renderer, [0, .5, 1])
    assert renderer.count == 1 and renderer.streams[0].closed


def test_gpu_identity_matches_opengl_suffix_and_rejects_different_devices():
    def identity(renderer):
        return profile.gpu_identity({"gpu_info": {"renderer": renderer}})

    assert identity(" NVIDIA GeForce Example GPU/PCIe/SSE2 ") == identity("nvidia geforce example gpu")
    assert identity("NVIDIA Corporation NVIDIA GeForce Example GPU/PCIe/SSE2 4.6.0 NVIDIA 000.00") == identity(
        "NVIDIA GeForce Example GPU")
    assert identity("NVIDIA GeForce Example GPU") != identity("Intel Arc GPU")
