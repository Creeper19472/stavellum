"""Verify the Rust Vulkan DLL through the same external ABI as production."""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from native_support import require_vulkan_device

ROOT = Path(__file__).resolve().parents[1]
DLL = ROOT / "src/stavellum/native/rhi/stavellum_rust.dll"


def test_installed_rust_library_precedes_legacy(tmp_path, monkeypatch):
    from stavellum import _rhi

    module = tmp_path / "src/stavellum/_rhi.py"
    module.parent.mkdir(parents=True)
    module.touch()
    resources = module.parent / "native/rhi"
    resources.mkdir(parents=True)
    for name in ("stavellum_rust.dll", "stavellum_rhi.dll"):
        (resources / name).touch()
    monkeypatch.setattr(_rhi, "__file__", str(module))
    monkeypatch.delenv("STAVELLUM_RHI_DLL", raising=False)
    assert _rhi.library_path() == resources / "stavellum_rust.dll"


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "win32" or not DLL.exists(), reason="Built Rust DLL required")
def test_rust_vulkan_instancing_and_owned_pixels():
    require_vulkan_device()
    script = r'''
import ctypes, json, sys, threading
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "scripts"))
from benchmark_rust import bind, Quad, Frame, Batch
dll = bind(sys.argv[1])
assert dll.sprhi_abi_version() == 3
handle = dll.sprhi_create(65, 49, 1024, b"vulkan")
assert handle, dll.sprhi_last_error(None)
def quad(color):
    return Quad(0, (ctypes.c_float * 12)(0, 0, 65, 49, 0, 0, 1, 1, *color))
red = (Quad * 1)(quad((1, 0, 0, 1)))
blue = (Quad * 1)(quad((0, 0, 1, 1)))
blend = (Quad * 2)(quad((1, 0, 0, 1)), quad((0, 0, .5, .5)))
items = (Batch * 4)(Batch(red, 1), Batch(blue, 1), Batch(blend, 2), Batch(None, 0))
frames = (Frame * 4)()
assert dll.sprhi_submit_batch_owned(handle, items, 4, frames) == 0, dll.sprhi_last_error(handle)
expected = [bytes((0, 0, 255, 255)), bytes((255, 0, 0, 255)),
            bytes((128, 0, 127, 255)), bytes((0, 0, 0, 255))]
for index, frame in enumerate(frames):
    assert frame.size == 65 * 49 * 4
    pixels = ctypes.string_at(frame.pixels, frame.size)
    if index == 2:
        assert pixels[0] in (127, 128) and pixels[2] in (127, 128)
    else:
        assert pixels == expected[index] * (65 * 49)
before = json.loads(dll.sprhi_report(handle))
assert before["renderer_implementation"] == "rust-ash"
assert before["gpu_info"]["msaa_samples"] == 4
assert before["gpu_submission_count"] == 1
assert before["gpu_instance_count"] == 4
assert before["gpu_draw_call_count"] == 3
assert before["instance_stride_bytes"] == 48
bad = (Quad * 1)(quad((1, 1, 1, 1)))
bad[0].values[0] = float("nan")
outputs = (Frame * 2)(Frame(1, 2, 3), Frame(4, 5, 6))
invalid = (Batch * 2)(Batch(red, 1), Batch(bad, 1))
assert dll.sprhi_submit_batch_owned(handle, invalid, 2, outputs) != 0
assert all(not f.owner and not f.pixels and not f.size for f in outputs)
assert json.loads(dll.sprhi_report(handle))["gpu_submission_count"] == 1
results = []
def wrong_thread():
    output = Frame(1, 2, 3)
    status = dll.sprhi_submit_batch_owned(handle, items, 1, ctypes.byref(output))
    results.append(status != 0 and not output.owner and not output.pixels and output.size == 0)
thread = threading.Thread(target=wrong_thread)
thread.start(); thread.join(10)
assert not thread.is_alive() and results == [True]
saved = [ctypes.string_at(f.pixels, f.size) for f in frames]
assert dll.sprhi_close(handle) == 0
errors = []
def release():
    try:
        for f, data in zip(frames, saved):
            assert ctypes.string_at(f.pixels, f.size) == data
            dll.sprhi_release_frame(f.owner)
    except BaseException as error:
        errors.append(repr(error))
thread = threading.Thread(target=release)
thread.start(); thread.join(10)
assert not thread.is_alive() and not errors, errors
print(json.dumps(before))
'''
    result = subprocess.run([sys.executable, "-c", script, str(DLL)], cwd=ROOT,
                            capture_output=True, text=True, timeout=120,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            env=dict(os.environ))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["graphics_api"] == "vulkan"


def test_rust_abi_geometry_sizes():
    class Quad(ctypes.Structure):
        _fields_ = [("texture", ctypes.c_uint64), ("values", ctypes.c_float * 12)]

    assert ctypes.sizeof(Quad) == 56


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "win32" or not DLL.exists(), reason="Built Rust DLL required")
def test_repeated_device_creation_releases_vulkan_handles():
    require_vulkan_device()
    script = r'''
from stavellum._rhi import RhiTarget
for index in range(160):
    target = RhiTarget(64, 64, 1, "vulkan")
    target.close()
print("160 devices created and closed")
'''
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", script], cwd=ROOT,
                            capture_output=True, text=True, timeout=120,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                            env={**os.environ, "STAVELLUM_RHI_DLL": str(DLL)})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "160 devices created and closed" in result.stdout
