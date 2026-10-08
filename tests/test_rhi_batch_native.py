"""Exercise the batch ABI with real Vulkan, independently of Python batching."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from native_support import require_vulkan_device

from stavellum.rendering._rhi import library_path

NATIVE_BATCH_SCRIPT = r'''
import ctypes, json, os, threading
from pathlib import Path
import PySide6
from stavellum.rendering._rhi import Quad, _OwnedFrame, library_path
from stavellum.domain.models import RenderSettings
from stavellum.graphics.qt import prepare_render_app

prepare_render_app(RenderSettings(render_backend="gpu"))
directory = os.add_dll_directory(str(Path(PySide6.__file__).parent))
dll = ctypes.CDLL(str(library_path()))
pointer = ctypes.c_void_p
class BatchItem(ctypes.Structure):
    _fields_ = [("quads", ctypes.POINTER(Quad)), ("count", ctypes.c_size_t)]

for name, result, arguments in (
    ("sprhi_abi_version", ctypes.c_uint32, ()),
    ("sprhi_create", pointer, (ctypes.c_int32, ctypes.c_int32, ctypes.c_uint64, ctypes.c_char_p)),
    ("sprhi_upload", ctypes.c_int, (pointer, ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, pointer)),
    ("sprhi_remove", ctypes.c_int, (pointer, ctypes.c_uint64)),
    ("sprhi_submit", ctypes.c_int, (pointer, ctypes.POINTER(Quad), ctypes.c_size_t, pointer, ctypes.c_size_t)),
    ("sprhi_submit_owned", ctypes.c_int, (pointer, ctypes.POINTER(Quad), ctypes.c_size_t, ctypes.POINTER(_OwnedFrame))),
    ("sprhi_submit_batch_owned", ctypes.c_int, (pointer, ctypes.POINTER(BatchItem), ctypes.c_size_t, ctypes.POINTER(_OwnedFrame))),
    ("sprhi_release_frame", None, (pointer,)),
    ("sprhi_report", ctypes.c_char_p, (pointer,)),
    ("sprhi_last_error", ctypes.c_char_p, (pointer,)),
    ("sprhi_close", ctypes.c_int, (pointer,)),
):
    function = getattr(dll, name)
    function.restype, function.argtypes = result, arguments

assert dll.sprhi_abi_version() == 3
assert ctypes.sizeof(BatchItem) == 16 and ctypes.sizeof(_OwnedFrame) == 24
width, height = 65, 49
size = width * height * 4
handle = dll.sprhi_create(width, height, 0, b"vulkan")
assert handle, dll.sprhi_last_error(None)

def report():
    return json.loads(dll.sprhi_report(handle))

source = (ctypes.c_uint8 * 16)(255, 0, 0, 255, 19, 83, 201, 255,
                             0, 0, 255, 255, 255, 255, 255, 255)
assert dll.sprhi_upload(handle, 1, 2, 2, 8, source) == 0
assert dll.sprhi_upload(handle, 2, 2, 2, 8, source) == 0

def quad(texture=0, x=0, y=0, w=width, h=height, color=(1, 1, 1, 1)):
    return Quad(texture, x, y, w, h, 0, 0, 1, 1, *color)

patterns = [
    [quad(color=(1, 0, 0, 1))],
    [],
    [quad(texture=1)],
    [quad(x=2.25, y=3.5, w=19.75, h=27.25, color=(0, .5, 0, .5)),
     quad(x=15.125, y=21.25, w=33.5, h=22.5, color=(0, 0, .75, .75))],
    [quad(texture=1, x=-3.5, w=width / 2), quad(texture=1, x=width / 2, w=width / 2),
     quad(texture=2, y=height / 2, h=height / 2)],
    [quad(color=(.3, .2, .1, 1)), quad(color=(.25, 0, 0, .25)),
     quad(x=3.75, y=9.5, w=45.25, h=21.5, color=(0, 0, .5, .5))],
    [quad(texture=2, x=17.25, y=2.5, w=38.75, h=31.25, color=(.4,) * 4)],
    [quad(color=(.125, .25, .875, 1)), quad(x=32.25, y=21.75, w=7.5, h=9.25)],
]
commands = [(Quad * len(values))(*values) for values in patterns]
expected = []
for values in commands:
    frame = _OwnedFrame()
    assert dll.sprhi_submit_owned(handle, values, len(values), ctypes.byref(frame)) == 0
    expected.append(ctypes.string_at(frame.pixels, frame.size))
    dll.sprhi_release_frame(frame.owner)
    copy = (ctypes.c_uint8 * size)()
    assert dll.sprhi_submit(handle, values, len(values), copy, size) == 0
    assert bytes(copy) == expected[-1]
assert expected[0][:4] == bytes((0, 0, 255, 255))
assert expected[1] == bytes((0, 0, 0, 255)) * (width * height)
assert expected[2][:4] == bytes((0, 0, 255, 255))
assert expected[2][-4:] == bytes((255, 255, 255, 255))

retained = []
indices = [7, 2, 7, 0, 1, 5, 4, 6, 3, 0, 2, 1, 6, 7, 4, 5, 3]
for batch_size in (1, 2, 4, 8):
    for offset in range(0, len(indices), batch_size):
        group = indices[offset:offset + batch_size]
        items = (BatchItem * len(group))(*(BatchItem(commands[i], len(commands[i])) for i in group))
        frames = (_OwnedFrame * len(group))()
        assert dll.sprhi_submit_batch_owned(handle, items, len(group), frames) == 0, dll.sprhi_last_error(handle)
        for index, frame in zip(group, frames):
            assert frame.owner and frame.pixels and frame.size == size
            saved = ctypes.string_at(frame.pixels, frame.size)
            assert saved == expected[index], (batch_size, offset, index)
            retained.append((frame, saved))

# Each invalid member rejects the entire batch before the GPU starts, clears
# every output slot, and leaves the renderer usable for a subsequent batch.
invalid = [
    BatchItem(None, 1), BatchItem(commands[0], 1_000_001),
    BatchItem((Quad * 1)(quad(texture=9472)), 1),
    BatchItem((Quad * 1)(quad(x=float("nan"))), 1),
    BatchItem((Quad * 1)(quad(w=-1)), 1),
    BatchItem((Quad * 1)(quad(color=(1, 1, 1, float("inf")))), 1),
]
for bad in invalid:
    before = report()["gpu_submission_count"]
    items = (BatchItem * 2)(BatchItem(commands[0], 1), bad)
    frames = (_OwnedFrame * 2)(_OwnedFrame(1, 2, 3), _OwnedFrame(4, 5, 6))
    assert dll.sprhi_submit_batch_owned(handle, items, 2, frames) != 0
    assert all(not frame.owner and not frame.pixels and frame.size == 0 for frame in frames)
    assert report()["gpu_submission_count"] == before
    items[1] = BatchItem(commands[1], 0)
    assert dll.sprhi_submit_batch_owned(handle, items, 2, frames) == 0
    for index, frame in enumerate(frames):
        assert ctypes.string_at(frame.pixels, frame.size) == expected[index]
        dll.sprhi_release_frame(frame.owner)

for count in (0, 9):
    frames = (_OwnedFrame * 9)()
    assert dll.sprhi_submit_batch_owned(handle, None, count, frames) != 0
    assert b"between 1 and 8" in dll.sprhi_last_error(handle)
for pointer_handle, items in ((None, (BatchItem * 2)()), (handle, None)):
    frames = (_OwnedFrame * 2)(_OwnedFrame(1, 2, 3), _OwnedFrame(4, 5, 6))
    assert dll.sprhi_submit_batch_owned(pointer_handle, items, 2, frames) != 0
    assert all(not frame.owner and not frame.pixels and frame.size == 0 for frame in frames)
assert dll.sprhi_submit_batch_owned(handle, (BatchItem * 1)(), 1, None) != 0

thread_errors = []
def wrong_thread():
    frame = _OwnedFrame(1, 2, 3)
    status = dll.sprhi_submit_batch_owned(handle, (BatchItem * 1)(), 1, ctypes.byref(frame))
    if status == 0 or frame.owner or frame.pixels or frame.size:
        thread_errors.append("wrong-thread batch was accepted or published output")
thread = threading.Thread(target=wrong_thread)
thread.start()
thread.join(10)
assert not thread.is_alive() and not thread_errors

statistics = report()
assert statistics["gpu_batch_peak_size"] == 8
assert statistics["readback_mode"] == "rhi-batch-sync"
assert statistics["gpu_submission_count"] == sum(statistics["gpu_batch_size_histogram"].values())
assert statistics["gpu_submitted_frame_count"] == sum(
    int(key) * value for key, value in statistics["gpu_batch_size_histogram"].items())
assert statistics["readback_output_peak_bytes"] == 8 * size
assert statistics["readback_staging_peak_bytes"] == 8 * size
assert statistics["readback_buffer_peak_bytes"] == 8 * size
assert statistics["memory_copy_bytes"] == len(patterns) * size
assert statistics["inplace_format_conversion_bytes"] == (
    statistics["owned_readback_frame_count"] * size
    if statistics["readback_format"] == "RGBA8" else 0)
assert dll.sprhi_remove(handle, 1) == 0 and dll.sprhi_remove(handle, 2) == 0
assert dll.sprhi_close(handle) == 0

# Writer-like readers and final releases work after all GPU resources close.
failures = []
def release_frames():
    try:
        while retained:
            frame, saved = retained.pop()
            assert ctypes.string_at(frame.pixels, frame.size) == saved
            dll.sprhi_release_frame(frame.owner)
    except BaseException as error:
        failures.append(repr(error))
thread = threading.Thread(target=release_frames)
thread.start()
thread.join(10)
assert not thread.is_alive() and not failures and not retained, failures
directory.close()
print(json.dumps(statistics), flush=True)
'''


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "win32", reason="Windows native renderer required")
@pytest.mark.parametrize("rgba", [False, True])
def test_native_batch_preserves_pixels_atomic_errors_and_independent_cpu_owners(rgba, monkeypatch):
    if not library_path().exists():
        pytest.skip("Native Vulkan renderer DLL has not been built")
    require_vulkan_device()
    if rgba:
        monkeypatch.setenv("STAVELLUM_RHI_RGBA_READBACK", "1")
    else:
        monkeypatch.delenv("STAVELLUM_RHI_RGBA_READBACK", raising=False)
    result = subprocess.run(
        [sys.executable, "-c", NATIVE_BATCH_SCRIPT], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["readback_format"] == ("RGBA8" if rgba else "BGRA8")
