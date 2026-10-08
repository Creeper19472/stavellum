"""Native readback storage must survive Qt sharing, Python views and the writer."""

from __future__ import annotations

import ctypes
import gc
import json
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
from native_support import require_vulkan_device
from PySide6.QtGui import QImage

from stavellum.rendering import _rhi
from stavellum.rendering.gpu import GpuBackendError


@pytest.fixture
def resource_layout(tmp_path, monkeypatch):
    package = tmp_path / "src/stavellum"
    package.mkdir(parents=True)
    monkeypatch.setattr(_rhi, "files", lambda name: package)
    monkeypatch.delenv("STAVELLUM_RHI_DLL", raising=False)
    packaged = package / "native/rhi/stavellum_rhi.dll"
    cached = tmp_path / ".cache/rhi/build/stavellum_rhi.dll"
    return packaged, cached


def test_explicit_native_library_override_takes_precedence(resource_layout, tmp_path, monkeypatch):
    packaged, cached = resource_layout
    for dll in (packaged, cached):
        dll.parent.mkdir(parents=True)
        dll.touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STAVELLUM_RHI_DLL", "diagnostics/explicit.dll")
    assert _rhi.library_path() == tmp_path / "diagnostics/explicit.dll"


def test_packaged_native_library_takes_precedence_over_development_cache(resource_layout):
    packaged, cached = resource_layout
    for dll in (packaged, cached):
        dll.parent.mkdir(parents=True)
        dll.touch()
    assert _rhi.library_path() == packaged


def test_development_native_library_is_used_without_packaged_resources(resource_layout):
    _, cached = resource_layout
    assert _rhi.library_path() == cached

OWNERSHIP_SCRIPT = r'''
import gc, json, os, random, sys, threading
from PySide6.QtGui import QColor, QImage
from stavellum.rendering._rhi import Quad, RhiTarget
from stavellum.domain.models import RenderSettings
from stavellum.graphics.qt import prepare_render_app

prepare_render_app(RenderSettings(render_backend="gpu"))
api = sys.argv[1]
width, height = 65, 49

def configure(copy, rgba):
    for key, enabled in (("STAVELLUM_RHI_COPY_READBACK", copy),
                         ("STAVELLUM_RHI_RGBA_READBACK", rgba)):
        if enabled:
            os.environ[key] = "1"
        else:
            os.environ.pop(key, None)

def texture():
    image = QImage(2, 2, QImage.Format.Format_RGBA8888)
    for x, y, color in ((0, 0, QColor("red")), (1, 0, QColor(19, 83, 201)),
                        (0, 1, QColor("blue")), (1, 1, QColor("white"))):
        image.setPixelColor(x, y, color)
    return image

def draw(target, source):
    ident = target.texture(source)
    return target.render([Quad(ident, 0, 0, width, height,
                              0, 0, 1, 1, 1, 1, 1, 1)])

def collect():
    gc.collect()
    # Reuse similarly sized allocations so a stale pointer is unlikely to pass.
    pressure = [bytearray([index]) * (width * height * 4) for index in range(32)]
    return pressure

baseline = None
reports = []
for copy in (False, True):
    for rgba in (False, True):
        configure(copy, rgba)
        target = RhiTarget(width, height, 0, api)
        source = texture()
        retained = []
        output = []
        for repeat in range(5):
            image = draw(target, source)
            assert image.format() == QImage.Format.Format_ARGB32
            assert image.bytesPerLine() == width * 4
            assert image.pixelColor(2, 2) == QColor("red")
            assert image.pixelColor(2, 45) == QColor("blue")
            assert image.pixelColor(64, 2) == QColor(19, 83, 201)
            assert image.pixelColor(64, 48) == QColor("white")
            output.append(bytes(image.constBits()))
            retained.append(QImage(image))
            assert not target._textures
        blended = target.render([
            Quad(0, 7, 11, 20, 15, 0, 0, 1, 1, .5, 0, 0, .5),
            Quad(0, 17, 17, 19, 18, 0, 0, 1, 1, 0, 0, .25, .25),
        ])
        assert 95 <= blended.pixelColor(20, 20).red() <= 96
        assert 63 <= blended.pixelColor(20, 20).blue() <= 64
        output.append(bytes(blended.constBits()))
        retained.append(QImage(blended))
        if baseline is None:
            baseline = output
        else:
            assert output == baseline, (api, copy, rgba, "readback bytes differ")
        report = target.report()
        assert report["readback_format"] == ("RGBA8" if rgba else "BGRA8")
        size = width * height * 4
        assert report["owned_readback_frame_count"] == (0 if copy else 6)
        assert report["copied_readback_frame_count"] == (6 if copy else 0)
        assert report["memory_copy_bytes"] == (6 * size if copy else 0)
        assert report["readback_buffer_peak_bytes"] == size * (2 if copy else 1)
        if not copy:
            assert report["memory_copy_seconds"] == 0
            assert report["readback_copy_path"] == ("rgba-inplace-sse2-swizzle" if rgba else "bgra-owned-buffer")
        else:
            assert report["memory_copy_seconds"] > 0
            assert report["readback_copy_path"] == ("rgba-sse2-swizzle" if rgba else "bgra-memcpy")
        assert report["inplace_format_conversion_bytes"] == (6 * size if rgba and not copy else 0)
        if rgba and not copy:
            assert report["inplace_format_conversion_seconds"] > 0
        reports.append(report)
        target.close()
        del target, image, blended, source
        pressure = collect()
        assert [bytes(image.constBits()) for image in retained] == baseline
        del retained, pressure

# Exercise ownership beyond a returned image and renderer's lifetime. Qt plain
# shallow copies must retain the native buffer after the Python wrapper is gone.
for rgba in (False, True):
    configure(False, rgba)
    target = RhiTarget(width, height, 0, api)
    source = texture()
    image = draw(target, source)
    saved = bytes(image.constBits())
    first = QImage(image)
    second = QImage(first)
    readonly = memoryview(image.constBits())
    target.close()
    del target, source, image, first
    pressure = collect()
    assert bytes(second.constBits()) == saved
    assert bytes(readonly) == saved

    del pressure
    target = RhiTarget(width, height, 0, api)
    image = draw(target, texture())
    # Qt detaches a shared image before mutable bits are handed to the caller.
    sibling = QImage(image)
    writable = image.bits().cast("B")
    old_byte = writable[0]
    writable[0] = old_byte ^ 0xff
    changed = bytes(image.constBits())
    assert changed != saved
    assert bytes(sibling.constBits()) == saved
    assert bytes(readonly) == saved
    target.close()
    del target, sibling
    pressure = collect()
    assert bytes(writable) == changed
    assert bytes(readonly) == saved
    # Qt bits views borrow storage; keep their owning QImages until consumption
    # ends, as the export writer does. The renderer may already be destroyed.
    del readonly, writable, image, second, pressure
    gc.collect()

    # Both view forms remain readable with the image alive after target close.
    for mutable in (False, True):
        target = RhiTarget(width, height, 0, api)
        image = draw(target, texture())
        view = memoryview(image.bits() if mutable else image.constBits())
        expected = bytes(view)
        target.close()
        del target
        pressure = collect()
        assert bytes(view) == expected, (api, rgba, mutable, "changed image view")
        del view, image, pressure
        gc.collect()

    # Alternate final references and deletion order across frames. Final release
    # happens on a writer-like thread after the graphics thread has closed.
    target = RhiTarget(width, height, 0, api)
    references = []
    for index in range(12):
        image = target.render([
            Quad(0, 0, 0, width, height, 0, 0, 1, 1,
                 (index + 1) / 16, (index + 2) / 17, (index + 3) / 18, 1),
        ])
        expected = bytes(image.constBits())
        references.append([image, QImage(image),
                           (memoryview(image.constBits()), QImage(image)), expected])
    target.close()
    del image, target
    failures = []
    def release_elsewhere(items):
        try:
            random.Random(9472).shuffle(items)
            while items:
                entry = items.pop()
                expected = entry.pop()
                random.Random(len(items)).shuffle(entry)
                while entry:
                    reference = entry.pop()
                    actual = bytes(reference.constBits()) if isinstance(reference, QImage) else bytes(reference[0])
                    assert actual == expected
                    del reference
                    gc.collect()
        except BaseException as error:
            failures.append(repr(error))
    thread = threading.Thread(target=release_elsewhere, args=(references,))
    thread.start()
    thread.join(15)
    assert not thread.is_alive(), "cross-thread native storage release stalled"
    assert not references and not failures, failures
    del references, thread
    gc.collect()

print(json.dumps(reports), flush=True)
'''


WRITER_SCRIPT = r'''
import json, os, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, "tests")
from test_export_pipeline import Process
from native_support import require_vulkan_device
from test_render import rendered_document
from stavellum.exporting import export
from stavellum.domain.models import RenderSettings
from stavellum.graphics.qt import prepare_render_app
from stavellum.rendering.render import FrameRenderer
from stavellum.presentation.scene import compile_scene

os.environ.pop("STAVELLUM_RHI_COPY_READBACK", None)
os.environ.pop("STAVELLUM_RHI_RGBA_READBACK", None)
api = sys.argv[1]
prepare_render_app(RenderSettings(render_backend="gpu"))
scene = compile_scene(rendered_document())
scene = replace(scene, settings=replace(scene.settings, cache_megabytes=0, render_backend="gpu"))
count = 12
with FrameRenderer(scene) as renderer:
    expected_frames = []
    for index in range(count):
        stream = renderer.export_frames([index / scene.settings.fps], lambda: False)
        _, image = next(stream)
        stream.close()
        expected_frames.append(bytes(image.constBits()))
    expected = b"".join(expected_frames)
    del image, expected_frames
    child = Process(partial=4093)
    def popen(arguments, **kwargs):
        assert arguments[arguments.index("-pixel_format") + 1] == "bgra"
        return child
    export.subprocess.Popen = popen
    report = export._encode_attempt(
        "ffmpeg", "libx264", scene, renderer, Path("unused.wav"), Path("unused.mp4"),
        count, count / scene.settings.fps, lambda *args: None, lambda: False)
    assert child.stdin.data == expected, "queued native image storage changed during partial writes"
    assert set(child.stdin.threads) == {"export-writer"}
    assert child.stdin.closed and child.stderr.closed
    assert report["frames_written"] == count and report["succeeded"]
    assert 1 <= report["frame_queue_peak"] <= 2
    assert report["frame_buffer_peak_bytes"] <= 4 * scene.settings.width * scene.settings.height * 4
    assert report["export_pixel_format"] == "bgra"
    assert report["memory_copy_seconds"] == 0
    stream = renderer.export_frames([0], lambda: False)
    _, retained = next(stream)
    stream.close()
    saved = bytes(retained.constBits())
assert bytes(retained.constBits()) == saved
print(json.dumps({"api": api, "frames": count, "queue_peak": report["frame_queue_peak"]}), flush=True)
'''


ABI_FAILURE_SCRIPT = r'''
import ctypes, json, os, sys
from PySide6.QtGui import QColor
from stavellum.rendering._rhi import Quad, RhiTarget, _OwnedFrame
from stavellum.domain.models import RenderSettings
from stavellum.graphics.qt import prepare_render_app

os.environ.pop("STAVELLUM_RHI_COPY_READBACK", None)
os.environ.pop("STAVELLUM_RHI_RGBA_READBACK", None)
prepare_render_app(RenderSettings(render_backend="gpu"))
api = sys.argv[1]
target = RhiTarget(65, 49, 0, api)

def command(**changes):
    quad = Quad(0, 0, 0, 65, 49, 0, 0, 1, 1, 1, 1, 1, 1)
    for name, value in changes.items():
        setattr(quad, name, value)
    return (Quad * 1)(quad)

cases = [
    (None, None, 0, "Null RHI handle"),
    (target._handle, None, 1, "Null quad array"),
    (target._handle, command(texture_id=9472), 1, "Frame refers to missing texture"),
    (target._handle, command(w=-1), 1, "Invalid quad geometry"),
    (target._handle, command(x=float("nan")), 1, "Invalid quad geometry"),
    (target._handle, command(a=float("inf")), 1, "Invalid quad attribute"),
    (target._handle, command(), 1_000_001, "Too many frame commands"),
]
for handle, quads, count, error in cases:
    before = target.report()["gpu_submitted_frame_count"]
    # Seed all output fields: a failed ABI call must publish no stale ownership.
    output = _OwnedFrame(9472, 9473, 1234)
    status = target._dll.sprhi_submit_owned(handle, quads, count, ctypes.byref(output))
    assert status != 0
    assert error in target._dll.sprhi_last_error(handle).decode("utf-8")
    assert output.owner is None and output.pixels is None and output.size == 0
    assert target.report()["gpu_submitted_frame_count"] == before
    # Invalid inputs are rejected before beginOffscreenFrame. A subsequent valid
    # frame must work on the same renderer rather than requiring recreation.
    image = target.render([])
    assert image.pixelColor(32, 24) == QColor("black")
    assert target.report()["gpu_submitted_frame_count"] == before + 1
target.close()
print(json.dumps({"api": api, "rejected": len(cases), "frames": len(cases)}), flush=True)
'''


def run_native(script: str, api: str):
    require_vulkan_device()
    root = Path(__file__).resolve().parents[1]
    dll = _rhi.library_path()
    if not dll.exists():
        pytest.skip("Native Vulkan renderer DLL has not been built")
    result = subprocess.run(
        [sys.executable, "-c", script, api], cwd=root, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=90,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.integration
@pytest.mark.parametrize("api", ["vulkan"])
@pytest.mark.skipif(sys.platform != "win32", reason="Windows native renderer required")
def test_native_owned_readback_matches_copy_and_survives_detach_views_gc_and_threads(api):
    reports = run_native(OWNERSHIP_SCRIPT, api)
    assert len(reports) == 4
    assert all(report["graphics_api"] == api for report in reports)


@pytest.mark.integration
@pytest.mark.parametrize("api", ["vulkan"])
@pytest.mark.skipif(sys.platform != "win32", reason="Windows native renderer required")
def test_owned_rhi_export_images_survive_queued_partial_writer_reads(api):
    report = run_native(WRITER_SCRIPT, api)
    assert report["api"] == api and report["frames"] == 12


@pytest.mark.integration
@pytest.mark.parametrize("api", ["vulkan"])
@pytest.mark.skipif(sys.platform != "win32", reason="Windows native renderer required")
def test_owned_rhi_native_failure_clears_output_and_preframe_errors_leave_target_usable(api):
    report = run_native(ABI_FAILURE_SCRIPT, api)
    assert report["api"] == api and report["rejected"] == report["frames"] == 7


class FakeNativeFrames:
    """Hold native-like allocations until the exported release function runs."""

    def __init__(self, storage, released):
        self.storage = storage
        self.released = released
        self.next_owner = 9472

    def allocate(self, size):
        owner = self.next_owner
        self.next_owner += 1
        expected = bytes(index % 256 for index in range(size))
        buffer = (ctypes.c_uint8 * size)(*expected)
        self.storage[owner] = buffer
        return _rhi._OwnedFrame(owner, ctypes.addressof(buffer), size), expected

    def sprhi_release_frame(self, owner):
        assert owner in self.storage, "native frame freed twice or with the wrong owner"
        self.released.append(owner)
        del self.storage[owner]


def fake_owned_target():
    storage, released = {}, []
    dll = FakeNativeFrames(storage, released)
    target = _rhi.RhiTarget.__new__(_rhi.RhiTarget)
    target.width, target.height, target._dll = 3, 2, dll
    return target, dll, storage, released


def test_owned_unit_image_aliases_native_pixels_without_a_hidden_copy():
    target, dll, storage, released = fake_owned_target()
    frame, _ = dll.allocate(3 * 2 * 4)
    image = target._owned_image(frame)
    shallow = QImage(image)
    assert image.pixelColor(0, 0).blue() == 0
    storage[frame.owner][0] = 201
    storage[frame.owner][3] = 255
    # Writing the native allocation must be immediately visible through both
    # Qt wrappers. Pixel equality alone would not detect an unnoticed Qt copy.
    for retained in (image, shallow):
        assert retained.pixelColor(0, 0).blue() == 201
        assert retained.pixelColor(0, 0).alpha() == 255
    assert released == []
    del retained, image, shallow
    gc.collect()
    assert released == [frame.owner] and storage == {}


def test_owned_unit_shallow_images_release_once_and_keep_dll_alive_without_target():
    target, dll, storage, released = fake_owned_target()
    frame, expected = dll.allocate(3 * 2 * 4)
    owner = frame.owner
    image = target._owned_image(frame)
    first, last = QImage(image), QImage(image)
    dll_reference = weakref.ref(dll)
    del image, target, dll
    gc.collect()
    assert dll_reference() is not None and released == [] and owner in storage
    assert bytes(last.constBits()) == expected
    del first
    gc.collect()
    assert released == []
    view = last.constBits()
    assert bytes(view) == expected
    del view, last
    gc.collect()
    assert released == [owner] and storage == {}
    assert dll_reference() is None
    gc.collect()
    assert released == [owner]


def test_owned_unit_mutable_bits_detach_and_release_when_last_native_share_is_gone():
    target, dll, storage, released = fake_owned_target()
    frame, expected = dll.allocate(3 * 2 * 4)
    image = target._owned_image(frame)
    first, last = QImage(image), QImage(image)
    writable = image.bits().cast("B")
    writable[0] ^= 0xff
    changed = bytes(image.constBits())
    assert changed != expected
    assert bytes(first.constBits()) == bytes(last.constBits()) == expected
    del first
    gc.collect()
    assert released == []
    del last
    gc.collect()
    assert released == [frame.owner] and storage == {}
    assert bytes(image.constBits()) == bytes(writable) == changed
    del writable, image
    gc.collect()
    assert released == [frame.owner]


@pytest.mark.parametrize("corruption", ["no_pixels", "too_short", "too_long"])
def test_owned_unit_corrupt_frame_releases_its_owner_once(corruption):
    target, dll, storage, released = fake_owned_target()
    frame, _ = dll.allocate(3 * 2 * 4)
    owner = frame.owner
    if corruption == "no_pixels":
        frame.pixels = None
    elif corruption == "too_short":
        frame.size -= 1
    else:
        frame.size += 4
    with pytest.raises(GpuBackendError, match="Invalid owned RHI frame"):
        target._owned_image(frame)
    assert released == [owner] and storage == {}
    gc.collect()
    assert released == [owner]


def test_owned_unit_empty_frame_does_not_release_a_nonexistent_owner():
    target, _, storage, released = fake_owned_target()
    with pytest.raises(GpuBackendError, match="Invalid owned RHI frame"):
        target._owned_image(_rhi._OwnedFrame())
    gc.collect()
    assert released == [] and storage == {}


@pytest.mark.parametrize("failure", ["exception", "null_image"])
def test_owned_unit_image_constructor_failure_releases_once(monkeypatch, failure):
    target, dll, storage, released = fake_owned_target()
    frame, _ = dll.allocate(3 * 2 * 4)
    owner = frame.owner

    class BrokenQImage:
        Format = QImage.Format

        def __new__(cls, *args):
            if failure == "exception":
                raise RuntimeError("cannot construct image")
            return QImage()

    monkeypatch.setattr(_rhi, "QImage", BrokenQImage)
    error = RuntimeError if failure == "exception" else GpuBackendError
    message = "cannot construct image" if failure == "exception" else "Cannot wrap owned RHI pixels"
    with pytest.raises(error, match=message):
        target._owned_image(frame)
    assert released == [owner] and storage == {}
    gc.collect()
    assert released == [owner]
