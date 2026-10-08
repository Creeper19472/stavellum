"""The Vulkan compositor must preserve geometry, image ownership and clocks."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from native_frames import frame_layout
from native_support import require_vulkan_device
from PySide6.QtGui import QColor, QImage
from test_render import colored_activity_document, rendered_document

from stavellum._rhi import RhiTarget, library_path
from stavellum.rhi import RhiFrameRenderer, clipped_quad
from stavellum.scene import compile_scene


def test_fractional_clipping_adjusts_uv_without_rounding_or_bleed_loss():
    quad = clipped_quad(7, (10.25, 20.5, 100, 50), (20.75, 25.25, 40.5, 20.5))
    assert (quad.x, quad.y, quad.w, quad.h) == pytest.approx((20.75, 25.25, 40.5, 20.5))
    assert (quad.u0, quad.v0, quad.u1, quad.v1) == pytest.approx((.105, .095, .51, .505))
    assert clipped_quad(7, (0, 0, 3, 2), (3, 0, 1, 2)) is None
    assert clipped_quad(7, (0, 0, 0, 2), (0, 0, 1, 2)) is None


def test_removed_graphics_api_cannot_be_selected(rhi_scene):
    with pytest.raises(ValueError, match="vulkan"):
        RhiTarget(65, 49, 0, "opengl")
    with pytest.raises(ValueError, match="vulkan"):
        RhiFrameRenderer(rhi_scene, "opengl")


class FakeTarget:
    """Only the native boundary is replaced; real assets/layout remain exercised."""

    def __init__(self, width, height, cache_megabytes, api):
        self.width, self.height, self.api = width, height, api
        self.submitted = []
        self.fail = False
        self.closed = False

    def texture(self, image):
        return image.cacheKey()

    def render(self, commands):
        if self.fail:
            raise RuntimeError("device lost")
        self.submitted.append(commands)
        image = QImage(self.width, self.height, QImage.Format.Format_ARGB32)
        image.fill(QColor("red"))
        return image

    def report(self):
        return {"graphics_api": self.api, "gpu_info": {"renderer": "test", "msaa_samples": 4}}

    def close(self):
        self.closed = True


@pytest.fixture
def rhi_scene():
    return compile_scene(rendered_document())


@pytest.fixture
def fake_native(monkeypatch):
    monkeypatch.setattr("stavellum.rhi.RhiTarget", FakeTarget)


def test_assets_are_reused_without_mutating_scene_or_rasterizing_output(rhi_scene, fake_native, monkeypatch):
    settings = replace(rhi_scene.settings)
    with RhiFrameRenderer(rhi_scene) as renderer:
        def forbidden(*args):
            raise AssertionError("A full CPU output frame must never be painted")
        monkeypatch.setattr(renderer._assets, "render_frame", forbidden)
        monkeypatch.setattr(renderer._assets, "paint_frame", forbidden)
        first = renderer.commands(1)
        second = renderer.commands(1)
        assert [(q.texture_id, q.x, q.y, q.w, q.h) for q in first] == [
            (q.texture_id, q.x, q.y, q.w, q.h) for q in second]
        assert renderer._metadata is not None and renderer._metadata.height() < settings.height
        assert any(q.texture_id != 0 for q in first)
        assert renderer._assets.cache_peak_bytes <= renderer._assets.cache_limit
        frame = renderer.render_frame(1)
        assert frame.format() == QImage.Format.Format_RGBA8888
    assert rhi_scene.settings == settings


def test_logo_texture_geometry_opacity_order_and_reuse(rhi_scene, fake_native):
    settings = replace(rhi_scene.settings, logo_enabled=True, logo_display_mode="intro")
    scene = replace(rhi_scene, settings=settings)
    with RhiFrameRenderer(scene) as renderer:
        assert renderer._assets._logo is None
        renderer.commands(0)
        assert renderer._assets._logo is None
        first = renderer.commands(1)
        image, rect, opacity = renderer._assets.logo_overlay(1)
        key = image.cacheKey()
        quad = first[-1]
        assert quad.texture_id == key
        assert (quad.x, quad.y, quad.w, quad.h) == pytest.approx(rect.getRect())
        assert (quad.r, quad.g, quad.b, quad.a) == pytest.approx((opacity,) * 4)
        half = renderer.commands(6.4)[-1]
        assert half.texture_id == key and half.a == pytest.approx(0.4)
        assert all(q.texture_id != key for q in renderer.commands(6.8))
        assert renderer.commands(1)[-1].texture_id == key
    assert renderer._assets._logo is None


def test_stream_order_cancel_early_close_restart_and_owned_images(rhi_scene, fake_native):
    renderer = RhiFrameRenderer(rhi_scene)
    stream = renderer.export_frames([2, 0, 1], lambda: False)
    assert stream.pixel_format == "bgra"
    index, retained = next(stream)
    saved = bytes(retained.constBits())
    assert index == 0
    with pytest.raises(RuntimeError, match="one export"):
        renderer.export_frames([0], lambda: False)
    assert [i for i, _ in stream] == [1, 2]
    assert renderer._stream is None
    empty = renderer.export_frames([], lambda: False)
    assert list(empty) == []
    cancelled = renderer.export_frames([0], lambda: True)
    count = renderer.frame_count
    with pytest.raises(InterruptedError):
        next(cancelled)
    cancelled.close()
    assert renderer.frame_count == count and renderer._stream is None
    other = renderer.export_frames([0, 1], lambda: False)
    assert next(other)[0] == 0
    other.close()
    other.close()
    renderer.close()
    renderer.close()
    assert bytes(retained.constBits()) == saved
    assert stream.report()["rendered_frame_count"] == 4
    with pytest.raises(RuntimeError, match="closed"):
        renderer.render_frame(0)


def test_explicit_native_failure_aborts_without_api_fallback(rhi_scene, fake_native):
    with RhiFrameRenderer(rhi_scene) as renderer:
        renderer._target.fail = True
        stream = renderer.export_frames([0], lambda: False)
        with pytest.raises(RuntimeError, match="device lost"):
            next(stream)
        assert renderer._stream is None and renderer.frame_count == 0
        assert renderer.backend_report()["render_fallback_reasons"] == []


@pytest.mark.parametrize("velocity", [127, 40])
def test_colored_lamp_commands_flash_and_fill_entire_rect_without_borders(velocity, fake_native):
    scene = compile_scene(colored_activity_document(velocity=velocity, intro_delay=2))

    def lamp_solids(commands, time, part_id):
        x, y, width, height = frame_layout(scene, time).rows[part_id].indicator_rect
        return [quad for quad in commands
                if quad.texture_id == 0
                and x - 1e-4 <= quad.x and y - 1e-4 <= quad.y
                and quad.x + quad.w <= x + width + 1e-4
                and quad.y + quad.h <= y + height + 1e-4]

    def rgb(quad):
        return tuple(channel / quad.a for channel in (quad.r, quad.g, quad.b))

    def signature(commands):
        return [tuple(getattr(quad, field) for field, _ in quad._fields_) for quad in commands]

    with RhiFrameRenderer(scene) as renderer:
        times = (0, 1.999, 2, 2.05, 2.125, 2.25, 2.31, 2.371, 2.5, 2.75, 2.81, 2.9)
        expected = {time: signature(renderer.commands(time)) for time in times}
        active = {}
        independent = {}
        for time in times:
            commands = renderer.commands(time)
            quads = lamp_solids(commands, time, "part")
            if time < 2 or time in (2.371, 2.9):
                assert len(quads) == 4
                assert all(quad.r == quad.g == quad.b for quad in quads)
                assert all(quad.r > .8 for quad in quads)  # Resting lamps recover the light border.
            else:
                assert len(quads) == 1
                level = velocity / 127 * (.5 if time in (2.31, 2.81) else 1)
                quad = quads[0]
                row = frame_layout(scene, time).rows["part"]
                assert (quad.x, quad.y, quad.w, quad.h) == pytest.approx(row.indicator_rect)
                opacity = row.opacity
                assert quad.a == pytest.approx(level * opacity, abs=.5 / 255 + 2e-7)
                assert quad.b > quad.g > quad.r  # The regular blue source covers the old border too.
                active[time] = quad
            if time >= 2:
                solo = lamp_solids(commands, time, "solo")
                assert len(solo) == 1
                assert (solo[0].x, solo[0].y, solo[0].w, solo[0].h) == pytest.approx(
                    frame_layout(scene, time).rows["solo"].indicator_rect)
                assert solo[0].r > solo[0].b > solo[0].g
                assert solo[0].a == pytest.approx(1)
                independent[time] = solo[0]
        peak, halfway, hold, released = [rgb(active[time]) for time in (2, 2.05, 2.125, 2.31)]
        assert max(peak) > max(halfway) > max(hold)
        assert 1 - min(peak) / max(peak) < 1 - min(hold) / max(hold)
        assert released == pytest.approx(hold, abs=2e-7)
        assert active[2.31].a == pytest.approx(active[2.125].a / 2, abs=.5 / 255 + 2e-7)
        assert rgb(active[2.5]) == pytest.approx(peak, abs=2e-7)  # A fresh note flashes again.
        assert max(rgb(independent[2])) > max(rgb(independent[2.125]))
        for time in reversed(times):
            assert signature(renderer.commands(time)) == expected[time]


REAL_SCRIPT = r'''
import json, os, sys, threading
sys.path.insert(0, "tests")
from native_frames import frame_layout
from dataclasses import replace
sys.path.insert(0, "tests")
from test_render import colored_activity_document, lamp_interior, lamp_rectangle, rendered_document
from stavellum.models import RenderSettings
from stavellum.qt import prepare_render_app
from stavellum._rhi import RhiTarget, Quad
from stavellum.rhi import RhiFrameRenderer
from stavellum.render import FrameRenderer
from stavellum.scene import compile_scene
from PySide6.QtGui import QImage, QColor
prepare_render_app(RenderSettings(render_backend="gpu"))
os.environ.pop("STAVELLUM_RHI_COPY_READBACK", None)
reports = []
def pixels(renderer,time):
    image=renderer.render_frame(time)
    return bytes(image.constBits())
for api in ("vulkan",):
    default_pixels = None
    for force_rgba in (False, True):
        if force_rgba:
            os.environ["STAVELLUM_RHI_RGBA_READBACK"] = "1"
        else:
            os.environ.pop("STAVELLUM_RHI_RGBA_READBACK", None)
        target = RhiTarget(65, 49, 0, api)
        texture = QImage(2, 2, QImage.Format.Format_RGBA8888)
        texture.setPixelColor(0, 0, QColor("red")); texture.setPixelColor(1, 0, QColor(19,83,201))
        texture.setPixelColor(0, 1, QColor("blue")); texture.setPixelColor(1, 1, QColor("white"))
        outputs = []
        for _ in range(3):
            ident = target.texture(texture)
            image = target.render([Quad(ident, 0, 0, 65, 49, 0, 0, 1, 1, 1, 1, 1, 1)])
            assert image.pixelColor(2,2) == QColor("red")
            assert image.pixelColor(2,45) == QColor("blue")
            assert image.pixelColor(64,2) == QColor(19,83,201)  # Last-column pixel
            assert image.pixelColor(64,48) == QColor("white")  # SSE2 scalar remainder in RGBA mode
            assert not target._textures  # zero-budget eviction happens after completion
            outputs.append(bytes(image.constBits()))
        image = target.render([
            Quad(0, 7, 11, 20, 15, 0,0,1,1, .5,0,0,.5),
            Quad(0, 17, 17, 19, 18, 0,0,1,1, 0,0,.25,.25),
        ])
        assert 127 <= image.pixelColor(15,15).red() <= 128
        assert image.pixelColor(15,15).alpha() == 255
        assert 95 <= image.pixelColor(20,20).red() <= 96
        assert 63 <= image.pixelColor(20,20).blue() <= 64
        assert image.pixelColor(20,20).alpha() == 255
        assert image.pixelColor(15,35) == QColor("black")
        saved = bytes(image.constBits())
        outputs.append(saved)
        if default_pixels is None:
            default_pixels = outputs
        else:
            assert outputs == default_pixels, (api, "RGBA and BGRA readback differ")
        report = target.report()
        assert report["readback_format"] == ("RGBA8" if force_rgba else "BGRA8")
        assert report["readback_copy_path"] == ("rgba-inplace-sse2-swizzle" if force_rgba else "bgra-owned-buffer")
        reports.append(report)
        wrong_thread=[]
        def wrong():
            try: target.render([])
            except Exception as e: wrong_thread.append(str(e))
        thread=threading.Thread(target=wrong); thread.start(); thread.join()
        assert wrong_thread
        target.close(); target.close()
        assert bytes(image.constBits()) == saved
    os.environ.pop("STAVELLUM_RHI_RGBA_READBACK", None)
    document=rendered_document(piano=True)
    document.settings.render_backend="gpu"
    scene=compile_scene(document)
    scene=replace(scene, settings=replace(scene.settings, cache_megabytes=0))
    with RhiFrameRenderer(scene,api) as renderer:
        samples=[0,.125,.5,1,4,9.5]
        expected={time:pixels(renderer,time) for time in samples}
        for time in [9.5,.5,4,0,.125,1,.5]:
            assert pixels(renderer,time)==expected[time]
        retained=renderer.render_frame(.5)
    assert bytes(retained.constBits())==expected[.5]
    for velocity in (127,40):
        document=colored_activity_document(velocity=velocity,intro_delay=2)
        scene=compile_scene(document)
        samples=[0,1.999,2,2.05,2.125,2.25,2.31,2.371,2.5,2.81,2.9]
        with RhiFrameRenderer(scene,api) as renderer, FrameRenderer(scene) as cpu:
            expected={time:pixels(renderer,time) for time in samples}
            for time in samples:
                frame=renderer.render_frame(time)
                reference=cpu.render_frame(time)
                for part_id,row in frame_layout(scene, time).rows.items():
                    lit=time>=2 and (part_id=="solo" or time not in (2.371,2.9))
                    crop=lamp_rectangle if lit else lamp_interior
                    actual_image=crop(frame,row.indicator_rect)
                    wanted_image=crop(reference,row.indicator_rect)
                    actual=bytes(actual_image.constBits())
                    wanted=bytes(wanted_image.constBits())
                    assert len(actual)==len(wanted)
                    delta=max(abs(a-b) for a,b in zip(actual,wanted))
                    assert delta<=2,(api,velocity,time,part_id,delta)
            for time in reversed(samples):
                assert pixels(renderer,time)==expected[time]
    document=rendered_document()
    document.settings.logo_enabled=True
    document.settings.logo_size_ratio=.2
    scene=compile_scene(document)
    for mode in ("persistent","fade_in","intro"):
        timed=replace(scene,settings=replace(scene.settings,logo_display_mode=mode))
        with RhiFrameRenderer(timed,api) as renderer, FrameRenderer(timed) as cpu:
            samples=[0,.5,1,6.4,6.8]
            expected={t:pixels(renderer,t) for t in samples}
            for t in samples:
                actual_frame=renderer.render_frame(t)
                reference_frame=cpu.render_frame(t)
                crop=actual_frame.rect().adjusted(540,260,-5,-5)
                actual_image=actual_frame.copy(crop)
                wanted_image=reference_frame.copy(crop)
                actual=bytes(actual_image.constBits())
                wanted=bytes(wanted_image.constBits())
                differences=[abs(a-b) for a,b in zip(actual,wanted)]
                assert sum(differences)/len(differences)<=2,(mode,t,"mean",sum(differences)/len(differences))
                assert max(differences)<=32,(mode,t,"max",max(differences))
            for t in reversed(samples):
                assert pixels(renderer,t)==expected[t]
            with FrameRenderer(replace(timed,settings=replace(timed.settings,render_backend="gpu"))) as exported:
                actual=[]
                for _,image in exported.export_frames([1,0,6.8],lambda:False):
                    converted=image.convertToFormat(QImage.Format.Format_RGBA8888)
                    actual.append(bytes(converted.constBits()))
                assert actual==[expected[1],expected[0],expected[6.8]]
print(json.dumps(reports),flush=True)
'''


@pytest.mark.integration
@pytest.mark.skipif(sys.platform != "win32", reason="Windows native renderer required")
def test_real_rhi_channels_blend_direction_eviction_random_access_and_ownership():
    root = Path(__file__).resolve().parents[1]
    dll = library_path()
    if not dll.exists():
        pytest.skip("Native Vulkan renderer DLL has not been built")
    require_vulkan_device()
    result = subprocess.run([sys.executable, "-c", REAL_SCRIPT], cwd=root, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=90,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert result.returncode == 0, result.stdout + result.stderr
    reports = json.loads(result.stdout.strip().splitlines()[-1])
    assert [report["graphics_api"] for report in reports] == ["vulkan", "vulkan"]
    assert all(report["gpu_info"]["msaa_samples"] == 4 for report in reports)
