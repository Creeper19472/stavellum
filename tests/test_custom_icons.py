"""Portable custom assets, real image output and offline Font Awesome libraries."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import threading
import time
import wave
import zipfile
from importlib.resources import files
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QImage

from stavellum.domain.models import (
    IconAsset,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    RenderSettings,
    TrackInfo,
    load_document,
    save_document,
)
from stavellum.graphics.icons import (
    FONT_AWESOME_VERSION,
    free_catalog,
    icon_thumbnail,
    import_icon,
    make_icon_asset,
    resolve_icon,
    scan_fontawesome,
    whiten_fontawesome,
)
from stavellum.graphics.qt import ensure_app
from stavellum.presentation.scene import compile_scene
from stavellum.rendering.render import FrameRenderer
from stavellum.ui.icon_picker import LIBRARIES_KEY, IconListModel, IconPicker

SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50"><rect width="100" height="50" fill="#f02080" fill-opacity="0.5"/></svg>'
PRO_SVG = b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 50">
<!-- Synthetic test fixture, no third-party Pro artwork. -->
<path class="fa-secondary" fill="currentColor" d="M0 0h50v50H0z"/>
<path class="fa-primary" fill="currentColor" d="M50 0h50v50H50z"/>
</svg>'''


@pytest.fixture(autouse=True)
def app():
    return ensure_app()


def document_with_asset(reference, asset):
    return ProjectDocument(
        ProjectIR("fixture.mid", "midi", "Icons", tracks=[TrackInfo("v", "Violin")],
                  notes=[NoteEvent("n", "v", 0, 480, 72)], duration_ticks=960),
        [PartMapping("v", "Violin", ["v"], instrument="violin", icon=reference, use_icon=True)],
        settings=RenderSettings(width=640, height=360, render_backend="cpu", fps=24),
        icon_assets={reference[6:]: asset},
    )


def test_svg_color_transparency_and_proportional_rendering():
    reference, asset = make_icon_asset(SVG, "pink.svg", "image/svg+xml")
    image = icon_thumbnail(reference, {reference[6:]: asset}, 104)
    assert image.pixelColor(52, 52).getRgb() == pytest.approx((240, 32, 128, 128), abs=1)
    assert image.pixelColor(52, 10).alpha() == 0
    assert image.pixelColor(52, 90).alpha() == 0
    assert resolve_icon(reference, {reference[6:]: asset}) is resolve_icon(reference, {reference[6:]: asset})


@pytest.mark.parametrize(("suffix", "media_type"), [("png", "image/png"), ("jpg", "image/jpeg"), ("webp", "image/webp")])
def test_bitmap_import_survives_deleted_source_and_preserves_aspect(tmp_path, suffix, media_type):
    source = QImage(100, 50, QImage.Format.Format_RGBA8888)
    source.fill(QColor(240, 32, 128, 128 if suffix != "jpg" else 255))
    path = tmp_path / f"图标.{suffix}"
    assert source.save(str(path))
    reference, asset = import_icon(path)
    path.unlink()
    image = icon_thumbnail(reference, {reference[6:]: asset}, 104)
    assert asset.media_type == media_type
    center = image.pixelColor(52, 52)
    assert center.red() == pytest.approx(240, abs=2)
    assert center.green() == pytest.approx(32, abs=2)
    assert center.blue() == pytest.approx(128, abs=2)
    assert center.alpha() == (255 if suffix == "jpg" else 128)
    assert image.pixelColor(52, 10).alpha() == 0


def test_embedded_assets_roundtrip_move_and_deduplicate(tmp_path):
    source = tmp_path / "icon.svg"
    source.write_bytes(SVG)
    reference, asset = import_icon(source)
    document = document_with_asset(reference, asset)
    duplicate_ref, duplicate = import_icon(source)
    document.icon_assets.setdefault(duplicate_ref[6:], duplicate)
    assert len(document.icon_assets) == 1
    original = tmp_path / "first" / "icons.stproj"
    save_document(document, original)
    moved = tmp_path / "moved.stproj"
    shutil.move(original, moved)
    source.unlink()
    restored = load_document(moved)
    assert restored.schema_version == 1
    assert restored.mappings == document.mappings
    assert restored.icon_assets == document.icon_assets
    scene = pickle.loads(pickle.dumps(compile_scene(restored)))
    assert scene.parts[0].icon == reference
    assert scene.icon_assets == document.icon_assets
    with FrameRenderer(scene) as renderer:
        assert not renderer.render_frame(0).isNull()


@pytest.mark.parametrize("damage", ["base64", "hash", "missing", "media_type"])
def test_structurally_corrupt_embedded_assets_are_rejected(damage):
    reference, asset = make_icon_asset(SVG, "icon.svg", "image/svg+xml")
    payload = document_with_asset(reference, asset).to_dict()
    record = payload["icon_assets"][reference[6:]]
    if damage == "base64":
        record["data"] = "!not base64!"
    elif damage == "hash":
        record["data"] = base64.b64encode(SVG + b" ").decode()
    elif damage == "missing":
        payload["icon_assets"] = {}
    else:
        record["media_type"] = "text/html"
    with pytest.raises(ValueError, match="图标"):
        ProjectDocument.from_dict(payload)


def test_only_referenced_assets_are_saved_without_mutating_editor_state(tmp_path):
    reference, asset = make_icon_asset(SVG, "icon.svg", "image/svg+xml")
    document = document_with_asset(reference, asset)
    unused_ref, unused_asset = make_icon_asset(PRO_SVG, "unused.svg", "image/svg+xml")
    document.icon_assets[unused_ref[6:]] = unused_asset
    save_document(document, tmp_path / "icons.stproj")
    assert load_document(tmp_path / "icons.stproj").icon_assets == {reference[6:]: asset}
    assert len(document.icon_assets) == 2
    document.mappings[0].icon = "none"
    assert "icon_assets" not in document.to_dict()


def test_legacy_documents_omit_empty_assets_and_keep_auto_and_hide_semantics():
    reference, asset = make_icon_asset(SVG, "icon.svg", "image/svg+xml")
    document = document_with_asset(reference, asset)
    document.icon_assets.clear()
    document.mappings[0].icon = ""
    assert "icon_assets" not in document.to_dict()
    assert ProjectDocument.from_dict(document.to_dict()).icon_assets == {}
    assert compile_scene(document).parts[0].icon == "violin"
    document.mappings[0].icon = "none"
    assert compile_scene(document).parts[0].icon == ""
    document.mappings[0].icon = reference
    document.icon_assets[reference[6:]] = asset
    document.mappings[0].use_icon = False
    scene = compile_scene(document)
    assert scene.parts[0].icon == ""
    assert scene.icon_assets == {}
    restored = ProjectDocument.from_dict(document.to_dict())
    assert restored.mappings[0].icon == reference
    assert not restored.mappings[0].use_icon
    restored.mappings[0].use_icon = True
    scene = compile_scene(restored)
    assert scene.parts[0].icon == reference
    assert scene.icon_assets == {reference[6:]: asset}


@pytest.mark.parametrize("reference", ["fa:solid:missing-icon", "fa:thin:guitar", "fa:solid:../guitar", "asset:missing"])
def test_invalid_explicit_references_report_errors_instead_of_legacy_fallback(reference):
    with pytest.raises(ValueError):
        resolve_icon(reference)


@pytest.mark.parametrize("payload", [b"<svg>invalid", b"not an image",
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><image href="elsewhere.png"/></svg>'])
def test_invalid_or_nonportable_svg_is_rejected(payload):
    with pytest.raises(ValueError):
        make_icon_asset(payload, "bad.svg", "image/svg+xml")


def test_hash_valid_but_undecodable_asset_reports_rendering_error():
    data = b"not a png"
    digest = hashlib.sha256(data).hexdigest()
    asset = IconAsset("bad.png", "image/png", base64.b64encode(data).decode())
    document = document_with_asset(f"asset:{digest}", asset)
    with pytest.raises(ValueError, match="解码"):
        compile_scene(document)


def test_free_archive_contains_complete_offline_styles_and_original_attribution():
    catalog = free_catalog()
    assert FONT_AWESOME_VERSION == "7.3.1"
    assert len(catalog) == 2883
    assert {entry.style for entry in catalog} == {"solid", "regular", "brands"}
    root = files("stavellum").joinpath("assets", "fontawesome")
    index = json.loads(root.joinpath("index.json").read_text(encoding="utf-8"))
    archive_data = root.joinpath("icons.zip").read_bytes()
    assert hashlib.sha256(archive_data).hexdigest() == index["archive_sha256"]
    import io

    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        for entry in index["icons"]:
            data = archive.read(entry["path"])
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            assert b"Font Awesome Free 7.3.1" in data
    for reference in ("fa:solid:guitar", "fa:regular:bell", "fa:brands:github"):
        image = icon_thumbnail(reference)
        visible = [image.pixelColor(x, y) for y in range(48) for x in range(48)
                   if image.pixelColor(x, y).alpha()]
        assert visible and all(c.red() == c.green() == c.blue() == 255 for c in visible)


def synthetic_library(tmp_path):
    library = tmp_path / "Pro 测试"
    for relative, data in (("svgs/solid/guitar.svg", SVG), ("svgs-full/solid/guitar.svg", PRO_SVG),
                           ("svgs-full/sharp-solid/guitar.svg", PRO_SVG),
                           ("svgs-full/duotone/solid/music.svg", PRO_SVG)):
        target = library / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return library


@pytest.mark.parametrize("subdirectory", ["", "svgs", "svgs-full"])
def test_external_library_supports_pro_sharp_duotone_and_prefers_full(tmp_path, subdirectory):
    library = synthetic_library(tmp_path)
    entries = scan_fontawesome(str(library / subdirectory))
    assert {(entry.style, entry.name) for entry in entries} == {
        ("solid", "guitar"), ("sharp-solid", "guitar"), ("duotone-solid", "music")}
    entry = next(item for item in entries if item.style == "solid")
    assert "svgs-full" in entry.path
    reference, asset = entry.asset()
    shutil.rmtree(library)
    image = icon_thumbnail(reference, {reference[6:]: asset}, 104)
    assert image.pixelColor(25, 52).getRgb() == (255, 255, 255, 102)
    assert image.pixelColor(75, 52).getRgb() == (255, 255, 255, 255)
    assert b"Synthetic test fixture" in asset.decoded()


def test_duotone_inline_styles_preserve_opacity_and_transparent_cutouts():
    svg = PRO_SVG.replace(b'class="fa-secondary" fill="currentColor"',
                          b'class="fa-secondary" style="fill:currentColor;opacity:var(--fa-secondary-opacity,.4)"')
    reference, asset = make_icon_asset(whiten_fontawesome(svg), "music", "image/svg+xml")
    image = icon_thumbnail(reference, {reference[6:]: asset}, 104)
    assert image.pixelColor(25, 52).alpha() == 102


def test_scan_can_be_cancelled_before_decoding_or_reading_svg(tmp_path):
    library = synthetic_library(tmp_path)
    assert scan_fontawesome(str(library), lambda: True) == []
    with pytest.raises(ValueError, match="不存在"):
        scan_fontawesome(str(tmp_path / "missing"))


def test_library_model_only_decodes_requested_thumbnail(tmp_path):
    model = IconListModel()
    model.set_entries(scan_fontawesome(str(synthetic_library(tmp_path))))
    assert not model.thumbnails
    assert model.data(model.index(0), Qt.ItemDataRole.DisplayRole)
    assert not model.thumbnails
    assert not model.data(model.index(0), Qt.ItemDataRole.DecorationRole).isNull()
    assert len(model.thumbnails) == 1


def test_showing_whole_library_only_decodes_visible_items(tmp_path, monkeypatch):
    from stavellum.graphics.icons import IconEntry

    calls = []
    original = IconEntry.asset
    def counted(entry):
        calls.append(entry)
        return original(entry)
    monkeypatch.setattr(IconEntry, "asset", counted)
    picker = IconPicker(QSettings(str(tmp_path / "icons.ini"), QSettings.Format.IniFormat))
    picker.tabs.setCurrentIndex(2)
    picker.show()
    ensure_app().processEvents()
    assert len(picker.model.entries) == 2883
    assert 1 <= len(calls) < 100
    picker.reject()
    picker.deleteLater()


def test_cancelled_scan_survives_dialog_closing_without_changing_selection(tmp_path, monkeypatch):
    from stavellum.ui import icon_picker

    started = threading.Event()
    def slow_scan(directory, cancelled):
        started.set()
        while not cancelled():
            time.sleep(.001)
        return []
    monkeypatch.setattr(icon_picker, "scan_fontawesome", slow_scan)
    settings = QSettings(str(tmp_path / "icons.ini"), QSettings.Format.IniFormat)
    settings.setValue(LIBRARIES_KEY, [str(tmp_path)])
    picker = IconPicker(settings)
    worker = picker._thread
    assert started.wait(2)
    picker.reject()
    assert picker.selection is None and picker._thread is None
    assert worker.wait(2000)
    ensure_app().processEvents()
    assert worker not in icon_picker._SCANS
    picker.deleteLater()


def test_picker_search_filters_and_cancel_preserve_selection(tmp_path):
    settings = QSettings(str(tmp_path / "icons.ini"), QSettings.Format.IniFormat)
    picker = IconPicker(settings)
    picker.tabs.setCurrentIndex(2)
    picker.search.setText("guitar")
    assert picker.model.entries and all("guitar" in entry.name for entry in picker.model.entries)
    picker.style.setCurrentIndex(picker.style.findData("solid"))
    assert all(entry.style == "solid" for entry in picker.model.entries)
    picker.view.setCurrentIndex(picker.model.index(0))
    picker._choose()
    reference, asset = picker.selection
    assert reference.startswith("asset:") and asset.source.startswith("fontawesome:")
    previous = copy.deepcopy(picker.selection)
    picker.reject()
    assert picker.selection == previous
    cancelled = IconPicker(settings)
    cancelled.reject()
    assert cancelled.selection is None
    picker.deleteLater()
    cancelled.deleteLater()


def test_picker_remembers_directories_and_reports_missing_libraries(tmp_path):
    settings = QSettings(str(tmp_path / "icons.ini"), QSettings.Format.IniFormat)
    missing = str(tmp_path / "missing")
    settings.setValue(LIBRARIES_KEY, [missing])
    picker = IconPicker(settings)
    worker = picker._thread
    assert worker.wait(5000)
    ensure_app().processEvents()
    assert picker.directories == [missing]
    assert "不存在" in picker.status.text()
    picker.source.setCurrentIndex(picker.source.findData(missing))
    picker._remove_library()
    assert settings.value(LIBRARIES_KEY) == []
    picker.reject()
    picker.deleteLater()


def test_embedded_icon_crosses_real_spawned_preview_worker():
    from stavellum.ui.background import BackgroundJob

    reference, asset = make_icon_asset(SVG, "pink.svg", "image/svg+xml")
    document = document_with_asset(reference, asset)
    job = BackgroundJob("compile", document)
    results, failures = [], []
    job.succeeded.connect(results.append)
    job.failed.connect(lambda message, details: failures.append((message, details)))
    try:
        job.start()
        deadline = time.monotonic() + 20
        while job.running and time.monotonic() < deadline:
            ensure_app().processEvents()
            time.sleep(.01)
        assert not job.running and not failures, failures
        assert results[0].icon_assets == document.icon_assets
        with FrameRenderer(results[0]) as renderer:
            assert not renderer.render_frame(0).isNull()
    finally:
        job.shutdown()
        job.deleteLater()


EXPORT_SCRIPT = r'''
import json, subprocess, sys
from pathlib import Path
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage
from stavellum.rendering.gpu import GpuBackendError
from stavellum.domain.models import load_document, RenderSettings
from stavellum.graphics.qt import prepare_render_app
from stavellum.presentation.scene import compile_scene
from stavellum.rendering.render import FrameRenderer
from stavellum.rendering.raster import RasterFrameRenderer
from stavellum.exporting.export import export_video

directory, backend = Path(sys.argv[1]), sys.argv[2]
prepare_render_app(RenderSettings(render_backend=backend))
reports = []
for project in sorted(directory.glob('*.stproj')):
    document = load_document(project)
    document.settings.render_backend = backend
    document.settings.video_encoder = 'libx264'
    document.settings.preset = 'ultrafast'
    scene = compile_scene(document)
    try:
        renderer = FrameRenderer(scene)
    except GpuBackendError as exc:
        print(json.dumps({'skip': str(exc)}))
        raise SystemExit(0)
    with renderer:
        assert renderer.backend == backend
        frame = renderer.render_frame(.25)
        frame.save(str(directory / (project.stem + '-' + backend + '.png')))
        layout = renderer._evaluator.evaluate(.25).layout
        row = layout.rows[scene.parts[0].part_id]
        x, y, w, h = row.indicator_rect
        center_x = x - scene.layout.icon_source_gap * layout.scale - row.icon_size / 2
        rectangle = QRectF(center_x-row.icon_size/2, y+h/2-row.icon_size/2,
                           row.icon_size, row.icon_size).toAlignedRect()
        crop = frame.copy(rectangle)
        with RasterFrameRenderer(scene) as cpu:
            reference = cpu.render_frame(.25).copy(rectangle)
        differences = [abs(a-b) for a,b in zip(bytes(crop.constBits()), bytes(reference.constBits()))]
        assert sum(differences)/len(differences) < 8, (backend, project, max(differences))
        if project.stem in ('svg', 'bitmap'):
            expected = (120,16,64) if project.stem == 'svg' else (0,255,80)
            pixel = frame.pixelColor(round(center_x), round(y+h/2))
            assert all(abs(a-b) <= 4 for a,b in zip(pixel.getRgb()[:3],expected)), pixel.getRgb()
        assert any(crop.pixelColor(cx,cy).value() > 30 for cy in range(crop.height()) for cx in range(crop.width()))
    target = directory / (project.stem + '-' + backend + '.mp4')
    export_video(document, target)
    report = json.loads(target.with_suffix('.render.json').read_text(encoding='utf-8'))
    assert report['render_backend'] == backend, report
    decoded_path = directory / (project.stem + '-' + backend + '.rgba')
    subprocess.run(['ffmpeg','-v','error','-i',str(target),'-ss','0.25','-frames:v','1',
                    '-an','-f','rawvideo','-pix_fmt','rgba',str(decoded_path)], capture_output=True, check=True)
    decoded = decoded_path.read_bytes()
    assert len(decoded) == scene.settings.width * scene.settings.height * 4
    image = QImage(decoded, scene.settings.width, scene.settings.height, QImage.Format.Format_RGBA8888).copy()
    crop = image.copy(rectangle)
    if project.stem in ('svg','bitmap'):
        pixel = image.pixelColor(round(center_x), round(y+h/2))
        assert all(abs(a-b) <= 20 for a,b in zip(pixel.getRgb()[:3],expected)), pixel.getRgb()
    reports.append({'project':project.stem,'backend':report['render_backend'], 'frames':report['frames']})
print(json.dumps({'reports':reports}))
'''


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_custom_and_fontawesome_icons_in_real_cpu_vulkan_and_mp4(tmp_path, backend):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg required")
    if backend == "gpu":
        from stavellum.rendering._rhi import library_path

        if sys.platform != "win32" or not library_path().exists():
            pytest.skip("Windows Vulkan renderer required")
    source = tmp_path / "source.png"
    image = QImage(100, 50, QImage.Format.Format_RGBA8888)
    image.fill(QColor(0, 255, 80))
    assert image.save(str(source))
    entries = {
        "svg": make_icon_asset(SVG, "pink.svg", "image/svg+xml"),
        "bitmap": import_icon(source),
        "fontawesome": next(entry for entry in free_catalog() if entry.style == "solid" and entry.name == "guitar").asset(),
    }
    with wave.open(str(tmp_path / "audio.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * 8000)
    for name, (reference, asset) in entries.items():
        document = document_with_asset(reference, asset)
        document.audio_path = str(tmp_path / "audio.wav")
        save_document(document, tmp_path / f"{name}.stproj")
    source.unlink()
    result = subprocess.run([sys.executable, "-c", EXPORT_SCRIPT, str(tmp_path), backend],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    if "skip" in report:
        pytest.skip(report["skip"])
    assert len(report["reports"]) == 3
    assert all(entry["backend"] == backend and entry["frames"] == 24 for entry in report["reports"])
