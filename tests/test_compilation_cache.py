"""Cache reuse must skip work while preserving current documents and exact frames."""

from __future__ import annotations

import copy
import json
import math
import shutil
import subprocess
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from importlib.resources import files

import pytest
from test_render import pixels
from test_scene import score_document

from stavellum.domain.models import Diagnostic, NoteEvent, VolumeRoute, load_document, save_document
from stavellum.engraving import notation
from stavellum.graphics.qt import ensure_app
from stavellum.presentation import compilation_cache as cache_module
from stavellum.presentation import scene
from stavellum.presentation.compilation_cache import CompilationCache, geometry_key, timeline_key
from stavellum.rendering.render import FrameRenderer
from stavellum.ui.background import BackgroundJob


@pytest.fixture
def document():
    document = score_document(bars=2)
    document.settings.width, document.settings.height = 640, 360
    document.settings.render_backend = "cpu"
    ensure_app()
    return document


def hits(compiled):
    report = compiled.compilation_report
    return report["geometry_cache_hit"], report["timeline_cache_hit"]


def test_extracted_curve_source_change_invalidates_both_cache_layers(document, tmp_path, monkeypatch):
    package = tmp_path / "package"
    shutil.copytree(files("stavellum"), package, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(cache_module, "files", lambda name: package)
    cache_module.engine_fingerprint.cache_clear()
    try:
        assert hits(scene.compile_scene(document)) == (False, False)
        assert hits(scene.compile_scene(document)) == (True, True)
        curves = package / "presentation/curves.py"
        curves.write_bytes(curves.read_bytes() + b"\n# changed curve implementation\n")
        cache_module.engine_fingerprint.cache_clear()
        assert hits(scene.compile_scene(document)) == (False, False)
        assert hits(scene.compile_scene(document)) == (True, True)
    finally:
        cache_module.engine_fingerprint.cache_clear()


def test_warm_cache_skips_both_compilers_and_refreshes_display_data(document, monkeypatch):
    original = document.to_dict()
    cold = scene.compile_scene(document)
    assert hits(cold) == (False, False)
    assert document.to_dict() == original

    def forbidden(*args, **kwargs):
        pytest.fail("a warm cache must not compile")

    monkeypatch.setattr(notation, "build_notation", forbidden)
    monkeypatch.setattr(scene, "compile_visibility", forbidden)
    document.metadata.title = "新标题"
    document.mappings[0].name = "新的声部名称"
    document.mappings[0].icon = "violin"
    document.project.tracks[0].color = "#123456"
    document.project.diagnostics.append(Diagnostic("info", "current-import", "新的导入诊断"))
    warm = scene.compile_scene(document)
    assert hits(warm) == (True, True)
    assert warm.svg == cold.svg
    assert warm.metadata.title == "新标题"
    assert warm.parts[0].name == "新的声部名称"
    assert warm.parts[0].icon == "violin"
    assert warm.parts[0].activity_color == "#123456"
    assert warm.diagnostics[0].code == "current-import"
    assert all(item.message.startswith("新的声部名称：")
               for item in warm.diagnostics if item.track_id == warm.parts[0].part_id)
    assert asdict(warm.layout) == asdict(cold.layout)


@pytest.mark.parametrize("field,value", [
    ("fps", 24), ("crf", 20), ("preset", "fast"), ("video_encoder", "libx264"),
    ("render_backend", "auto"), ("nvenc_cq", 20), ("nvenc_preset", "p4"),
    ("title_x", .1), ("title_y", .8), ("title_font_size", 55),
    ("subtitle_font_size", 30), ("credits_font_size", 30), ("logo_enabled", True),
    ("logo_display_mode", "intro"), ("logo_size_ratio", .1), ("logo_opacity", .5),
    ("logo_enter_seconds", .5), ("logo_hold_seconds", 3), ("logo_exit_seconds", .5),
    ("cache_megabytes", 32),
])
def test_display_and_encoder_settings_reuse_both_layers(document, field, value):
    scene.compile_scene(document)
    setattr(document.settings, field, value)
    updated = scene.compile_scene(document)
    assert hits(updated) == (True, True)
    assert getattr(updated.settings, field) == value


@pytest.mark.parametrize("field,value", [
    ("width", 800), ("height", 480), ("staff_scale", 1.1),
    ("score_left", .18), ("score_right", .9), ("score_top", .25), ("score_bottom", .7),
    ("header_width", .12), ("score_start_in_audio_sec", .25), ("enter_seconds", .5),
    ("exit_seconds", .5), ("reflow_seconds", .5), ("animation_stable_seconds", 1),
    ("intro_delay_seconds", .5), ("overlay_enter_seconds", .5),
    ("overlay_exit_seconds", .5), ("announcement_hold_seconds", 3),
    ("announcement_auto_hide", True),
])
def test_layout_settings_invalidate_only_timeline(document, field, value):
    scene.compile_scene(document)
    setattr(document.settings, field, value)
    assert hits(scene.compile_scene(document)) == (True, False)
    assert hits(scene.compile_scene(document)) == (True, True)


@pytest.mark.parametrize("owner,field,value", [
    ("project", "bpm", 130), ("project", "ppq", 960),
    ("project", "numerator", 3), ("project", "denominator", 8),
    ("project", "duration_ticks", 4800),
    ("mapping", "quantization", 8), ("mapping", "triplets", False),
    ("mapping", "clef", "bass"), ("mapping", "key_signature", 2),
    ("mapping", "transpose", 12), ("mapping", "grand_staff", True),
    ("mapping", "instrument", "piano"), ("mapping", "auto_staccato", False),
    ("mapping", "auto_grace", False), ("mapping", "auto_dynamics", False),
    ("mapping", "auto_simplify_accidentals", False), ("mapping", "auto_ottava", True),
    ("mapping", "keyswitches", [60]), ("mapping", "articulations", {"carrier": "pizz."}),
    ("note", "pitch", 61), ("note", "velocity", 120),
    ("note", "duration_tick", 60), ("note", "key_release_tick", 60),
])
def test_notation_inputs_invalidate_both_layers(document, owner, field, value):
    scene.compile_scene(document)
    target = {"project": document.project, "mapping": document.mappings[0],
              "note": document.project.notes[0]}[owner]
    setattr(target, field, value)
    assert hits(scene.compile_scene(document)) == (False, False)


def test_order_disabled_mappings_and_volume_routes_are_keyed(document):
    original = geometry_key(document)
    document.mappings.reverse()
    assert geometry_key(document) != original
    document.mappings.reverse()
    disabled = copy.deepcopy(document.mappings[0])
    disabled.part_id, disabled.enabled = "disabled", False
    document.mappings.append(disabled)
    before = geometry_key(document)
    disabled.keyswitches = [60]
    assert geometry_key(document) != before
    document.project.volume_routes.append(VolumeRoute("carrier", ["volume"], {"volume": 1.0}))
    assert geometry_key(document) != before


def test_paths_do_not_change_keys_and_invalid_documents_never_use_cache(document, tmp_path):
    scene.compile_scene(document)
    document.audio_path = "different.wav"
    document.project.source_path = "moved.mid"
    save_document(document, tmp_path / "moved.stproj")
    assert hits(scene.compile_scene(load_document(tmp_path / "moved.stproj"))) == (True, True)
    document.project.timing_confirmed = False
    with pytest.raises(ValueError, match="确认"):
        scene.compile_scene(document)
    document.project.timing_confirmed = True
    document.mappings[0].icon = "asset:" + "0" * 64
    with pytest.raises(ValueError):
        scene.compile_scene(document)


@pytest.mark.parametrize("variant", ["simple", "piano", "ottava", "grace", "empty", "offset"])
def test_cached_and_rebuilt_frames_are_identical(document, variant):
    if variant == "piano":
        document.mappings[0].instrument = "piano"
    elif variant == "ottava":
        document.mappings[0].auto_ottava = True
        for note in document.project.notes:
            if note.track_id == "carrier":
                note.pitch += 36
    elif variant == "grace":
        document.project.notes.insert(0, NoteEvent("grace", "carrier", 0, 20, 59, 70))
        document.project.notes[1].start_tick = 30
    elif variant == "empty":
        document.project.notes = [note for note in document.project.notes if note.track_id == "carrier"]
    elif variant == "offset":
        document.settings.score_start_in_audio_sec = .25
        document.settings.intro_delay_seconds = .5
    original = copy.deepcopy(document.to_dict())
    cold = scene.compile_scene(document)
    warm = scene.compile_scene(document)
    forced = scene.compile_scene(document, force_rebuild=True)
    assert hits(warm) == (True, True)
    assert hits(forced) == (False, False)
    assert cold.svg == warm.svg == forced.svg
    assert asdict(cold.layout) == asdict(warm.layout) == asdict(forced.layout)
    with FrameRenderer(cold) as reference, FrameRenderer(warm) as restored, FrameRenderer(forced) as rebuilt:
        for seconds in (0, .1, 2, .5, 4, 0):
            expected = pixels(reference.render_frame(seconds))
            assert pixels(restored.render_frame(seconds)) == expected
            assert pixels(rebuilt.render_frame(seconds)) == expected
    assert document.to_dict() == original


def test_disabled_cache_does_not_read_or_write(document, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled caching must not create the service")

    monkeypatch.setattr(cache_module, "CompilationCache", forbidden)
    assert hits(scene.compile_scene(document, use_cache=False)) == (False, False)


@pytest.mark.parametrize("layer", ["geometry", "timeline"])
def test_corrupt_cache_rebuilds_only_missing_layer(document, layer):
    scene.compile_scene(document)
    service = CompilationCache()
    key = geometry_key(document)
    if layer == "timeline":
        key = timeline_key(key, document.settings)
    service._path(layer, key).write_bytes(b"corrupt")
    assert hits(scene.compile_scene(document)) == ((False, True) if layer == "geometry" else (True, False))
    assert hits(scene.compile_scene(document)) == (True, True)


def test_invalid_structure_and_references_are_misses(document):
    scene.compile_scene(document)
    service = CompilationCache()
    key = geometry_key(document)
    data = service.read("geometry", key, lambda data: data)
    data["parts"][0]["source_top"] = "invalid"
    service.write("geometry", key, data)
    assert hits(scene.compile_scene(document))[0] is False
    data = service.read("geometry", key, lambda data: data)
    data["element_part_ids"]["broken"] = "nonexistent"
    service.write("geometry", key, data)
    assert hits(scene.compile_scene(document))[0] is False
    key = timeline_key(key, document.settings)
    data = service.read("timeline", key, lambda data: data)
    data["layout"]["zoom"]["keys"] = []
    service.write("timeline", key, data)
    assert hits(scene.compile_scene(document)) == (True, False)


def test_version_and_engine_changes_invalidate_cache(document, monkeypatch):
    scene.compile_scene(document)
    service = CompilationCache()
    key = geometry_key(document)
    path = service._path("geometry", key)
    envelope = json.loads(zlib.decompress(path.read_bytes()))
    envelope["version"] += 1
    path.write_bytes(zlib.compress(json.dumps(envelope).encode()))
    assert hits(scene.compile_scene(document))[0] is False
    monkeypatch.setattr(cache_module, "engine_fingerprint", lambda: "changed-engine")
    assert hits(scene.compile_scene(document)) == (False, False)


def test_cache_storage_failure_is_nonfatal(document, tmp_path, monkeypatch):
    unavailable = tmp_path / "not-a-directory"
    unavailable.write_text("occupied")
    monkeypatch.setenv("STAVELLUM_CACHE_DIR", str(unavailable))
    assert hits(scene.compile_scene(document)) == (False, False)
    assert hits(scene.compile_scene(document)) == (False, False)


def test_cancelled_compile_retains_complete_layers(document):
    cancelled = False

    def progress(fraction, message):
        nonlocal cancelled
        if fraction == .6:
            cancelled = True

    with pytest.raises(InterruptedError):
        scene.compile_scene(document, progress=progress, cancel=lambda: cancelled)
    assert hits(scene.compile_scene(document)) == (True, False)
    assert not list(CompilationCache().directory.glob("*.tmp"))


def test_cache_cancellation_before_commit_leaves_old_entry(tmp_path):
    service = CompilationCache(tmp_path)
    key = "a" * 64
    service.write("geometry", key, {"old": True})
    with pytest.raises(InterruptedError):
        service.write("geometry", key, {"new": True}, lambda: True)
    assert service.read("geometry", key, lambda value: value) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_budget_prunes_old_entries_and_oversized_entries_bypass(tmp_path):
    service = CompilationCache(tmp_path, budget_bytes=400)
    service.write("geometry", "a" * 64, {"first": "small"})
    service.write("timeline", "b" * 64, {"second": "small"})
    assert service.read("geometry", "a" * 64, lambda value: value)
    service.write("geometry", "c" * 64, {"third": "small"})
    assert sum(path.stat().st_size for path in tmp_path.glob("*.json.z")) <= 400
    assert service.read("geometry", "a" * 64, lambda value: value)
    service.budget_bytes = 1
    service.write("geometry", "d" * 64, {"oversized": True})
    assert service.read("geometry", "d" * 64, lambda value: value) is None
    (tmp_path / "unrelated.txt").write_text("preserved")
    assert service.clear()
    assert (tmp_path / "unrelated.txt").exists()


def test_concurrent_writes_and_clear_never_expose_partial_data(tmp_path):
    service = CompilationCache(tmp_path)
    key = "a" * 64

    def operation(index):
        if index % 5 == 0:
            service.clear()
        else:
            service.write("geometry", key, {"value": index})
        value = service.read("geometry", key, lambda value: value)
        assert value is None or type(value["value"]) is int

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(operation, range(30)))
    service.write("geometry", key, {"value": 100})
    assert service.read("geometry", key, lambda value: value) == {"value": 100}


def test_infinite_exit_time_has_portable_json_encoding(document, monkeypatch):
    original = scene.compile_visibility

    def infinite_exit(compiled):
        original(compiled)
        compiled.layout.tempo_exit_time = math.inf

    monkeypatch.setattr(scene, "compile_visibility", infinite_exit)
    scene.compile_scene(document)
    warm = scene.compile_scene(document)
    assert hits(warm) == (True, True)
    assert warm.layout.tempo_exit_time == math.inf


def test_spawned_workers_share_cache(document):
    app = ensure_app()
    for expected in ((False, False), (True, True)):
        received, errors = [], []
        job = BackgroundJob("compile", document)
        job.succeeded.connect(received.append)
        job.failed.connect(lambda message, details: errors.append(details))
        job.start()
        deadline = time.monotonic() + 30
        while job.running and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        try:
            assert not job.running and not errors, errors
            assert len(received) == 1 and hits(received[0]) == expected
        finally:
            job.shutdown()


@pytest.mark.integration
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg required")
def test_preview_cache_and_uncached_video_produce_identical_decoded_frames(document, tmp_path, monkeypatch):
    import wave

    from stavellum.exporting import export

    document.project.notes = document.project.notes[:2]
    document.project.duration_ticks = 240
    document.settings.fps = 24
    document.settings.video_encoder = "libx264"
    document.settings.preset = "ultrafast"
    document.audio_path = str(tmp_path / "silence.wav")
    with wave.open(document.audio_path, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        audio.writeframes(b"\0\0" * 24000)
    scene.compile_scene(document)
    cached, baseline = tmp_path / "cached.mp4", tmp_path / "baseline.mp4"
    export.export_video(document, cached)
    summary = json.loads(cached.with_suffix(".render.json").read_text(encoding="utf-8"))
    assert summary["compilation"]["geometry_cache_hit"]
    assert summary["compilation"]["timeline_cache_hit"]
    original = scene.compile_scene
    monkeypatch.setattr(export, "compile_scene", lambda doc, **kwargs: original(doc, use_cache=False, **kwargs))
    export.export_video(document, baseline)

    def decoded(path):
        return subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(path),
                               "-an", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
                              capture_output=True, check=True).stdout

    assert decoded(cached) == decoded(baseline)
