"""Headless options preserve saved choices and initialize renderers in the right order."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from stavellum.cli import main
from stavellum.domain.models import (
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    TrackInfo,
    load_document,
    save_document,
)


@pytest.fixture
def project_file(tmp_path):
    project = ProjectIR("source.mid", "midi", "CLI", tracks=[TrackInfo("a", "Violin"),
                                                           TrackInfo("b", "Cello")],
                        notes=[NoteEvent("n", "a", 0, 480, 60)])
    document = ProjectDocument(project, [
        PartMapping("a", "Violin", ["a"], auto_staccato=False, auto_grace=True,
                    auto_simplify_accidentals=False, auto_ottava=False),
        PartMapping("b", "Cello", ["b"], auto_staccato=True, auto_grace=False, auto_ottava=True),
    ])
    document.settings.render_backend = "cpu"
    document.settings.video_encoder = "libx264"
    document.settings.nvenc_cq = 22
    document.settings.nvenc_preset = "p6"
    target = tmp_path / "input.stproj"
    save_document(document, target)
    return target


@pytest.fixture
def headless_runners(monkeypatch):
    from stavellum.engraving import notation
    from stavellum.exporting import export
    from stavellum.graphics import qt
    from stavellum.presentation import scene
    from stavellum.rendering import render

    state = SimpleNamespace(documents=[], prepared=[], renderers=[], saved=True, fail_frame=False)

    def prepare(settings):
        state.prepared.append(deepcopy(settings))

    def compile_score(document):
        assert state.prepared
        assert state.prepared[-1].render_backend == document.settings.render_backend
        state.documents.append(document)
        return SimpleNamespace(settings=document.settings)

    class Renderer:
        def __init__(self, compiled):
            self.closed = 0
            self.scene = compiled
            state.renderers.append(self)

        def render_frame(self, seconds):
            state.frame_time = seconds
            if state.fail_frame:
                raise RuntimeError("frame failed")
            return SimpleNamespace(save=lambda path: state.saved)

        def close(self):
            self.closed += 1

    def video(document, output, progress):
        assert state.prepared[-1].video_encoder == document.settings.video_encoder
        state.documents.append(document)
        return str(output)

    def parts(document, output, progress):
        state.documents.append(document)
        return []

    monkeypatch.setattr(qt, "prepare_render_app", prepare)
    monkeypatch.setattr(scene, "compile_scene", compile_score)
    monkeypatch.setattr(render, "FrameRenderer", Renderer)
    monkeypatch.setattr(export, "export_video", video)
    monkeypatch.setattr(notation, "export_parts", parts)
    return state


@pytest.mark.parametrize("command", ["frame", "render", "parts"])
@pytest.mark.parametrize("overrides", [[], ["--auto-staccato", "--no-auto-grace", "--no-auto-dynamics"]])
def test_notation_options_preserve_saved_values_or_override_all_parts(
        command, overrides, project_file, headless_runners, tmp_path):
    before = project_file.read_bytes()
    assert main([command, str(project_file), "--output", str(tmp_path / "result"), *overrides]) == 0
    mappings = headless_runners.documents[-1].mappings
    actual = [(mapping.auto_staccato, mapping.auto_grace) for mapping in mappings]
    assert actual == ([(True, False), (True, False)] if overrides else [(False, True), (True, False)])
    assert all(mapping.auto_dynamics == (not bool(overrides)) for mapping in mappings)
    assert project_file.read_bytes() == before
    if command == "frame":
        assert headless_runners.renderers[-1].closed == 1


@pytest.mark.parametrize("command", ["frame", "render", "parts"])
@pytest.mark.parametrize("field_name, flag", [
    ("auto_simplify_accidentals", "auto-simplify-accidentals"),
    ("auto_ottava", "auto-ottava"),
])
@pytest.mark.parametrize("enabled", [None, True, False])
def test_notation_simplification_preserves_or_overrides_saved_choices_without_rewriting_source(
        command, field_name, flag, enabled, project_file, headless_runners, tmp_path):
    before = project_file.read_bytes()
    options = [] if enabled is None else [f"--{'' if enabled else 'no-'}{flag}"]
    expected = [False, True] if enabled is None else [enabled, enabled]
    assert main([command, str(project_file), "--output", str(tmp_path / "result"), *options]) == 0
    assert [getattr(mapping, field_name) for mapping in headless_runners.documents[-1].mappings] == expected
    assert project_file.read_bytes() == before


@pytest.mark.parametrize("field_name, flag", [
    ("auto_simplify_accidentals", "auto-simplify-accidentals"),
    ("auto_ottava", "auto-ottava"),
])
@pytest.mark.parametrize("enabled", [None, True, False])
def test_import_defaults_to_simplification_and_accepts_overrides(
        field_name, flag, enabled, project_file, monkeypatch, tmp_path):
    from stavellum import importers

    source = load_document(project_file).project
    original = deepcopy(source)
    monkeypatch.setattr(importers, "import_project", lambda *args: source)
    target = tmp_path / "imported.stproj"
    options = [] if enabled is None else [f"--{'' if enabled else 'no-'}{flag}"]
    expected = (field_name != "auto_ottava") if enabled is None else enabled
    assert main(["import", "source.mid", "--output", str(target), *options]) == 0
    assert all(getattr(mapping, field_name) == expected for mapping in load_document(target).mappings)
    assert source == original


def test_import_can_disable_all_automatic_recognition_options(
        project_file, headless_runners, monkeypatch, tmp_path):
    from stavellum import importers

    source = load_document(project_file).project
    monkeypatch.setattr(importers, "import_project", lambda *args: deepcopy(source))
    target = tmp_path / "imported.stproj"
    audio_path = tmp_path / "原曲.flac"
    assert main(["import", "source.mid", "--output", str(target),
                 "--audio", str(audio_path),
                 "--no-auto-staccato", "--no-auto-grace", "--no-auto-dynamics",
                 "--no-auto-simplify-accidentals", "--no-auto-ottava"]) == 0
    assert all(not mapping.auto_staccato and not mapping.auto_grace and not mapping.auto_dynamics
               and not mapping.auto_simplify_accidentals and not mapping.auto_ottava
               for mapping in load_document(target).mappings)
    assert load_document(target).audio_path == str(audio_path.resolve())
    assert main(["render", str(target), "--output", str(tmp_path / "result.mp4")]) == 0
    assert headless_runners.documents[-1].audio_path == str(audio_path.resolve())


def test_render_backend_and_nvenc_quality_override_without_changing_cpu_settings(
        project_file, headless_runners, tmp_path):
    assert main(["render", str(project_file), "--output", str(tmp_path / "result.mp4"),
                 "--render-backend", "gpu", "--video-encoder", "h264_nvenc",
                 "--nvenc-cq", "17", "--nvenc-preset", "p7"]) == 0
    settings = headless_runners.documents[-1].settings
    assert (settings.render_backend, settings.video_encoder) == ("gpu", "h264_nvenc")
    assert (settings.nvenc_cq, settings.nvenc_preset) == (17, "p7")
    assert (settings.crf, settings.preset) == (18, "medium")


def test_frame_prepares_selected_backend_before_compiling_and_closes_on_failure(
        project_file, headless_runners, tmp_path, capsys):
    headless_runners.fail_frame = True
    assert main(["frame", str(project_file), "--output", str(tmp_path / "frame.png"),
                 "--render-backend", "gpu", "--time", "1.25"]) == 1
    assert headless_runners.prepared[-1].render_backend == "gpu"
    assert headless_runners.frame_time == 1.25
    assert headless_runners.renderers[-1].closed == 1
    assert "frame failed" in capsys.readouterr().err


def test_frame_save_failure_closes_renderer(project_file, headless_runners, tmp_path, capsys):
    headless_runners.saved = False
    assert main(["frame", str(project_file), "--output", str(tmp_path / "frame.png")]) == 1
    assert headless_runners.renderers[-1].closed == 1
    assert "无法保存 PNG" in capsys.readouterr().err


def test_invalid_nvenc_quality_is_rejected_before_render_initialization(
        project_file, headless_runners, tmp_path, capsys):
    assert main(["render", str(project_file), "--output", str(tmp_path / "result.mp4"),
                 "--nvenc-cq", "0"]) == 1
    assert not headless_runners.prepared and not headless_runners.documents
    assert "CQ" in capsys.readouterr().err
