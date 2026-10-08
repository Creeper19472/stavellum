"""New settings and legacy gate migration preserve source contracts."""

from copy import deepcopy

import pytest
from test_models import project_document

from stavellum.domain.models import ProjectDocument, load_document, save_document


def test_acceleration_and_recognition_settings_roundtrip(tmp_path):
    document = project_document()
    document.settings.render_backend = "gpu"
    document.settings.video_encoder = "h264_nvenc"
    document.settings.nvenc_cq = 23
    document.settings.nvenc_preset = "p7"
    document.mappings[0].auto_staccato = False
    document.mappings[0].auto_grace = False
    document.project.notes[0].key_release_tick = 100
    destination = tmp_path / "settings.stproj"
    save_document(document, destination)
    restored = load_document(destination)
    assert restored.settings == document.settings
    assert restored.mappings == document.mappings
    assert restored.project.notes == document.project.notes
    restored.validate()


@pytest.mark.parametrize(("field", "value"), [
    ("render_backend", "cuda"), ("video_encoder", "h264_amf"),
    ("nvenc_cq", 0), ("nvenc_cq", 52), ("nvenc_preset", "medium"),
])
def test_invalid_acceleration_settings_are_rejected(field, value):
    document = project_document()
    setattr(document.settings, field, value)
    with pytest.raises(ValueError):
        document.validate()


@pytest.mark.parametrize("release", [6, 511])
def test_gate_must_stay_within_source_note(release):
    document = project_document()
    document.project.notes[0].key_release_tick = release
    with pytest.raises(ValueError, match="释放"):
        document.validate()


def legacy_payload():
    value = project_document().to_dict()
    for field in ("render_backend", "video_encoder", "nvenc_cq", "nvenc_preset"):
        value["settings"].pop(field)
    for field in ("auto_staccato", "auto_grace"):
        value["mappings"][0].pop(field)
    for event in value["project"]["notes"]:
        event.pop("key_release_tick")
    return value


def test_legacy_flp_gate_migration_is_only_for_missing_fields():
    payload = legacy_payload()
    payload["project"]["notes"][1]["key_release_tick"] = None
    before = deepcopy(payload)
    document = ProjectDocument.from_dict(payload)
    assert payload == before
    assert document.project.notes[0].key_release_tick == 510
    assert document.project.notes[1].key_release_tick is None
    assert document.settings.render_backend == document.settings.video_encoder == "auto"
    assert document.mappings[0].auto_grace and document.mappings[0].auto_staccato
    assert any(d.code == "legacy-flp-gate" for d in document.project.diagnostics)
    # Once saved, an explicit unknown gate must remain unknown on another load.
    restored = ProjectDocument.from_dict(document.to_dict())
    assert restored.project.notes == document.project.notes
    assert restored.project.diagnostics == document.project.diagnostics


@pytest.mark.parametrize("source_type", ["midi", "flp"])
def test_unknown_legacy_gate_is_not_guessed(source_type):
    payload = legacy_payload()
    payload["project"]["source_type"] = source_type
    if source_type == "flp":
        payload["project"]["diagnostics"].append({
            "severity": "warning", "code": "step_trigger_duration", "message": "unknown gate",
        })
    document = ProjectDocument.from_dict(payload)
    assert all(event.key_release_tick is None for event in document.project.notes)
