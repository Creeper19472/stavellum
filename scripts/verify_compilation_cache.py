"""Measure cold/warm compilation, first-frame drawing and real spawned project opens.

Run with python scripts/verify_compilation_cache.py --output artifacts/compilation-cache.
Every timing uses the same CPU renderer and a separate, disposable cache directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QSettings

from stavellum.domain.models import (
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    RenderSettings,
    TrackInfo,
    save_document,
)
from stavellum.graphics.qt import ensure_app
from stavellum.presentation.compilation_cache import CompilationCache
from stavellum.presentation.scene import compile_scene
from stavellum.rendering.render import FrameRenderer
from stavellum.ui.gui import MainWindow


def document_for(bars):
    notes = [NoteEvent(f"carrier-{index}", "carrier", index * 120, 120,
                       (60, 64, 67, 72)[index % 4], 80) for index in range(bars * 16)]
    notes.extend((NoteEvent("target-first", "target", 0, 120, 79, 100),
                  NoteEvent("target-last", "target", (bars - 1) * 1920, 480, 79, 100)))
    return ProjectDocument(
        ProjectIR("benchmark.mid", "midi", "Cache benchmark", tracks=[
            TrackInfo("carrier", "Carrier"), TrackInfo("target", "Target")],
            notes=notes, duration_ticks=bars * 1920),
        [PartMapping("carrier", "Carrier", ["carrier"], clef="treble", key_signature=0),
         PartMapping("target", "Target", ["target"], clef="treble", key_signature=0)],
        settings=RenderSettings(width=640, height=360, render_backend="cpu"),
    )


def drawing(compiled, output, name):
    started = time.perf_counter()
    with FrameRenderer(compiled) as renderer:
        image = renderer.render_frame(0)
        first_frame_seconds = time.perf_counter() - started
        assert image.save(str(output / f"{name}.png"))
        digests = []
        for seconds in (0, compiled.score_duration / 2, max(0, compiled.score_duration - 1)):
            image = renderer.render_frame(seconds)
            digests.append(hashlib.sha256(bytes(image.constBits())).hexdigest())
    return first_frame_seconds, digests


def opening(app, project, settings_path):
    window = MainWindow(settings=QSettings(str(settings_path), QSettings.Format.IniFormat))
    failures = []
    window._job_error = lambda message, details: failures.append(details)
    started = time.perf_counter()
    window.open_path(str(project), check_unsaved=False)
    deadline = started + 120
    try:
        while not failures and (window.renderer is None or window._job is not None):
            app.processEvents()
            if time.perf_counter() > deadline:
                raise TimeoutError("project open did not complete")
            time.sleep(.005)
        if failures:
            raise RuntimeError("\n".join(failures))
        return time.perf_counter() - started, window._scene.compilation_report
    finally:
        window._dirty = False
        window.close()
        window.deleteLater()
        app.processEvents()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/compilation-cache"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = ensure_app()
    rows = []
    with tempfile.TemporaryDirectory(prefix="stavellum-cache-benchmark-") as temporary:
        directory = Path(temporary)
        os.environ["STAVELLUM_CACHE_DIR"] = str(directory / "cache")
        for bars in (16, 64, 128):
            document = document_for(bars)
            project = directory / f"{bars}.stproj"
            save_document(document, project)
            samples = []
            for repeat in range(args.repeats):
                cold = compile_scene(document, force_rebuild=True)
                cold_drawing, reference = drawing(cold, output, f"{bars}-cold-{repeat}")
                warm = compile_scene(document)
                warm_drawing, restored = drawing(warm, output, f"{bars}-warm-{repeat}")
                assert reference == restored
                assert warm.compilation_report["geometry_cache_hit"]
                assert warm.compilation_report["timeline_cache_hit"]
                CompilationCache().clear()
                cold_open, cold_open_report = opening(app, project, directory / "gui.ini")
                assert not cold_open_report["geometry_cache_hit"]
                warm_open, warm_open_report = opening(app, project, directory / "gui.ini")
                assert warm_open_report["geometry_cache_hit"] and warm_open_report["timeline_cache_hit"]
                samples.append({
                    "cold_compile_seconds": cold.compilation_report["total_compile_seconds"],
                    "warm_compile_seconds": warm.compilation_report["total_compile_seconds"],
                    "cold_first_frame_seconds": cold_drawing,
                    "warm_first_frame_seconds": warm_drawing,
                    "cold_open_seconds": cold_open, "warm_open_seconds": warm_open,
                    "cold_report": cold.compilation_report, "warm_report": warm.compilation_report,
                    "cold_open_report": cold_open_report, "warm_open_report": warm_open_report,
                    "frame_sha256": reference,
                })
                print(f"{bars} bars, repeat {repeat + 1}/{args.repeats}: "
                      f"compile {samples[-1]['cold_compile_seconds']:.3f}s -> "
                      f"{samples[-1]['warm_compile_seconds']:.3f}s; "
                      f"open {cold_open:.3f}s -> {warm_open:.3f}s", flush=True)
            medians = {name: statistics.median(sample[name] for sample in samples)
                       for name in samples[0] if name.endswith("_seconds")}
            rows.append({"bars": bars, "notes": len(document.project.notes),
                         "medians": medians, "samples": samples})
    (output / "benchmark.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
