"""Alternate an explicit Git baseline and required Rust core, including pixel and encode checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import shutil
import statistics
import time
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from _baseline import load_package, module
from PySide6.QtGui import QImage

from stavellum.domain.models import (
    Metadata,
    NoteEvent,
    PartMapping,
    ProjectDocument,
    ProjectIR,
    RenderSettings,
    TrackInfo,
)
from stavellum.exporting.export import _encode_attempt
from stavellum.graphics.qt import ensure_app
from stavellum.presentation._core import library_path
from stavellum.presentation.scene import compile_scene
from stavellum.rendering._rhi import library_path as renderer_library_path
from stavellum.rendering.render import FrameRenderer


def document(dense, width, height):
    count = 7 if dense else 2
    notes = [NoteEvent(f"{part}-{index}", f"track-{part}", index * 240, 240,
                       [60, 64, 67, 72][(index + part) % 4], 70 + index % 50)
             for part in range(count) for index in range(360 if dense else 96)
             if dense or part == 0 or index < 8 or 32 <= index < 40 or index >= 64]
    return ProjectDocument(
        ProjectIR("core-benchmark.mid", "midi", "Core benchmark",
                  tracks=[TrackInfo(f"track-{part}", f"Voice {part}") for part in range(count)],
                  notes=notes),
        [PartMapping(f"part-{part}", f"Voice {part}", [f"track-{part}"], instrument="violin",
                     clef="treble", key_signature=0, icon="violin") for part in range(count)],
        settings=RenderSettings(width=width, height=height, fps=60, cache_megabytes=64,
                                render_backend="cpu", intro_delay_seconds=2,
                                announcement_auto_hide=True, preset="ultrafast"),
        metadata=Metadata(title="Scene core benchmark"),
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgba(image):
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return bytes(converted.constBits())


def consume(renderer, times):
    started = time.perf_counter()
    with renderer.export_frames(times, lambda: False) as stream:
        for _, image in stream:
            del image
        report = stream.report()
    return time.perf_counter() - started, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/core-benchmark"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--evaluation-frames", type=int, default=1000)
    parser.add_argument("--encoder", choices=("libx264", "h264_nvenc"), default="libx264")
    parser.add_argument("--backends", nargs="+", choices=("cpu", "gpu"), default=["cpu", "gpu"])
    parser.add_argument("--phases", nargs="+", choices=("cold", "warm", "encode"), default=["cold", "warm", "encode"])
    parser.add_argument("--cases", nargs="+", choices=("dense", "sparse"), default=["dense", "sparse"])
    parser.add_argument("--sizes", nargs="+", default=["1920x1080", "3840x2160"])
    args = parser.parse_args()
    if min(args.repeats, args.frames, args.evaluation_frames) < 1:
        parser.error("counts must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("FFmpeg is required for the export benchmark")
    sizes = [tuple(map(int, value.split("x"))) for value in args.sizes]
    if any(len(size) != 2 or min(size) < 1 for size in sizes):
        parser.error("sizes must be WIDTHxHEIGHT")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    package, commit = load_package(args.baseline_ref, directory / "baseline", current_native=True)
    old_scene = module(package, "scene")
    old_models = module(package, "models")
    old_render = module(package, "render")
    old_raster = module(package, "raster")
    ensure_app()
    audio = directory / "silence.wav"
    with wave.open(str(audio), "wb") as output:
        output.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        output.writeframes(b"\0" * (math.ceil(args.frames / 60 + 1) * 48000 * 4))
    report = {
        "status": "running", "baseline_commit": commit,
        "platform": platform.platform(), "python": platform.python_version(),
        "core_library": str(library_path()), "core_sha256": digest(library_path()),
        "scene_compute_backend": "rust", "scene_compute_abi": 1,
        "renderer_library": str(renderer_library_path()), "renderer_sha256": digest(renderer_library_path()),
        "baseline_renderer": "Both versions use the installed compositor DLL; only Python scene evaluation differs.",
        "repeats": args.repeats, "frames": args.frames, "encoder": args.encoder,
        "limitations": "Single host; generated scenes; compilation excluded; bounded clips include encoder startup/drain. No whole-song speedup claim.",
        "samples": [], "quality": [], "medians": {}, "evaluation": [],
    }

    def save():
        (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        for case in args.cases:
            for width, height in sizes:
                doc = document(case == "dense", width, height)
                scenes = {
                    "current": compile_scene(doc, use_cache=False),
                    "baseline": old_scene.compile_scene(old_models.ProjectDocument.from_dict(doc.to_dict()), use_cache=False),
                }
                classes = {"baseline": old_render.FrameRenderer, "current": FrameRenderer}
                label = f"{case}-{width}x{height}"
                times = [6 + i / 60 for i in range(args.frames)]
                rng = random.Random(582)
                evaluation_times = [rng.uniform(0, scenes["current"].score_duration + 2)
                                    for _ in range(args.evaluation_frames)]
                values = {"baseline": [], "current": []}
                with old_raster.RasterFrameRenderer(scenes["baseline"]) as old_assets, \
                        FrameRenderer(scenes["current"]) as current_assets:
                    def baseline(t):
                        scene = scenes["baseline"]
                        layout, x = scene.layout_at(t), scene.camera_x_at(t)
                        return x, layout, [old_scene.activity_levels(p, scene.settings.audio_time(t)) for p in scene.parts], old_assets._tile_plan(layout, x)

                    def current(t):
                        state = current_assets._evaluator.evaluate(t)
                        return state, current_assets._tile_plan(state)

                    functions = {"baseline": baseline, "current": current}
                    for function in functions.values():
                        for t in evaluation_times[:50]:
                            function(t)
                    for repeat in range(args.repeats):
                        for version in (["baseline", "current"] if repeat % 2 == 0 else ["current", "baseline"]):
                            started = time.perf_counter()
                            for t in evaluation_times:
                                functions[version](t)
                            values[version].append((time.perf_counter() - started) * 1000 / len(evaluation_times))
                    report["evaluation"].append({
                        "case": label, "parts": len(doc.mappings), "notes": len(doc.project.notes),
                        "samples_ms": values,
                        "speedup": statistics.median(values["baseline"]) / statistics.median(values["current"]),
                        "initialization_seconds": current_assets._evaluator.initialization_seconds,
                    })
                    overhead = {"native": [], "materialized": []}
                    for repeat in range(args.repeats):
                        for mode in (["native", "materialized"] if repeat % 2 == 0 else ["materialized", "native"]):
                            started = time.perf_counter()
                            for t in evaluation_times:
                                if mode == "native":
                                    current_assets._evaluator._target.frame(t, doc.settings.audio_time(t))
                                else:
                                    current_assets._evaluator.evaluate(t)
                            overhead[mode].append((time.perf_counter() - started) * 1000 / len(evaluation_times))
                    report["evaluation"][-1].update({
                        "conversion_samples_ms": overhead,
                        "materialization_and_cache_overhead_ms": max(0, statistics.median(overhead["materialized"]) - statistics.median(overhead["native"])),
                    })
                for backend in args.backends:
                    configured = {version: replace(scene, settings=replace(scene.settings, render_backend=backend))
                                  for version, scene in scenes.items()}
                    quality_times = [0, 1.999999, 2, 2.05, 6, 10, 16.001, 20, 24]
                    quality_times += [rng.uniform(0, scenes["current"].score_duration + 2) for _ in range(8)]
                    rng.shuffle(quality_times)
                    quality_times += quality_times[:3]
                    with classes["baseline"](configured["baseline"]) as reference, \
                            classes["current"](configured["current"]) as candidate:
                        for t in quality_times:
                            before = reference.render_frame(t)
                            after = candidate.render_frame(t)
                            equal = rgba(before) == rgba(after)
                            report["quality"].append({"case": label, "backend": backend, "time": t, "pixel_exact": equal})
                            if not equal:
                                before.save(str(directory / f"{label}-{backend}-{t}-baseline.png"))
                                after.save(str(directory / f"{label}-{backend}-{t}-current.png"))
                                raise RuntimeError(f"Pixel mismatch: {label}/{backend}/{t}")
                    for batch_size in (1, 2, 4):
                        # Adjacent frames exercise actual batches; jumping between
                        # clusters also checks the bounded state cache and lookahead.
                        batch_times = [10 + i / 60 for i in range(4)] + [6 + i / 60 for i in range(4)]
                        with classes["baseline"](configured["baseline"]) as reference, \
                                classes["current"](configured["current"]) as candidate:
                            if backend == "gpu":
                                reference._gpu._requested_batch_size = batch_size
                                candidate._gpu._requested_batch_size = batch_size
                            with reference.export_frames(batch_times, lambda: False) as before_stream, \
                                    candidate.export_frames(batch_times, lambda: False) as after_stream:
                                for (before_index, before), (after_index, after) in zip(before_stream, after_stream, strict=True):
                                    equal = before_index == after_index and rgba(before) == rgba(after)
                                    report["quality"].append({"case": label, "backend": backend, "time": batch_times[before_index],
                                                              "requested_batch_size": batch_size, "pixel_exact": equal})
                                    if not equal:
                                        raise RuntimeError(f"Batch pixel mismatch: {label}/{backend}/{batch_size}/{before_index}")
                    for phase in args.phases:
                        key = f"{label}-{backend}-{phase}"
                        for repeat in range(args.repeats):
                            for version in (["baseline", "current"] if repeat % 2 == 0 else ["current", "baseline"]):
                                with classes[version](configured[version]) as renderer:
                                    if phase == "warm":
                                        consume(renderer, iter(times))
                                    if phase == "encode":
                                        destination = directory / f"{key}-{version}.mp4"
                                        started = time.perf_counter()
                                        offset_renderer = SimpleNamespace(render_backend=renderer.render_backend, export_frames=lambda ts, cancel: renderer.export_frames((t + 6 for t in ts), cancel))
                                        encoding_scene = replace(configured[version], settings=replace(configured[version].settings, intro_delay_seconds=0))
                                        stats = _encode_attempt(ffmpeg, args.encoder, encoding_scene, offset_renderer,
                                                                audio, destination, args.frames, args.frames / 60,
                                                                lambda *args: None, lambda: False)
                                        elapsed = time.perf_counter() - started
                                    else:
                                        elapsed, stats = consume(renderer, iter(times))
                                    native = renderer.backend_report()
                                    if native["render_backend"] != backend:
                                        raise RuntimeError("Benchmark unexpectedly fell back from requested backend")
                                    report["samples"].append({"key": key, "repeat": repeat, "version": version,
                                                              "seconds": elapsed, "stream": stats, "backend": native})
                                save()
                        medians = {version: statistics.median(item["seconds"] for item in report["samples"]
                                                             if item["key"] == key and item["version"] == version)
                                   for version in ("baseline", "current")}
                        medians["change_percent"] = (medians["current"] / medians["baseline"] - 1) * 100
                        report["medians"][key] = medians
                        print(json.dumps({"case": key, **medians}), flush=True)
                save()
        report["status"] = "complete"
        report["evaluation_target_met"] = all(item["speedup"] >= 3 for item in report["evaluation"] if item["case"].startswith("dense"))
        report["regressions_over_5_percent"] = [key for key, value in report["medians"].items() if value["change_percent"] > 5]
        report["acceptance_met"] = report["evaluation_target_met"] and not report["regressions_over_5_percent"]
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
