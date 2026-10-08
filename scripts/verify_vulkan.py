"""Verify production Vulkan MP4s and optional CPU or explicit historical controls.

Quality screenshots are generated before timing. --quality-reviewed records the
operator's visual review; MAE alone never authorizes a full-song run. No backend
fallback is permitted. Project/audio inputs are read only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter

from stavellum.domain.models import load_document
from stavellum.exporting.encoding import probe_nvenc, video_arguments
from stavellum.exporting.export import _encode_attempt, audio_duration
from stavellum.graphics.qt import prepare_render_app
from stavellum.presentation.scene import compile_scene
from stavellum.rendering.render import FrameRenderer

_baseline_spec = importlib.util.spec_from_file_location("_benchmark_baseline", Path(__file__).with_name("_baseline.py"))
_baseline = importlib.util.module_from_spec(_baseline_spec)
_baseline_spec.loader.exec_module(_baseline)

MODES = ("rhi-vulkan",)
CASES = (("intro", 0.0), ("dense", 82.0), ("transition", 96.25))


def renderer_for(scene, mode, baseline=None):
    if mode == "baseline-opengl":
        if baseline is None:
            raise ValueError("baseline-opengl requires an explicit Git revision")
        return baseline(scene, export_readback="sync")
    if mode not in ("cpu", "rhi-vulkan"):
        raise ValueError("Unknown diagnostic renderer: " + mode)
    backend = "cpu" if mode == "cpu" else "gpu"
    return FrameRenderer(replace(scene, settings=replace(scene.settings, render_backend=backend)))


def gpu_identity(report):
    name = report["gpu_info"]["renderer"].split("/", 1)[0].strip().casefold()
    name = name.removeprefix("nvidia corporation ")
    if not name or any(marker in name for marker in ("llvmpipe", "lavapipe", "software")):
        raise RuntimeError("Expected hardware GPU for this controlled experiment: " + name)
    return name


def medians_and_gate(results, quality_passed, modes=MODES):
    medians = {}
    for label in dict.fromkeys(row["case"] for row in results if row["case"] != "full"):
        medians[label] = {mode: statistics.median(
            row["pipeline_seconds"] for row in results if row["case"] == label and row["mode"] == mode
        ) for mode in modes}
    gains = {"export_reduction_vs_baseline_opengl": None}
    regression = None
    passed = quality_passed
    if "baseline-opengl" in modes:
        original = sum(row["baseline-opengl"] for row in medians.values())
        vulkan = sum(row["rhi-vulkan"] for row in medians.values())
        gains["export_reduction_vs_baseline_opengl"] = 1 - vulkan / original
        regression = any(row["rhi-vulkan"] > row["baseline-opengl"] * 1.05 for row in medians.values())
        passed = quality_passed and gains["export_reduction_vs_baseline_opengl"] >= .10 and not regression
    return medians, {**gains, "any_case_regresses_over_5_percent": regression,
                     "quality_passed": quality_passed, "full_song_gate_passed": passed,
                     "historical_performance_comparison": "baseline-opengl" in modes}


class OffsetRenderer:
    def __init__(self, renderer, start):
        self.renderer, self.start = renderer, start

    @property
    def render_backend(self):
        return self.renderer.render_backend

    def export_frames(self, times, cancel):
        return self.renderer.export_frames((self.start + seconds for seconds in times), cancel)


def image_array(image):
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4).copy()


def quality_check(scene, directory, reviewed, modes=MODES, baseline=None, cases=CASES, duration=8):
    samples = sorted({0, .125, .5, 1, *(t for _, start in cases for t in
                     (start, start + duration / 2, start + duration - 1/60)), 140.0, 144.0, 148.0})
    reports, metrics = {}, []
    quality = directory / "quality"
    quality.mkdir(parents=True, exist_ok=True)
    # One renderer at a time; every sample remains reproducible from disk.
    for mode in modes:
        with renderer_for(scene, mode, baseline) as renderer:
            for seconds in samples:
                image = renderer.render_frame(seconds)
                if not image.save(str(quality / f"{seconds:g}-{mode}.png")):
                    raise RuntimeError("Cannot save quality evidence")
            reports[mode] = renderer.backend_report()
        expected = "cpu" if mode == "cpu" else "gpu"
        if reports[mode]["render_backend"] != expected or reports[mode].get("render_fallback_reasons"):
            raise RuntimeError("Quality renderer unexpectedly fell back")
    gpu_reports = [reports[mode] for mode in modes if mode != "cpu"]
    if len({gpu_identity(report) for report in gpu_reports}) != 1:
        raise RuntimeError("Quality groups must run on the same hardware GPU")
    if any(report["gpu_info"]["msaa_samples"] != 4 for report in gpu_reports):
        raise RuntimeError("Quality GPU groups must all use 4xMSAA")
    reference = "baseline-opengl" if "baseline-opengl" in modes else "rhi-vulkan"
    for seconds in samples:
        frames = {mode: QImage(str(quality / f"{seconds:g}-{mode}.png")) for mode in modes}
        arrays = {mode: image_array(image) for mode, image in frames.items()}
        for mode in modes:
            delta = np.abs(arrays[reference].astype(np.int16) - arrays[mode].astype(np.int16))
            metrics.append({"seconds": seconds, "mode": mode, "reference_mode": reference,
                            "mae_255": float(delta[:, :, :3].mean()),
                            "opaque_alpha": bool(np.all(arrays[mode][:, :, 3] == 255))})
            if mode != reference:
                rgba = np.empty((*delta.shape[:2], 4), np.uint8)
                rgba[:, :, :3] = np.minimum(delta[:, :, :3] * 8, 255).astype(np.uint8)
                rgba[:, :, 3] = 255
                QImage(rgba.data, scene.settings.width, scene.settings.height,
                       scene.settings.width * 4, QImage.Format.Format_RGBA8888).save(
                           str(quality / f"{seconds:g}-{mode}-difference-x8.png"))
        overview = QImage(640 * len(modes), 360, QImage.Format.Format_RGBA8888)
        painter = QPainter(overview)
        try:
            for index, mode in enumerate(modes):
                painter.drawImage(QRectF(index * 640, 0, 640, 360), frames[mode])
        finally:
            painter.end()
        overview.save(str(quality / f"{seconds:g}-overview.png"))
        metadata_rect = (0, round(scene.settings.height * .55), scene.settings.width, round(scene.settings.height * .45))
        for mode, image in frames.items():
            image.copy(*metadata_rect).save(str(quality / f"{seconds:g}-{mode}-metadata.png"))
    screening_passed = all(row["opaque_alpha"] and (row["mode"] == "cpu" or row["mae_255"] <= 1)
                           for row in metrics)
    return {"samples": metrics, "backend_reports": reports, "screening_mae_limit_255": 1,
            "screening_passed": screening_passed, "visual_review_recorded": reviewed,
            "quality_passed": screening_passed and reviewed,
            "column_order": list(modes),
            "note": "Inspect staff, glyphs, seams, metadata, activity lamps and tempo. Single-mode runs have no pixel control; CPU MAE is descriptive."}


def write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def probe_video(ffprobe, output, frames, width=1920, height=1080):
    process = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_entries",
                              "stream=codec_type,codec_name,nb_read_frames,width,height,avg_frame_rate,channels,duration",
                              "-of", "json", str(output)], capture_output=True, check=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    streams = json.loads(process.stdout)["streams"]
    video = next(row for row in streams if row["codec_type"] == "video")
    audio = next(row for row in streams if row["codec_type"] == "audio")
    if (int(video["nb_read_frames"]), video["width"], video["height"], video["avg_frame_rate"],
        video["codec_name"], audio["codec_name"], audio["channels"]) != (frames, width, height, "60/1", "h264", "aac", 2):
        raise RuntimeError("Encoded MP4 does not satisfy experiment format/frame/audio contract")
    return streams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/vulkan-verification"))
    parser.add_argument("--baseline-ref", help="Historical Git revision containing traditional OpenGL")
    parser.add_argument("--cpu", action="store_true", help="Include the explicit software renderer")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--starts", type=float, nargs="+")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--quality-reviewed", action="store_true")
    parser.add_argument("--full", action="store_true", help="Run three whole-song rounds only if all gates pass")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (args.repeats < 3 or args.seconds <= 0 or not math.isfinite(args.seconds)
            or (args.starts and any(start < 0 or not math.isfinite(start) for start in args.starts))):
        parser.error("At least three rounds required; seconds positive and starts finite/nonnegative")
    try:
        width, height = map(int, args.size.lower().split("x"))
        if min(width, height) < 1 or width % 2 or height % 2:
            raise ValueError
    except ValueError:
        parser.error("--size requires positive even WIDTHxHEIGHT for yuv420p")
    cases = tuple((f"at-{start:g}", start) for start in args.starts) if args.starts else CASES
    modes = (*MODES, *(("cpu",) if args.cpu else ()), *(("baseline-opengl",) if args.baseline_ref else ()))
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("FFmpeg and FFprobe required")
    if os.environ.get("STAVELLUM_RHI_TIMESTAMPS") not in (None, "", "0") and not args.quality_only:
        parser.error("Disable GPU timestamps for performance runs; use a separate diagnostic run")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "benchmark.json"
    document = load_document(args.project)
    document.settings = replace(document.settings, render_backend="gpu", width=width, height=height,
                                fps=60, cache_megabytes=64, video_encoder="h264_nvenc", nvenc_cq=18, nvenc_preset="p5")
    document.validate()
    prepare_render_app(document.settings)
    started = time.perf_counter()
    scene = compile_scene(document)
    report = {"project": args.project.name, "compile_seconds": time.perf_counter() - started,
              "modes": list(modes), "baseline_ref": args.baseline_ref, "size": args.size,
              "encoder": "h264_nvenc", "encoder_parameters": video_arguments(scene.settings, "h264_nvenc"),
              "limitations": "Synchronous QRhi offscreen frames; CPU engraving and CPU rawvideo pipe retained.",
              "results": [], "medians": {}, "status": "quality"}
    write_report(destination, report)  # a failed rerun must not leave an old success report
    baseline = None
    try:
        if args.baseline_ref:
            package, report["baseline_commit"] = _baseline.load_package(args.baseline_ref, directory / "baseline-source")
            baseline = _baseline.renderer(package)
        report["quality"] = quality_check(scene, directory, args.quality_reviewed, modes, baseline, cases, args.seconds)
    except BaseException as error:
        report["status"], report["error"] = "failed", str(error)
        write_report(destination, report)
        raise
    write_report(destination, report)
    if args.quality_only:
        print(json.dumps({"quality": report["quality"]["screening_passed"], "report": str(destination)}), flush=True)
        return
    if not report["quality"]["quality_passed"]:
        raise RuntimeError("Inspect quality PNGs and rerun with --quality-reviewed; quality must pass before timing")
    available, reason = probe_nvenc(ffmpeg, scene.settings, lambda: False)
    if not available:
        raise RuntimeError("NVENC unavailable: " + reason)

    def run_case(label, start, duration, full=False):
        frames = math.ceil(duration * 60)
        duration = frames / 60
        audio = Path(document.audio_path).resolve()
        encoding_scene = scene
        if not full:
            audio = directory / f"{label}.wav"
            delay = scene.settings.intro_delay_seconds * 1000
            audio_filter = (f"adelay=delays={delay:.9f}:all=1,apad,atrim=start={start:.9f}:duration={duration:.9f},asetpts=PTS-STARTPTS")
            subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", document.audio_path,
                            "-af", audio_filter, "-t", str(duration), "-c:a", "pcm_s16le", str(audio)],
                           capture_output=True, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            encoding_scene = replace(scene, settings=replace(scene.settings, intro_delay_seconds=0))
        for repeat in range(args.repeats):
            rotation = repeat % len(modes)
            for mode in modes[rotation:] + modes[:rotation]:
                output = directory / f"{label}-{mode}.mp4"
                log = directory / f"{label}-{mode}-round-{repeat + 1}.log"
                started = time.perf_counter()
                with renderer_for(scene, mode, baseline) as renderer:
                    initialization = time.perf_counter() - started
                    started = time.perf_counter()
                    last_update = [started]

                    def progress(fraction, message):
                        if time.perf_counter() - last_update[0] >= 20:
                            print(json.dumps({"case": label, "mode": mode, "progress": fraction}), flush=True)
                            last_update[0] = time.perf_counter()

                    try:
                        measurements = _encode_attempt(ffmpeg, "h264_nvenc", encoding_scene,
                                                       OffsetRenderer(renderer, start), audio, output,
                                                       frames, duration, progress, lambda: False)
                    except BaseException as error:
                        log.write_text(str(error), encoding="utf-8")
                        raise
                    wall = time.perf_counter() - started
                    backend = renderer.backend_report()
                expected = "cpu" if mode == "cpu" else "gpu"
                if backend["render_backend"] != expected or backend.get("render_fallback_reasons"):
                    raise RuntimeError("Benchmark renderer unexpectedly fell back")
                if mode != "cpu" and gpu_identity(backend) != gpu_identity(report["quality"]["backend_reports"]["rhi-vulkan"]):
                    raise RuntimeError("Benchmark GPU changed")
                log.write_text(json.dumps(measurements, ensure_ascii=False, indent=2), encoding="utf-8")
                result = {**backend, **measurements, "case": label, "start_seconds": start, "mode": mode,
                          "repeat": repeat + 1, "frames": frames, "initialization_seconds": initialization,
                          "pipeline_seconds": wall, "ffprobe": probe_video(ffprobe, output, frames, width, height)}
                report["results"].append(result)
                report["status"] = "timing"
                write_report(destination, report)
                print(json.dumps({"case": label, "mode": mode, "round": repeat + 1, "seconds": wall}), flush=True)
    try:
        for label, start in cases:
            run_case(label, start, args.seconds)
        report["medians"], report["gate"] = medians_and_gate(report["results"], True, modes)
        report["full_song"] = {"requested": args.full, "executed": False,
                               "skip_reason": "Performance/quality gate not satisfied" if not report["gate"]["full_song_gate_passed"] else "Not requested"}
        write_report(destination, report)
        if args.full and report["gate"]["full_song_gate_passed"]:
            run_case("full", 0, scene.settings.presentation_duration(audio_duration(document.audio_path), scene.score_duration), True)
            report["full_song"] = {"requested": True, "executed": True}
            report["medians"]["full"] = {mode: statistics.median(
                row["pipeline_seconds"] for row in report["results"] if row["case"] == "full" and row["mode"] == mode
            ) for mode in modes}
        report["status"] = "complete"
    except BaseException as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        write_report(destination, report)
    print(json.dumps({"medians": report["medians"], "gate": report["gate"]}), flush=True)


if __name__ == "__main__":
    main()
