"""Serial high-resolution preview and MP4 comparisons using an optional Git baseline.

Both renderers receive the same compiled scene. Inputs are read only; reports,
baseline source, audio clips and the final repeat's snapshots stay in --output.
Preview warms the initial frame, then measures continuous playback, including
the cost of newly encountered tiles. It does not prewarm future frames.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import stavellum.presentation.visibility as visibility_module
from stavellum.domain.models import load_document
from stavellum.exporting.encoding import probe_nvenc, video_arguments
from stavellum.exporting.export import _encode_attempt
from stavellum.graphics.qt import prepare_render_app
from stavellum.presentation.scene import compile_scene
from stavellum.rendering.render import FrameRenderer

_baseline_spec = importlib.util.spec_from_file_location("_benchmark_baseline", Path(__file__).with_name("_baseline.py"))
_baseline = importlib.util.module_from_spec(_baseline_spec)
_baseline_spec.loader.exec_module(_baseline)
load_package = _baseline.load_package


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def baseline_module(ref, directory, name):
    package, _ = load_package(ref, directory / "baseline-source")
    return _baseline.module(package, name)


class OffsetRenderer:
    def __init__(self, renderer, start, frames):
        self.renderer, self.start, self.frames = renderer, start, frames
        self.times = []
        self.snapshots = {}

    @property
    def render_backend(self):
        return self.renderer.render_backend

    def export_frames(self, times, cancel):
        stream = self.renderer.export_frames((self.start + seconds for seconds in times), cancel)
        owner = self

        class Stream:
            pixel_format = stream.pixel_format

            def __iter__(self):
                return self

            def __next__(self):
                started = time.perf_counter()
                index, image = next(stream)
                owner.times.append(time.perf_counter() - started)
                if index in (0, owner.frames // 2, owner.frames - 1):
                    owner.snapshots[index] = image
                return index, image

            def close(self):
                stream.close()

            def report(self):
                return stream.report()

        return Stream()


def preview(renderer, start, frames):
    started = time.perf_counter()
    renderer.render_frame(start)
    cold = time.perf_counter() - started
    initial_misses = renderer.backend_report().get("tile_cache_misses")
    fixed_samples = []
    for _ in range(6):
        started = time.perf_counter()
        renderer.render_frame(start)
        fixed_samples.append((time.perf_counter() - started) * 1000)
    fixed_misses = renderer.backend_report().get("tile_cache_misses")
    samples = []
    for index in range(frames):
        started = time.perf_counter()
        renderer.render_frame(start + index / renderer.scene.settings.fps)
        samples.append((time.perf_counter() - started) * 1000)
    return {"cold_frame_ms": cold * 1000,
            "fixed_warm_p50_ms": percentile(fixed_samples, .5),
            "fixed_warm_p95_ms": percentile(fixed_samples, .95),
            "fixed_warm_tile_misses": (None if initial_misses is None
                                       else fixed_misses - initial_misses),
            "warm_p50_ms": percentile(samples, .5),
            "warm_p95_ms": percentile(samples, .95),
            "warm_p99_ms": percentile(samples, .99),
            "frame_samples_ms": samples, **renderer.backend_report()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--baseline-layout", action="store_true",
                        help="Use baseline layout for both renderers to isolate raster/cache changes")
    parser.add_argument("--sizes", nargs="+", default=["2560x1440"])
    parser.add_argument("--starts", type=float, nargs="+", default=[63, 151])
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--preview-frames", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--encoder", choices=["h264_nvenc", "libx264"], default="h264_nvenc")
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    if (args.repeats < 1 or args.preview_frames < 1 or args.seconds <= 0
            or not math.isfinite(args.seconds)
            or any(start < 0 or not math.isfinite(start) for start in args.starts)):
        parser.error("counts/duration must be positive and times finite and nonnegative")
    try:
        sizes = [tuple(map(int, size.lower().split("x"))) for size in args.sizes]
        if any(len(size) != 2 for size in sizes):
            raise ValueError
    except ValueError:
        parser.error("sizes must use WIDTHxHEIGHT")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not args.preview_only and (not ffmpeg or not ffprobe):
        parser.error("FFmpeg and FFprobe are required for MP4 verification")
    modes = {"optimized": FrameRenderer}
    if args.baseline_ref:
        modes["baseline"] = baseline_module(args.baseline_ref, directory, "render").FrameRenderer
    if args.baseline_layout:
        if not args.baseline_ref:
            parser.error("--baseline-layout requires --baseline-ref")
        visibility_module.compile_layout = baseline_module(args.baseline_ref, directory, "layout").compile_layout
    report = {"project": args.project.name, "baseline_ref": args.baseline_ref,
              "layout": "baseline" if args.baseline_layout else "current",
              "preview_warmup": "initial_frame_only",
              "backend": args.backend, "encoder": args.encoder,
              "repeats": args.repeats, "results": [], "medians": {}}
    destination = directory / "benchmark.json"
    for width, height in sizes:
        document = load_document(args.project)
        document.settings.width, document.settings.height = width, height
        document.settings.fps = 60
        document.settings.render_backend = args.backend
        document.validate()
        prepare_render_app(document.settings)
        started = time.perf_counter()
        scene = compile_scene(document)
        compilation = time.perf_counter() - started
        if not args.preview_only and args.encoder == "h264_nvenc":
            available, reason = probe_nvenc(ffmpeg, scene.settings, lambda: False)
            if not available:
                raise RuntimeError("NVENC unavailable: " + reason)
        for start in args.starts:
            label = f"{width}x{height}-at-{start:g}"
            frames = math.ceil(args.seconds * scene.settings.fps)
            duration = frames / scene.settings.fps
            audio = directory / f"{label}.wav"
            if not args.preview_only:
                delay = scene.settings.intro_delay_seconds * 1000
                audio_filter = (f"adelay={delay:.9f}:all=1,apad,"
                                f"atrim=start={start:.9f}:duration={duration:.9f},asetpts=PTS-STARTPTS")
                subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y",
                                "-i", document.audio_path, "-af", audio_filter,
                                "-t", str(duration), "-c:a", "pcm_s16le", str(audio)],
                               check=True, capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for repeat in range(args.repeats):
                names = list(modes)
                names = names[repeat % len(names):] + names[:repeat % len(names)]
                for name in names:
                    with modes[name](scene) as renderer:
                        measurements = preview(renderer, start, args.preview_frames)
                        result = {"case": label, "mode": name, "repeat": repeat + 1,
                                  "kind": "preview", "compile_seconds": compilation,
                                  **measurements}
                    report["results"].append(result)
                    print(json.dumps({key: result[key] for key in (
                        "case", "mode", "repeat", "kind", "cold_frame_ms",
                        "warm_p50_ms", "warm_p95_ms")}), flush=True)
                    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    if args.preview_only:
                        continue
                    output = directory / f"{label}-{name}.mp4"
                    with modes[name](scene) as renderer:
                        offset = OffsetRenderer(renderer, start, frames)
                        encoding_scene = replace(scene, settings=replace(scene.settings, intro_delay_seconds=0))
                        statistics_report = _encode_attempt(
                            ffmpeg, args.encoder, encoding_scene, offset, audio, output,
                            frames, duration, lambda *_: None, lambda: False)
                        result = {"case": label, "mode": name, "repeat": repeat + 1,
                                  "kind": "export", "frames": frames,
                                  "encoder_parameters": video_arguments(scene.settings, args.encoder),
                                  "first_frame_ms": offset.times[0] * 1000,
                                  "remaining_p95_ms": percentile(offset.times[1:] or offset.times, .95) * 1000,
                                  **statistics_report, **renderer.backend_report()}
                        if repeat == args.repeats - 1:
                            for index, image in offset.snapshots.items():
                                if not image.save(str(directory / f"{label}-{name}-{index:04d}.png")):
                                    raise OSError("Failed to save verification frame")
                    probe = subprocess.run(
                        [ffprobe, "-v", "error", "-count_frames", "-show_entries",
                         "stream=codec_type,nb_read_frames,width,height,avg_frame_rate,duration",
                         "-of", "json", str(output)], check=True, capture_output=True,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    streams = json.loads(probe.stdout)["streams"]
                    video = next(stream for stream in streams if stream["codec_type"] == "video")
                    assert (int(video["nb_read_frames"]), video["width"], video["height"],
                            video["avg_frame_rate"]) == (frames, width, height, "60/1")
                    assert abs(float(video["duration"]) - duration) <= 1 / scene.settings.fps
                    assert any(stream["codec_type"] == "audio" for stream in streams)
                    result["ffprobe"] = streams
                    report["results"].append(result)
                    print(json.dumps({key: result[key] for key in (
                        "case", "mode", "repeat", "kind", "frames", "wall_seconds")}), flush=True)
                    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            medians = {}
            for name in modes:
                preview_results = [item for item in report["results"]
                                   if item["case"] == label and item["mode"] == name and item["kind"] == "preview"]
                medians[name] = {key: statistics.median(item[key] for item in preview_results)
                                 for key in ("cold_frame_ms", "warm_p50_ms", "warm_p95_ms", "warm_p99_ms")}
                exports = [item for item in report["results"]
                           if item["case"] == label and item["mode"] == name and item["kind"] == "export"]
                if exports:
                    medians[name]["export_wall_seconds"] = statistics.median(item["wall_seconds"] for item in exports)
            if "baseline" in medians:
                medians["preview_p95_reduction"] = 1 - medians["optimized"]["warm_p95_ms"] / medians["baseline"]["warm_p95_ms"]
                medians["preview_p50_reduction"] = 1 - medians["optimized"]["warm_p50_ms"] / medians["baseline"]["warm_p50_ms"]
                if not args.preview_only:
                    medians["export_reduction"] = 1 - medians["optimized"]["export_wall_seconds"] / medians["baseline"]["export_wall_seconds"]
            report["medians"][label] = medians
            destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
