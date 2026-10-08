"""Compare four rendering/encoding combinations on the same bounded 1080p60 clip.

Source projects and audio are read only. Each combination uses a new renderer,
and its actual output is independently checked with FFprobe.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from stavellum.domain.models import load_document
from stavellum.exporting.encoding import probe_nvenc
from stavellum.exporting.export import _encode_attempt
from stavellum.graphics.qt import prepare_render_app
from stavellum.presentation.scene import compile_scene
from stavellum.rendering.render import FrameRenderer


class ClipRenderer:
    def __init__(self, renderer, start: float, frames: int):
        self.renderer = renderer
        self.start = start
        self.frames = frames
        self.times: list[float] = []
        self.snapshots = {}

    @property
    def render_backend(self):
        return self.renderer.render_backend

    def render_frame(self, seconds):
        started = time.perf_counter()
        image = self.renderer.render_frame(self.start + seconds)
        self.times.append(time.perf_counter() - started)
        index = len(self.times) - 1
        if index in (0, self.frames // 2, self.frames - 1):
            self.snapshots[index] = image
        return image

    def export_frames(self, times, cancel):
        stream = self.renderer.export_frames((self.start + time for time in times), cancel)
        owner = self

        class ClipStream:
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

        return ClipStream()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=8)
    args = parser.parse_args()
    if args.start < 0 or args.seconds <= 0 or not math.isfinite(args.start + args.seconds):
        parser.error("start must be nonnegative and seconds must be positive and finite")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("FFmpeg and FFprobe are required")
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    document = load_document(args.project)
    document.settings.width, document.settings.height, document.settings.fps = 1920, 1080, 60
    document.settings.intro_delay_seconds = 0
    document.settings.render_backend = "auto"
    document.validate()
    prepare_render_app(document.settings)
    started = time.perf_counter()
    scene = compile_scene(document)
    compilation = time.perf_counter() - started
    frames = math.ceil(args.seconds * 60)
    duration = frames / 60
    audio = destination / "clip.wav"
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-ss", str(args.start),
                    "-i", document.audio_path, "-af", "apad", "-t", str(duration),
                    "-c:a", "pcm_s16le", str(audio)], check=True, capture_output=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    nvenc_available, nvenc_reason = probe_nvenc(ffmpeg, document.settings, lambda: False)
    results = []
    cache_measurements = []
    for backend in ("cpu", "gpu"):
        scene.settings = replace(scene.settings, render_backend=backend)
        # Separate cold-cache and repeated-window measurements from video export.
        try:
            with FrameRenderer(scene) as renderer:
                started = time.perf_counter()
                renderer.render_frame(args.start)
                cold_seconds = time.perf_counter() - started
                samples = [args.start + index / 60 for index in range(min(120, frames))]
                for seconds in samples[1:]:
                    renderer.render_frame(seconds)
                started = time.perf_counter()
                for seconds in samples:
                    renderer.render_frame(seconds)
                warm_seconds = (time.perf_counter() - started) / len(samples)
                cache_measurements.append({"backend": backend, "cold_frame_seconds": cold_seconds,
                                           "warm_mean_frame_seconds": warm_seconds,
                                           **renderer.backend_report()})
        except RuntimeError as error:
            cache_measurements.append({"backend": backend, "error": str(error)})
            continue
        for encoder in ("libx264", "h264_nvenc"):
            if encoder == "h264_nvenc" and not nvenc_available:
                results.append({"backend": backend, "encoder": encoder, "error": nvenc_reason})
                continue
            target = destination / f"{backend}-{encoder}.mp4"
            with FrameRenderer(scene) as renderer:
                clip = ClipRenderer(renderer, args.start, frames)
                statistics = _encode_attempt(ffmpeg, encoder, scene, clip, audio, target,
                                             frames, duration, lambda *_: None, lambda: False)
                data = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_streams",
                                       "-show_format", "-of", "json", str(target)],
                                      check=True, capture_output=True,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                info = json.loads(data.stdout)
                video = next(stream for stream in info["streams"] if stream["codec_type"] == "video")
                assert int(video["nb_read_frames"]) == frames
                assert video["avg_frame_rate"] == "60/1"
                assert (video["width"], video["height"]) == (1920, 1080)
                for index, image in clip.snapshots.items():
                    if not image.save(str(destination / f"{backend}-{encoder}-{index:04d}.png")):
                        raise OSError("Could not save verification frame")
                result = {"backend": backend, "encoder": encoder, "frames": frames,
                          "first_frame_seconds": clip.times[0],
                          "remaining_mean_frame_seconds": sum(clip.times[1:]) / max(1, frames - 1),
                          "file_bytes": target.stat().st_size, "ffprobe": info,
                          **statistics, **renderer.backend_report()}
                results.append(result)
                print(json.dumps({key: result[key] for key in (
                    "backend", "encoder", "wall_seconds", "render_seconds", "readback_seconds",
                    "pipe_wait_seconds", "frames")}), flush=True)
    report = {"sample_project": args.project.name, "start_seconds": args.start,
              "duration_seconds": duration, "compile_seconds": compilation,
              "cpu_crf": scene.settings.crf, "cpu_preset": scene.settings.preset,
              "nvenc_cq": scene.settings.nvenc_cq, "nvenc_preset": scene.settings.nvenc_preset,
              "cache_measurements": cache_measurements, "combinations": results}
    (destination / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
