"""Serial comparisons of Vulkan owned and copied CPU readback delivery.

Source files are read only. Every run encodes a real MP4 and verifies its frame
count with FFprobe. Keep the last video for each case and save all measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from stavellum.domain.models import load_document
from stavellum.exporting.encoding import probe_nvenc, video_arguments
from stavellum.exporting.export import _encode_attempt, audio_duration
from stavellum.graphics.qt import prepare_render_app
from stavellum.presentation.scene import compile_scene
from stavellum.rendering.render import FrameRenderer


@contextmanager
def readback_delivery(mode):
    name = "STAVELLUM_RHI_COPY_READBACK"
    previous = os.environ.get(name)
    os.environ[name] = "1" if mode == "copy" else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


class OffsetRenderer:
    def __init__(self, renderer, start):
        self.renderer, self.start = renderer, start

    @property
    def render_backend(self):
        return self.renderer.render_backend

    def render_frame(self, seconds):
        return self.renderer.render_frame(self.start + seconds)

    def export_frames(self, times, cancel):
        return self.renderer.export_frames((self.start + seconds for seconds in times), cancel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--starts", type=float, nargs="+", default=[0, 60, 140])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--encoder", choices=["h264_nvenc", "libx264"], default="h264_nvenc")
    args = parser.parse_args()
    if (args.repeats < 1 or args.seconds <= 0 or not math.isfinite(args.seconds)
            or any(start < 0 or not math.isfinite(start) for start in args.starts)):
        parser.error("repeats/seconds must be positive; starts must be finite and nonnegative")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("FFmpeg and FFprobe are required")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    document = load_document(args.project)
    document.settings.render_backend = "gpu"
    document.settings.width, document.settings.height, document.settings.fps = 1920, 1080, 60
    document.validate()
    prepare_render_app(document.settings)
    started = time.perf_counter()
    scene = compile_scene(document)
    compilation = time.perf_counter() - started
    if args.encoder == "h264_nvenc":
        available, reason = probe_nvenc(ffmpeg, scene.settings, lambda: False)
        if not available:
            raise RuntimeError("NVENC unavailable: " + reason)
    cases = [(f"at-{start:g}", start, args.seconds) for start in args.starts]
    if args.full:
        duration = scene.settings.presentation_duration(
            audio_duration(document.audio_path), scene.score_duration)
        cases.append(("full", 0, duration))
    results = []
    report = {"project": args.project.name, "compile_seconds": compilation,
              "encoder": args.encoder, "encoder_parameters": video_arguments(scene.settings, args.encoder),
              "gpu_info": {}, "results": results, "medians": {}}
    destination = directory / "benchmark.json"
    for label, start, duration in cases:
        frames = math.ceil(duration * scene.settings.fps)
        duration = frames / scene.settings.fps
        audio = Path(document.audio_path).resolve()
        encoding_scene = scene
        if label != "full":
            audio = directory / f"{label}.wav"
            # Clip presentation audio after the intro delay, keeping the same
            # audio time as the offset frames. Do not add the intro a second time.
            delay = scene.settings.intro_delay_seconds * 1000
            audio_filter = (f"adelay=delays={delay:.9f}:all=1,apad,"
                            f"atrim=start={start:.9f}:duration={duration:.9f},asetpts=PTS-STARTPTS")
            subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y",
                            "-i", document.audio_path, "-af", audio_filter, "-t", str(duration),
                            "-c:a", "pcm_s16le", str(audio)], check=True, capture_output=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            encoding_scene = replace(scene, settings=replace(scene.settings, intro_delay_seconds=0))
        for repeat in range(args.repeats):
            modes = ["owned", "copy"]
            modes = modes[repeat % 2:] + modes[:repeat % 2]
            for mode in modes:
                output = directory / f"{label}-{mode}.mp4"
                with readback_delivery(mode), FrameRenderer(scene) as renderer:
                    offset = OffsetRenderer(renderer, start)
                    started = time.perf_counter()
                    last_progress = [time.perf_counter()]

                    def progress(fraction, message):
                        if time.perf_counter() - last_progress[0] >= 30:
                            print(json.dumps({"case": label, "mode": mode,
                                              "progress": fraction}), flush=True)
                            last_progress[0] = time.perf_counter()

                    measurements = _encode_attempt(ffmpeg, args.encoder, encoding_scene, offset,
                                                   audio, output, frames, duration,
                                                   progress, lambda: False)
                    pipeline_seconds = time.perf_counter() - started
                    backend = renderer.backend_report()
                    report["gpu_info"] = backend["gpu_info"]
                probe = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_entries",
                                        "stream=codec_type,nb_read_frames,width,height,avg_frame_rate,duration",
                                        "-of", "json", str(output)], capture_output=True, check=True,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                streams = json.loads(probe.stdout)["streams"]
                video = next(stream for stream in streams if stream["codec_type"] == "video")
                assert int(video["nb_read_frames"]) == frames
                assert (video["width"], video["height"], video["avg_frame_rate"]) == (1920, 1080, "60/1")
                result = {**backend, **measurements, "case": label, "start_seconds": start,
                          "mode": mode, "repeat": repeat + 1, "frames": frames,
                          "duration_seconds": duration, "pipeline_seconds": pipeline_seconds,
                          "ffprobe": streams}
                results.append(result)
                print(json.dumps({key: result[key] for key in ["case", "mode", "repeat", "frames",
                                                              "pipeline_seconds", "readback_mode"]}), flush=True)
                destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        medians = {mode: statistics.median(item["pipeline_seconds"] for item in results
                                          if item["case"] == label and item["mode"] == mode)
                   for mode in ["owned", "copy"]}
        medians["owned_reduction_vs_copy"] = 1 - medians["owned"] / medians["copy"]
        report["medians"][label] = medians
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["medians"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
