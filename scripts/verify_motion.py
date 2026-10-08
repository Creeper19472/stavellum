"""Render a bounded before/after clip and measure score motion without recording a window.

Run with PYTHONPATH pointing at another checkout to capture that version. Projects
and source audio are read only; presentation overrides apply to the in-memory copy.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

from stavellum._frame import FrameEvaluator
from stavellum.models import load_document
from stavellum.render import FrameRenderer
from stavellum.scene import compile_scene


def statistics(values: list[float]) -> dict[str, float]:
    ordered = sorted(abs(value) for value in values)
    if not ordered:
        return {"rms": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "rms": math.sqrt(sum(value * value for value in ordered) / len(ordered)),
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * .95) - 1)],
        "max": ordered[-1],
    }


def motion_metrics(scene) -> dict:
    with FrameEvaluator(scene) as evaluator:
        return _motion_metrics(scene, evaluator)


def _motion_metrics(scene, evaluator) -> dict:
    fps = 60
    duration = max(0.0, scene.score_duration + scene.settings.score_start_in_audio_sec)
    count = math.ceil(duration * fps) + 1
    delay = scene.settings.intro_delay_seconds
    exact = [scene.axis.x_at(scene.beat_at_time(index / fps)) for index in range(count)]
    camera = [evaluator.evaluate(delay + index / fps).world_x for index in range(count)]

    def movement(velocities):
        changes = [right - left for left, right in zip(velocities, velocities[1:])]
        return {"velocity_px_per_second": statistics(velocities),
                "frame_velocity_change_px_per_second": statistics(changes),
                "acceleration_px_per_second_squared": statistics([value * fps for value in changes])}

    def pan(points):
        return movement([(right - left) * scene.scale * fps
                         for left, right in zip(points, points[1:])])

    max_drift = 0.0
    reverse_frames = 0
    max_zoom_ratio = 0.0
    deadline_exceptions = 0
    max_normal_zoom_ratio = 0.0
    right_flow = []
    left_flow = []
    current_beat_positions = []
    urgent = getattr(scene.layout, "urgent_intervals", [])
    for index in range(count - 1):
        time_now = delay + index / fps
        first = evaluator.evaluate(time_now).layout
        last = evaluator.evaluate(time_now + 1 / fps).layout
        max_drift = max(max_drift, abs(exact[index] - camera[index]) * first.scale)
        pan_delta = (camera[index + 1] - camera[index]) * last.scale
        zoom_delta = (scene.body_right - scene.play_x) * (last.scale / first.scale - 1)
        right_flow.append((zoom_delta - pan_delta) * fps)
        left_flow.append(((scene.body_left - scene.play_x) * (last.scale / first.scale - 1)
                          - pan_delta) * fps)
        current_beat_positions.append(scene.play_x + (exact[index] - camera[index]) * first.scale)
        if pan_delta > 1e-9:
            ratio = abs(zoom_delta) / pan_delta
            max_zoom_ratio = max(max_zoom_ratio, ratio)
            deadline_exceptions += ratio > .505
            if not any(start <= time_now <= end + 1 / fps for start, end in urgent):
                max_normal_zoom_ratio = max(max_normal_zoom_ratio, ratio)
        reverse_frames += zoom_delta - pan_delta > .01
    return {
        "exact_axis_pan": pan(exact), "display_camera_pan": pan(camera),
        "rendered_right_edge_motion": movement(right_flow),
        "rendered_left_edge_motion": movement(left_flow),
        "current_beat_screen_x_range": [min(current_beat_positions), max(current_beat_positions)],
        "metrics_audio_duration_seconds": duration,
        "camera_half_window_seconds": getattr(getattr(scene, "camera", None), "half_window_seconds", None),
        "maximum_drift_at_maximum_scale_px": max(abs(a - b) * scene.scale for a, b in zip(exact, camera)),
        "maximum_drift_at_display_scale_px": max_drift,
        "right_edge_zoom_to_pan_ratio_max": max_zoom_ratio,
        "normal_right_edge_zoom_to_pan_ratio_max": max_normal_zoom_ratio,
        "urgent_layout_intervals": urgent,
        "layout_recovery_intervals": getattr(scene.layout, "recovery_intervals", []),
        "frames_exceeding_normal_zoom_limit": deadline_exceptions,
        "right_edge_reverse_frames": reverse_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--hide-announcement", action="store_true")
    parser.add_argument("--intro", type=float, default=0)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.seconds <= 0 or args.intro < 0:
        parser.error("start/intro must be nonnegative and seconds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    document = load_document(args.project)
    document.settings.width, document.settings.height, document.settings.fps = 1920, 1080, 60
    document.settings.intro_delay_seconds = args.intro
    if args.hide_announcement:
        document.settings.announcement_auto_hide = True
        document.settings.announcement_hold_seconds = 2
    started = time.perf_counter()
    scene = compile_scene(document)
    summary = motion_metrics(scene)
    summary["compile_and_measure_seconds"] = time.perf_counter() - started
    summary["start_seconds"] = args.start
    summary["duration_seconds"] = args.seconds
    if not args.metrics_only:
        renderer = FrameRenderer(scene)
        encoder = shutil.which("ffmpeg")
        if not encoder:
            raise RuntimeError("FFmpeg is required for a verification clip")
        frames = math.ceil(args.seconds * 60)
        duration = frames / 60
        command = [encoder, "-v", "error", "-nostdin", "-y", "-f", "rawvideo",
                   "-pixel_format", "rgba", "-video_size", "1920x1080", "-framerate", "60",
                   "-i", "pipe:0"]
        audio_seek = max(0.0, args.start - args.intro)
        silent_prefix = max(0.0, args.intro - args.start)
        if document.audio_path:
            command += ["-ss", str(audio_seek), "-i", document.audio_path,
                        "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "256k",
                        "-af", f"adelay={silent_prefix * 1000:.6f}:all=1,apad"]
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-t", str(duration), "-movflags", "+faststart",
                    str(args.output / "clip.mp4")]
        with (args.output / "encode.log").open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log, stdout=subprocess.DEVNULL,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                for index in range(frames):
                    frame = renderer.render_frame(args.start + index / 60)
                    process.stdin.write(frame.constBits())
                    if index in {0, frames // 2, frames - 1}:
                        if not frame.save(str(args.output / f"frame-{index:04d}.png")):
                            raise RuntimeError("Failed to save verification frame")
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("FFmpeg failed; inspect encode.log")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        summary.update(frames=frames, fps=60, cache_peak_bytes=renderer.cache_peak_bytes,
                       cache_limit_bytes=renderer.cache_limit)
    summary["total_wall_seconds"] = time.perf_counter() - started
    (args.output / "motion.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
