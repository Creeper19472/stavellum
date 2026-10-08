"""Serial render-only diagnostics for production Vulkan and explicit Git baselines.

Each repetition starts with a fresh renderer and cold tile/texture caches. PNG
saving and optional pixel comparisons happen after timing. This measures frame
production, not MP4 export throughput; no encoder or writer queue is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from PySide6.QtGui import QImage

from stavellum.models import load_document
from stavellum.qt import prepare_render_app
from stavellum.render import FrameRenderer
from stavellum.scene import compile_scene

_baseline_spec = importlib.util.spec_from_file_location("_benchmark_baseline", Path(__file__).with_name("_baseline.py"))
_baseline = importlib.util.module_from_spec(_baseline_spec)
_baseline_spec.loader.exec_module(_baseline)
load_package, baseline_class = _baseline.load_package, _baseline.renderer

ROOT = Path(__file__).resolve().parents[1]
MODES = ("rhi-vulkan", "cpu")
TILE_COUNTERS = ("tile_cache_hits", "tile_cache_misses", "tile_cache_evictions",
                 "svg_raster_seconds", "cache_peak_bytes", "cache_limit_bytes",
                 "visible_tile_working_peak_bytes")
DELTA_COUNTERS = ("scene_evaluation_count", "scene_evaluation_seconds", "scene_evaluation_cache_hits", "gpu_upload_bytes", "gpu_upload_seconds", "gpu_submit_seconds",
                  "synchronous_readback_seconds", "memory_copy_seconds",
                  "render_seconds", "asset_prepare_seconds", "command_build_seconds",
                  "command_pack_seconds", "native_begin_frame_seconds",
                  "gpu_texture_cache_hits", "gpu_texture_cache_misses", "gpu_texture_cache_evictions",
                  "owned_readback_frame_count", "copied_readback_frame_count", "memory_copy_bytes",
                  "inplace_format_conversion_seconds", "inplace_format_conversion_bytes")


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    return ordered[lower] + (ordered[math.ceil(position)] - ordered[lower]) * (position - lower)


def summary(samples):
    return {"p50_ms": percentile(samples, .5), "p95_ms": percentile(samples, .95),
            "p99_ms": percentile(samples, .99), "frame_samples_ms": samples}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_identity(report):
    """Ignore the OpenGL-specific PCIe/SSE suffix when comparing devices."""
    name = report["gpu_info"]["renderer"].split("/", 1)[0].strip().casefold()
    return name.removeprefix("nvidia corporation ")


@contextmanager
def native_library(path):
    """Keep the DLL selection scoped to one fully closed renderer."""
    name = "STAVELLUM_RHI_DLL"
    previous = os.environ.get(name)
    os.environ[name] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def baseline_renderer(ref, directory, current_native=False):
    """Load historical composition, assets and Python ABI in their own package."""
    package, commit = load_package(ref, directory, current_native=current_native)
    return baseline_class(package, rhi=True), commit


def backend_report(renderer):
    report = renderer.backend_report()
    # Historical RHI wrappers do not expose the asset renderer's tile counters.
    assets = getattr(renderer, "_assets", renderer)
    asset_report = report if assets is renderer else assets.backend_report()
    report["tile_cache_report"] = {name: asset_report.get(name) for name in TILE_COUNTERS}
    return report


def counter_delta(before, after):
    delta = {name: after[name] - before[name] for name in DELTA_COUNTERS
             if isinstance(before.get(name), (int, float))
             and isinstance(after.get(name), (int, float))}
    delta["tile_cache_report"] = {
        name: after["tile_cache_report"][name] - before["tile_cache_report"][name]
        for name in TILE_COUNTERS[:4]
        if isinstance(before["tile_cache_report"].get(name), (int, float))
        and isinstance(after["tile_cache_report"].get(name), (int, float))}
    return delta


def consume(renderer, times, capture=()):
    stream = renderer.export_frames(times, lambda: False)
    samples, snapshots = [], {}
    started = time.perf_counter()
    try:
        while True:
            frame_started = time.perf_counter()
            try:
                index, image = next(stream)
            except StopIteration:
                break
            samples.append((time.perf_counter() - frame_started) * 1000)
            if index != len(samples) - 1:
                raise RuntimeError("Frame stream delivered an unexpected index")
            if index in capture:
                snapshots[index] = image  # owned storage; retain without copying
        elapsed = time.perf_counter() - started
    finally:
        stream.close()
    return {"wall_seconds": elapsed, "pixel_format": stream.pixel_format,
            **summary(samples)}, snapshots


def profile(renderer, start, frames, warm_frames):
    result, snapshots = consume(renderer,
                                (start + index / renderer.scene.settings.fps for index in range(frames)),
                                (0, frames // 2, frames - 1))
    if len(result["frame_samples_ms"]) != frames:
        raise RuntimeError("Frame stream did not produce the requested frame count")
    result["cold_frame_ms"] = result["frame_samples_ms"][0]
    result["remaining_p95_ms"] = percentile(result["frame_samples_ms"][1:] or result["frame_samples_ms"], .95)
    result["backend_report"] = backend_report(renderer)
    # Warm only this fixed timestamp, once. Do not prewarm future playback tiles.
    consume(renderer, [start])
    before = backend_report(renderer)
    warm, _ = consume(renderer, (start for _ in range(warm_frames)))
    after = backend_report(renderer)
    result["fixed_warm"] = {**warm, "counter_delta": counter_delta(before, after)}
    return result, snapshots


def pixel_comparison(reference, candidate, difference):
    import numpy as np

    images = [QImage(str(path)).convertToFormat(QImage.Format.Format_RGBA8888)
              for path in (reference, candidate)]
    if any(image.isNull() for image in images) or images[0].size() != images[1].size():
        raise RuntimeError("Cannot compare missing or differently sized PNGs")
    arrays = [np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4)
              for image in images]
    delta = np.abs(arrays[0].astype(np.int16) - arrays[1].astype(np.int16))
    rgba = np.empty(arrays[0].shape, dtype=np.uint8)
    rgba[:, :, :3] = np.minimum(delta[:, :, :3] * 8, 255)
    rgba[:, :, 3] = 255
    image = QImage(rgba.data, images[0].width(), images[0].height(), images[0].width() * 4,
                   QImage.Format.Format_RGBA8888)
    if not image.save(str(difference)):
        raise OSError("Cannot save pixel difference PNG")
    return {"reference": reference.name, "candidate": candidate.name,
            "exact_rgba_equal": bool(np.all(delta == 0)),
            "rgb_mae_255": float(delta[:, :, :3].mean()),
            "max_channel_error_255": int(delta.max()),
            "opaque_alpha": bool(np.all(arrays[1][:, :, 3] == 255)),
            "difference_x8": difference.name}


def write_report(path, report):
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rhi-profile"))
    parser.add_argument("--baseline-ref")
    parser.add_argument("--baseline-dll", type=Path)
    parser.add_argument("--baseline-current-native", action="store_true",
                        help="Use baseline composition with current Python ABI and native DLL")
    parser.add_argument("--modes", nargs="+", choices=(*MODES, "baseline-opengl", "baseline-rhi-vulkan"))
    parser.add_argument("--sizes", nargs="+", default=["1920x1080"])
    parser.add_argument("--starts", nargs="+", type=float, default=[0, 82, 96.25])
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warm-frames", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--quality", action="store_true", help="Compare final-round PNGs and save differences")
    args = parser.parse_args()
    if min(args.frames, args.warm_frames, args.repeats) < 1 or any(
            start < 0 or not math.isfinite(start) for start in args.starts):
        parser.error("frame/repeat counts must be positive; starts finite and nonnegative")
    if args.baseline_current_native and (not args.baseline_ref or args.baseline_dll):
        parser.error("--baseline-current-native requires --baseline-ref and excludes --baseline-dll")
    if (args.baseline_dll or args.baseline_current_native) and not args.baseline_ref:
        parser.error("native baseline options require --baseline-ref")
    if args.modes and (len(set(args.modes)) != len(args.modes) or any(
            mode.startswith("baseline-") and not args.baseline_ref for mode in args.modes)):
        parser.error("modes must be unique; baseline modes require --baseline-ref")
    if os.environ.get("STAVELLUM_RHI_TIMESTAMPS") not in (None, "", "0"):
        parser.error("Disable GPU timestamps for performance runs")
    try:
        sizes = [tuple(map(int, size.lower().split("x"))) for size in args.sizes]
        if any(len(size) != 2 or min(size) < 1 for size in sizes):
            raise ValueError
    except ValueError:
        parser.error("sizes must use positive WIDTHxHEIGHT")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "profile.json"
    packaged_dll = ROOT / "src/stavellum/native/rhi/stavellum_rhi.dll"
    default_dll = packaged_dll if packaged_dll.exists() else ROOT / ".cache/rhi/build/stavellum_rhi.dll"
    current_dll = Path(os.environ.get("STAVELLUM_RHI_DLL", str(default_dll))).resolve()
    report = {"project": args.project.name, "status": "initializing", "repeats": args.repeats,
              "frames": args.frames, "warm_frames": args.warm_frames,
              "baseline_ref": args.baseline_ref, "results": [], "medians": {}, "quality": [],
              "baseline_native": "current" if args.baseline_current_native else "baseline",
              "forced_rgba_readback": os.environ.get("STAVELLUM_RHI_RGBA_READBACK", "0"),
              "forced_copy_readback": os.environ.get("STAVELLUM_RHI_COPY_READBACK", "0"),
              "limitations": "Frame production only; no FFmpeg/writer. Stage times may contain other stages.",
              "baseline_assets": "Historical assets and ABI; the current compiled scene is shared",
              "snapshots": "First/middle/last frames of the final round, saved outside timing"}
    write_report(destination, report)
    try:
        report["source_sha256"] = {name: fingerprint(ROOT / path) for name, path in {
            "render": "src/stavellum/render.py", "rhi": "src/stavellum/rhi.py",
            "rhi_abi": "src/stavellum/_rhi.py", "native": "native/rhi/rhi.cpp"}.items()}
        modes = args.modes or ["rhi-vulkan"]
        if args.baseline_ref and not args.modes:
            modes += ["baseline-opengl"]
            if args.baseline_dll or args.baseline_current_native:
                modes += ["baseline-rhi-vulkan"]
        if "baseline-rhi-vulkan" in modes and not (args.baseline_dll or args.baseline_current_native):
            parser.error("baseline-rhi-vulkan requires --baseline-dll or --baseline-current-native")
        if any("rhi" in mode for mode in modes):
            report["current_dll_sha256"] = fingerprint(current_dll)
        baseline = baseline_opengl = None
        if args.baseline_ref:
            package, report["baseline_commit"] = load_package(
                args.baseline_ref, directory / "baseline-source", current_native=args.baseline_current_native)
            baseline_opengl = baseline_class(package)
            if "baseline-rhi-vulkan" in modes:
                baseline = baseline_class(package, rhi=True)
                args.baseline_dll = current_dll if args.baseline_current_native else args.baseline_dll.resolve()
                report["baseline_dll_sha256"] = fingerprint(args.baseline_dll)
        report["modes"] = modes
        for width, height in sizes:
            document = load_document(args.project)
            document.settings = replace(document.settings, width=width, height=height, fps=60,
                                        render_backend="gpu", cache_megabytes=64)
            document.validate()
            prepare_render_app(document.settings)
            started = time.perf_counter()
            scene = compile_scene(document)
            report.setdefault("compile_seconds", {})[f"{width}x{height}"] = time.perf_counter() - started
            for start in args.starts:
                label = f"{width}x{height}-at-{start:g}"
                for repeat in range(args.repeats):
                    rotation = repeat % len(modes)
                    for mode in modes[rotation:] + modes[:rotation]:
                        dll = args.baseline_dll if mode == "baseline-rhi-vulkan" else current_dll
                        started = time.perf_counter()
                        with native_library(dll):
                            if mode == "baseline-opengl":
                                renderer = baseline_opengl(scene, export_readback="sync")
                            elif mode == "baseline-rhi-vulkan":
                                renderer = baseline(scene, api="vulkan")
                            else:
                                backend = "cpu" if mode == "cpu" else "gpu"
                                renderer = FrameRenderer(replace(scene, settings=replace(scene.settings, render_backend=backend)))
                            with renderer:
                                initialization = time.perf_counter() - started
                                result, snapshots = profile(renderer, start, args.frames, args.warm_frames)
                        result.update(case=label, mode=mode, repeat=repeat + 1,
                                      width=width, height=height, start_seconds=start,
                                      frames=args.frames, fps=60, initialization_seconds=initialization)
                        backend = result["backend_report"]
                        requested = "cpu" if mode == "cpu" else "gpu"
                        if backend["render_backend"] != requested or backend.get("render_fallback_reasons"):
                            raise RuntimeError("A profile mode unexpectedly fell back from its requested GPU backend")
                        if mode != "cpu" and backend["gpu_info"]["msaa_samples"] != 4:
                            raise RuntimeError("Profile requires the same 4xMSAA for every backend")
                        if mode != "cpu":
                            identity = gpu_identity(backend)
                            if not identity:
                                raise RuntimeError("Profile backend did not identify its GPU")
                            if identity != report.setdefault("gpu_identity", identity):
                                raise RuntimeError("Profile runs must use the same GPU")
                        if repeat == args.repeats - 1:
                            for index, image in snapshots.items():
                                path = directory / f"{label}-{mode}-{index:04d}.png"
                                if not image.save(str(path)):
                                    raise OSError("Cannot save profile snapshot")
                        snapshots.clear()
                        report["results"].append(result)
                        report["status"] = "profiling"
                        write_report(destination, report)
                        print(json.dumps({key: result[key] for key in (
                            "case", "mode", "repeat", "wall_seconds", "cold_frame_ms", "remaining_p95_ms")}), flush=True)
                report["medians"][label] = {
                    mode: {name: statistics.median(row[name] for row in report["results"]
                                                  if row["case"] == label and row["mode"] == mode)
                           for name in ("wall_seconds", "cold_frame_ms", "remaining_p95_ms")}
                    for mode in modes}
                for mode in modes:
                    rows = [row for row in report["results"] if row["case"] == label and row["mode"] == mode]
                    report["medians"][label][mode]["fixed_warm_p95_ms"] = statistics.median(
                        row["fixed_warm"]["p95_ms"] for row in rows)
                if args.quality:
                    pairs = [(mode, "rhi-vulkan") for mode in modes if mode != "rhi-vulkan"]
                    for reference_mode, candidate_mode in pairs:
                        if reference_mode not in modes or candidate_mode not in modes:
                            continue
                        for index in sorted({0, args.frames // 2, args.frames - 1}):
                            reference = directory / f"{label}-{reference_mode}-{index:04d}.png"
                            candidate = directory / f"{label}-{candidate_mode}-{index:04d}.png"
                            difference = directory / f"{label}-{reference_mode}-vs-{candidate_mode}-{index:04d}-difference-x8.png"
                            report["quality"].append({"case": label, "frame": index,
                                                      **pixel_comparison(reference, candidate, difference)})
                write_report(destination, report)
        report["status"] = "complete"
    except BaseException as error:
        report["status"], report["error"] = "failed", str(error)
        raise
    finally:
        write_report(destination, report)
    print(json.dumps({"report": str(destination), "medians": report["medians"]}), flush=True)


if __name__ == "__main__":
    main()
