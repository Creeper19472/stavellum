"""Compare the saved synchronous RHI package with export batches, serially.

The coordinator imports no Qt modules. Each GPU run uses a fresh subprocess,
an explicit source package and its matching DLL. Local cases share one pickled
current compiled scene; full exports use each package's complete export chain.
Run quality, inspect its PNGs, then run local with --quality-reviewed. Full runs
require three complete local rounds and the selected candidate's quality gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import pickle
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "batch1", "batch2", "batch4")
BATCH_SIZES = {"baseline": 1, "batch1": 1, "batch2": 2, "batch4": 4}
CASES = {
    "intro": (1920, 1080, 0.0),
    "dense": (1920, 1080, 82.0),
    "transition": (1920, 1080, 96.25),
    "1440p": (2560, 1440, 63.0),
    "4k": (3840, 2160, 63.0),
}
DEFAULT_PROJECTS = (
    ROOT / "artifacts/demo/demo.stproj",
    ROOT / "artifacts/demo-2/demo.stproj",
)
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def package_source(path):
    """Accept a checkout, a src directory, or a saved package directory."""
    path = Path(path).resolve()
    for candidate in (path / "src", path, path.parent):
        if (candidate / "stavellum/__init__.py").is_file():
            return candidate
    raise FileNotFoundError("Cannot find stavellum package under " + str(path))


def package_fingerprints(source, dll):
    package = Path(source) / "stavellum"
    files = {str(path.relative_to(package)).replace("\\", "/"): fingerprint(path)
             for path in sorted(package.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts
             and path.suffix not in (".pyc", ".pyo")}
    return {"source": str(source), "files": files, "dll": str(dll),
            "dll_sha256": fingerprint(dll)}


def project_fingerprints(project):
    project = Path(project).resolve()
    payload = json.loads(project.read_text(encoding="utf-8-sig"))
    audio = Path(payload["audio_path"])
    if not audio.is_absolute():
        audio = project.parent / audio
    audio = audio.resolve()
    if not audio.is_file():
        raise FileNotFoundError("Project audio is missing: " + str(audio))
    return {"project": str(project), "project_sha256": fingerprint(project),
            "audio": str(audio), "audio_sha256": fingerprint(audio)}


def alternating_modes(modes, round_number):
    offset = (round_number - 1) % len(modes)
    return (*modes[offset:], *modes[:offset])


def quality_times(start, duration=8, fps=60):
    # Consecutive frames exercise batch 4 and a final incomplete batch; the
    # jumps cover different layouts and cache histories within the same case.
    return [start + i / fps for i in range(5)] + [
        start + duration / 2 + i / fps for i in range(4)] + [
        start + duration - (4 - i) / fps for i in range(4)]


def medians_and_gate(results, quality_passed, *, cases=tuple(CASES), rounds=3):
    medians = {}
    complete = True
    for case in cases:
        medians[case] = {}
        for mode in MODES:
            rows = [row for row in results if row["case"] == case and row["mode"] == mode]
            if {row["repeat"] for row in rows} != set(range(1, rounds + 1)):
                complete = False
            if rows:
                medians[case][mode] = statistics.median(row["pipeline_seconds"] for row in rows)
    candidates = {}
    if complete:
        baseline = sum(medians[case]["baseline"] for case in cases)
        for mode in ("batch2", "batch4"):
            total = sum(medians[case][mode] for case in cases)
            gain = 1 - total / baseline
            regressions = [case for case in cases
                           if medians[case][mode] > medians[case]["baseline"] * 1.05]
            candidates[mode] = {"aggregate_export_reduction": gain,
                                "regressing_cases_over_5_percent": regressions,
                                "passed": gain >= .05 - 1e-12 and not regressions,
                                "aggregate_seconds": total}
    eligible = [mode for mode, candidate in candidates.items() if candidate["passed"]]
    selected = min(eligible, key=lambda mode: candidates[mode]["aggregate_seconds"]) if eligible else None
    relative_difference = None
    if all(mode in candidates for mode in ("batch2", "batch4")):
        relative_difference = abs(candidates["batch2"]["aggregate_seconds"]
                                  - candidates["batch4"]["aggregate_seconds"]) / min(
                                      candidates["batch2"]["aggregate_seconds"],
                                      candidates["batch4"]["aggregate_seconds"])
        if "batch2" in eligible and "batch4" in eligible and relative_difference < .03:
            selected = "batch2"
    return medians, {"complete_three_rounds": complete, "quality_passed": quality_passed,
                     "candidates": candidates, "batch2_batch4_relative_difference": relative_difference,
                     "selected_mode": selected,
                     "full_song_gate_passed": complete and quality_passed and selected is not None,
                     "rules": "At least 5% aggregate reduction, no case over 5% slower; choose 2 if 2/4 differ by under 3%."}


def verify_mp4(ffmpeg, ffprobe, output, frames, width, height):
    """Decode video and audio, then inspect decoded frame count and stream shape."""
    started = time.perf_counter()
    decoded = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(output),
                              "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
                             capture_output=True, check=True, creationflags=NO_WINDOW)
    if decoded.stderr.strip():
        raise RuntimeError("MP4 decode reported errors: " + decoded.stderr.decode("utf-8", "replace"))
    probe = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_entries",
                            "stream=codec_type,codec_name,nb_read_frames,width,height,avg_frame_rate,channels,duration",
                            "-of", "json", str(output)], capture_output=True, check=True,
                           creationflags=NO_WINDOW)
    streams = json.loads(probe.stdout)["streams"]
    video = next(row for row in streams if row["codec_type"] == "video")
    audio = next(row for row in streams if row["codec_type"] == "audio")
    actual = (int(video["nb_read_frames"]), video["width"], video["height"],
              video["avg_frame_rate"], video["codec_name"], audio["codec_name"], audio["channels"])
    if actual != (frames, width, height, "60/1", "h264", "aac", 2):
        raise RuntimeError("MP4 frame/size/fps/stereo contract failed: " + str(actual))
    return {"decoded_successfully": True, "streams": streams,
            "verification_seconds": time.perf_counter() - started,
            "sha256": fingerprint(output)}


def helpers():
    spec = importlib.util.spec_from_file_location("_rhi_batch_vulkan_helpers",
                                                Path(__file__).with_name("verify_vulkan.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_document_for_run(project, width, height):
    from stavellum.domain.models import load_document

    document = load_document(project)
    document.settings = replace(document.settings, width=width, height=height, fps=60,
                                render_backend="gpu", cache_megabytes=64,
                                video_encoder="h264_nvenc", nvenc_cq=18, nvenc_preset="p5")
    if not Path(document.audio_path).is_file():
        raise FileNotFoundError("Project audio is missing: " + document.audio_path)
    document.validate()
    return document


def check_backend(report, frames):
    helper = helpers()
    if report["render_backend"] != "gpu" or report.get("render_fallback_reasons"):
        raise RuntimeError("GPU renderer unexpectedly fell back")
    if report["gpu_info"]["msaa_samples"] != 4:
        raise RuntimeError("All comparison groups must use 4xMSAA")
    if report["rendered_frame_count"] != frames:
        raise RuntimeError("Renderer frame count differs from the requested count")
    return helper.gpu_identity(report)


def compiled_worker(spec):
    from stavellum.graphics.qt import prepare_render_app
    from stavellum.presentation.scene import compile_scene

    document = load_document_for_run(spec["project"], spec["width"], spec["height"])
    prepare_render_app(document.settings)
    started = time.perf_counter()
    scene = compile_scene(document)
    elapsed = time.perf_counter() - started
    path = Path(spec["scene"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(scene, protocol=pickle.HIGHEST_PROTOCOL))
    return {"scene": str(path), "scene_sha256": fingerprint(path), "compile_seconds": elapsed,
            "settings": asdict(document.settings), "score_duration": scene.score_duration,
            "audio": document.audio_path, "audio_sha256": fingerprint(document.audio_path),
            "project_sha256": fingerprint(spec["project"])}


def read_scene(spec):
    if fingerprint(spec["scene"]) != spec["scene_sha256"]:
        raise RuntimeError("Shared compiled scene fingerprint changed")
    # The pickle is created locally by this script, never from external inputs.
    with Path(spec["scene"]).open("rb") as source:
        return pickle.load(source)


def encoding_worker(spec, scene, duration):
    from stavellum.exporting.encoding import probe_nvenc, video_arguments
    from stavellum.exporting.export import _encode_attempt
    from stavellum.rendering.render import FrameRenderer

    helper = helpers()
    available, reason = probe_nvenc(spec["ffmpeg"], scene.settings, lambda: False)
    if not available:
        raise RuntimeError("NVENC unavailable: " + reason)
    frames = math.ceil(duration * scene.settings.fps)
    duration = frames / scene.settings.fps
    started = time.perf_counter()
    with FrameRenderer(scene) as renderer:
        initialization = time.perf_counter() - started
        started = time.perf_counter()
        last_update = [started]

        def progress(fraction, message):
            if time.perf_counter() - last_update[0] >= 20:
                print(json.dumps({"case": spec["case"], "mode": spec["mode"],
                                  "progress": fraction}), flush=True)
                last_update[0] = time.perf_counter()

        encoding_scene = replace(scene, settings=replace(scene.settings, intro_delay_seconds=0))
        measurements = _encode_attempt(spec["ffmpeg"], "h264_nvenc", encoding_scene,
                                       helper.OffsetRenderer(renderer, spec["start"]),
                                       spec["audio"], spec["video"], frames, duration,
                                       progress, lambda: False)
        elapsed = time.perf_counter() - started
        backend = renderer.backend_report()
    gpu = check_backend(backend, frames)
    verification = verify_mp4(spec["ffmpeg"], spec["ffprobe"], spec["video"], frames,
                              scene.settings.width, scene.settings.height)
    return {**backend, **measurements, "pipeline_seconds": elapsed,
            "initialization_seconds": initialization, "frames": frames, "fps": 60,
            "width": scene.settings.width, "height": scene.settings.height,
            "encoder_parameters": video_arguments(scene.settings, "h264_nvenc"),
            "gpu_identity": gpu, "mp4": verification}


def quality_worker(spec, scene):
    from PySide6.QtGui import QImage

    from stavellum.rendering.render import FrameRenderer

    times = quality_times(spec["start"], spec["duration"])
    directory = Path(spec["result"]).parent
    baseline = spec["mode"] == "baseline"
    raw = directory / "frames.rgba.gz" if baseline else Path(spec["reference_raw"])
    frames = []
    size = scene.settings.width * scene.settings.height * 4
    with gzip.open(raw, "wb", compresslevel=1) if baseline else gzip.open(raw, "rb") as reference:
        with FrameRenderer(scene) as renderer:
            stream = renderer.export_frames(times, lambda: False)
            try:
                for expected_index, (index, image) in enumerate(stream):
                    if index != expected_index or index >= len(times):
                        raise RuntimeError("Quality stream delivered an unexpected index")
                    canonical = image.convertToFormat(QImage.Format.Format_RGBA8888)
                    pixels = bytes(canonical.constBits())
                    if len(pixels) != size:
                        raise RuntimeError("Unexpected raw quality image size")
                    if baseline:
                        reference.write(pixels)
                        expected = pixels
                    else:
                        expected = reference.read(size)
                    same = expected == pixels
                    row = {"index": index, "seconds": times[index], "bytes": size,
                           "sha256": hashlib.sha256(pixels).hexdigest(), "raw_bytes_equal": same}
                    if not same:
                        row["first_different_byte"] = next(
                            (i for i, (left, right) in enumerate(zip(expected, pixels)) if left != right),
                            min(len(expected), len(pixels)))
                    if index in (0, len(times) // 2, len(times) - 1):
                        image_path = directory / f"frame-{index:02d}.png"
                        if not canonical.save(str(image_path)):
                            raise RuntimeError("Cannot save quality PNG")
                        row["png"] = str(image_path)
                    frames.append(row)
                if len(frames) != len(times) or (not baseline and reference.read(1)):
                    raise RuntimeError("Raw quality reference/frame count mismatch")
            finally:
                stream.close()
            backend = renderer.backend_report()
    gpu = check_backend(backend, len(times))
    # Use a separate fresh renderer for the representative encoded clip.
    encoded = encoding_worker(spec, scene, 1.0)
    if encoded["gpu_identity"] != gpu:
        raise RuntimeError("Quality raw frames and encoded clip used different GPUs")
    return {"samples": frames, "raw_reference": str(raw), "raw_reference_sha256": fingerprint(raw),
            "exact_raw_bytes_equal": all(row["raw_bytes_equal"] for row in frames),
            "raw_backend_report": backend, "gpu_identity": gpu,
            "encoded_clip": encoded, "mp4": encoded["mp4"]}


def full_worker(spec):
    from stavellum.exporting.export import export_video

    document = load_document_for_run(spec["project"], 2560, 1440)
    started = time.perf_counter()
    last_update = [started]

    def progress(fraction, message):
        if time.perf_counter() - last_update[0] >= 20:
            print(json.dumps({"project": spec["project"], "mode": spec["mode"],
                              "repeat": spec["repeat"], "progress": fraction}), flush=True)
            last_update[0] = time.perf_counter()

    export_video(document, spec["video"], progress=progress)
    external = time.perf_counter() - started
    summary_path = Path(spec["video"]).with_suffix(".render.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gpu = check_backend(summary, summary["frames"])
    if summary["video_encoder"] != "h264_nvenc" or summary.get("encoder_fallback_reasons"):
        raise RuntimeError("Full export unexpectedly fell back from NVENC")
    return {**summary, "pipeline_seconds": summary["wall_seconds"],
            "export_call_seconds": external, "gpu_identity": gpu,
            "project_sha256": fingerprint(spec["project"]),
            "audio_sha256": fingerprint(document.audio_path), "settings": asdict(document.settings),
            "report_path": str(summary_path),
            "mp4": verify_mp4(spec["ffmpeg"], spec["ffprobe"], spec["video"], summary["frames"],
                               document.settings.width, document.settings.height)}


def worker_main(kind, spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    sys.path.insert(0, spec["source"])
    result = {"status": "running", **{key: spec[key] for key in
              ("project", "mode", "source", "dll", "fingerprints") if key in spec}}
    result["requested_batch_size"] = BATCH_SIZES[spec["mode"]]
    try:
        import stavellum

        if not Path(stavellum.__file__).resolve().is_relative_to(Path(spec["source"])):
            raise RuntimeError("Worker imported the wrong source package")
        if package_fingerprints(spec["source"], spec["dll"]) != spec["fingerprints"]:
            raise RuntimeError("Source/DLL changed before worker started")
        if spec.get("inputs") and project_fingerprints(spec["project"]) != spec["inputs"]:
            raise RuntimeError("Project/audio changed before worker started")
        if spec.get("audio_sha256") and fingerprint(spec["audio"]) != spec["audio_sha256"]:
            raise RuntimeError("Prepared audio fingerprint changed before worker started")
        if kind == "compile":
            result.update(compiled_worker(spec))
        elif kind == "full":
            result.update(full_worker(spec))
        else:
            scene = read_scene(spec)
            result.update(quality_worker(spec, scene) if kind == "quality"
                          else encoding_worker(spec, scene, spec["duration"]))
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_report(spec["result"], result)


def run_worker(kind, spec):
    result_path = Path(spec["result"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path = result_path.with_suffix(".worker.json")
    write_report(spec_path, spec)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=spec["source"], STAVELLUM_RHI_DLL=spec["dll"],
                       STAVELLUM_RHI_BATCH_SIZE=str(BATCH_SIZES[spec["mode"]]),
                       STAVELLUM_RHI_TIMESTAMPS="0", STAVELLUM_RHI_COPY_READBACK="0",
                       STAVELLUM_RHI_RGBA_READBACK="0", PYTHONIOENCODING="utf-8")
    started = time.perf_counter()
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", kind,
                              "--worker-spec", str(spec_path)], env=environment, cwd=ROOT,
                             creationflags=NO_WINDOW)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if process.returncode or result["status"] != "complete":
        raise RuntimeError("Isolated benchmark worker failed: " + result.get("error", str(process.returncode)))
    result["worker_process_seconds"] = time.perf_counter() - started
    result.update({key: spec[key] for key in ("case", "repeat", "start", "scene_sha256") if key in spec})
    write_report(result_path, result)
    return result


def update_result(report, stage, result):
    rows = report.setdefault(stage, [])
    identity = tuple(result.get(key) for key in ("project", "case", "mode", "repeat"))
    rows[:] = [row for row in rows if tuple(row.get(key) for key in
                                          ("project", "case", "mode", "repeat")) != identity]
    rows.append(result)


def full_summary(results, selected_mode, quality_passed, *, projects=DEFAULT_PROJECTS):
    modes = ("baseline", selected_mode) if selected_mode else ("baseline",)
    medians = {}
    complete = selected_mode is not None
    for project in projects:
        project = str(Path(project).resolve())
        medians[project] = {}
        for mode in modes:
            rows = [row for row in results if row["project"] == project and row["mode"] == mode]
            if {row["repeat"] for row in rows} != {1, 2, 3}:
                complete = False
            if rows:
                medians[project][mode] = {key: statistics.median(row[key] for row in rows)
                                          for key in ("pipeline_seconds", "total_wall_seconds", "compile_seconds")}
        counts = {row["frames"] for row in results if row["project"] == project}
        if len(counts) > 1:
            raise RuntimeError("Full comparison groups produced different frame counts")
    reduction, regressions = None, []
    if complete:
        baseline = sum(row["baseline"]["pipeline_seconds"] for row in medians.values())
        candidate = sum(row[selected_mode]["pipeline_seconds"] for row in medians.values())
        reduction = 1 - candidate / baseline
        regressions = [project for project, row in medians.items()
                       if row[selected_mode]["pipeline_seconds"] > row["baseline"]["pipeline_seconds"] * 1.05]
    return medians, {"complete_three_rounds": complete, "selected_mode": selected_mode,
                     "aggregate_export_reduction": reduction,
                     "regressing_projects_over_5_percent": regressions,
                     "passed": complete and quality_passed and reduction >= .05 - 1e-12 and not regressions,
                     "comparison_metric": "Export report wall_seconds, excluding compile and renderer initialization"}


def summarize(report):
    quality = report.get("quality", [])
    expected = {(case, mode) for case in CASES for mode in MODES}
    quality_complete = expected <= {(row["case"], row["mode"]) for row in quality}
    exact = quality_complete and all(row["exact_raw_bytes_equal"] and row["mp4"]["decoded_successfully"]
                                    for row in quality)
    gpu_names = {row["gpu_identity"] for stage in ("quality", "local", "full")
                 for row in report.get(stage, [])}
    if len(gpu_names) > 1:
        raise RuntimeError("Comparison runs used different hardware GPUs")
    for case in CASES:
        rows = [row for stage in ("quality", "local") for row in report.get(stage, []) if row["case"] == case]
        for key in ("scene_sha256", "audio_sha256", "project_sha256"):
            values = {row[key] for row in rows if key in row}
            if len(values) > 1:
                raise RuntimeError("Comparison inputs differ for " + case + ": " + key)
    report["quality_gate"] = {"complete": quality_complete, "exact_raw_bytes_equal": exact,
                              "visual_review_recorded": report.get("visual_review_recorded", False),
                              "passed": exact and report.get("visual_review_recorded", False)}
    report["medians"], report["gate"] = medians_and_gate(
        report.get("local", []), report["quality_gate"]["passed"])
    report["full_medians"], report["full_gate"] = full_summary(
        report.get("full", []), report["gate"]["selected_mode"], report["gate"]["full_song_gate_passed"],
        projects=report.get("full_projects", DEFAULT_PROJECTS))


def scene_for_case(project, width, height, directory, current):
    label = f"{project.stem}-{width}x{height}"
    metadata = directory / "scenes" / label / "compiled.json"
    inputs = project_fingerprints(project)
    identity = {"inputs": inputs, "width": width, "height": height,
                "fingerprints": current}
    if metadata.exists():
        previous = json.loads(metadata.read_text(encoding="utf-8"))
        if (previous.get("identity") == identity and Path(previous["scene"]).is_file()
                and fingerprint(previous["scene"]) == previous["scene_sha256"]):
            return previous
    spec = {"project": str(project), "width": width, "height": height,
            "source": current["source"], "dll": current["dll"], "fingerprints": current,
            "mode": "batch1", "inputs": inputs,
            "scene": str(metadata.with_name("scene.pkl")), "result": str(metadata)}
    result = run_worker("compile", spec)
    result["identity"] = identity
    write_report(metadata, result)
    return result


def case_audio(compiled, case, directory, ffmpeg):
    path = directory / "audio" / f"{case}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    start = CASES[case][2]
    delay = compiled["settings"]["intro_delay_seconds"] * 1000
    audio_filter = (f"adelay=delays={delay:.9f}:all=1,apad,"
                    f"atrim=start={start:.9f}:duration=8,asetpts=PTS-STARTPTS")
    subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-y", "-i", compiled["audio"],
                    "-af", audio_filter, "-t", "8", "-c:a", "pcm_s16le", str(path)],
                   capture_output=True, check=True, creationflags=NO_WINDOW)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("quality", "local", "full", "summary"), default="quality")
    parser.add_argument("--baseline", type=Path, default=ROOT / "artifacts/rhi-batch/baseline")
    parser.add_argument("--baseline-dll", type=Path)
    parser.add_argument("--current-source", type=Path, default=ROOT / "src")
    parser.add_argument("--current-dll", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/rhi-batch/verification")
    parser.add_argument("--project", type=Path, action="append", help="Local: one project. Full: repeat for both projects.")
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    parser.add_argument("--mode", choices=MODES, action="append")
    parser.add_argument("--round", type=int, choices=(1, 2, 3), action="append")
    parser.add_argument("--quality-reviewed", action="store_true")
    parser.add_argument("--selected-batch", type=int, choices=(2, 4))
    parser.add_argument("--worker", choices=("compile", "quality", "local", "full"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker_main(args.worker, args.worker_spec)
        return
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("FFmpeg and FFprobe are required")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "benchmark.json"
    packages = {}
    for mode, source, override in (("baseline", args.baseline, args.baseline_dll),
                                   ("current", args.current_source, args.current_dll)):
        source = package_source(source)
        dll = (override or source / "stavellum/native/rhi/stavellum_rhi.dll").resolve()
        packages[mode] = package_fingerprints(source, dll)
    versions = {"packages": packages, "python": sys.version,
                "benchmark_script_sha256": fingerprint(__file__),
                "helper_script_sha256": fingerprint(Path(__file__).with_name("verify_vulkan.py")),
                "native_sources": {name: fingerprint(ROOT / "native/rhi" / name)
                                   for name in ("rhi.cpp", "rhi.h")},
                "ffmpeg_version": subprocess.run([ffmpeg, "-version"], capture_output=True, check=True,
                                                  creationflags=NO_WINDOW).stdout.decode("utf-8", "replace").splitlines()[0]}
    report = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else {
        "versions": versions, "quality": [], "local": [], "full": [],
        "timing": "Local excludes compile/init/probe/decode. Full uses export report wall_seconds; total_wall_seconds is also retained.",
        "encoder": "h264_nvenc cq18 p5", "fps": 60, "cache_megabytes": 64, "msaa_samples": 4}
    if report["versions"] != versions:
        parser.error("Saved report fingerprints differ; choose a new --output to avoid mixing source/DLL revisions")
    if args.quality_reviewed:
        report["visual_review_recorded"] = True
    report["status"] = "running-" + args.stage
    write_report(destination, report)
    try:
        summarize(report)
        modes = tuple(dict.fromkeys(args.mode or MODES))
        rounds = tuple(dict.fromkeys(args.round or (1, 2, 3)))
        cases = tuple(dict.fromkeys(args.case or CASES))
        projects = tuple(path.resolve() for path in (args.project or
                         (DEFAULT_PROJECTS if args.stage == "full" else DEFAULT_PROJECTS[:1])))
        if args.stage in ("quality", "local"):
            if len(projects) != 1:
                raise ValueError("Use one --project for local/quality; separate outputs keep gates unambiguous")
            project = projects[0]
            if report.get("local_project") not in (None, str(project)):
                raise ValueError("Saved local/quality report belongs to a different project")
            inputs = project_fingerprints(project)
            if report.get("local_inputs") not in (None, inputs):
                raise ValueError("Saved local/quality project or audio changed; use a new --output")
            report["local_project"] = str(project)
            report["local_inputs"] = inputs
            for case in cases:
                width, height, start = CASES[case]
                compiled = scene_for_case(project, width, height, directory, packages["current"])
                audio = case_audio(compiled, case, directory, ffmpeg)
                for repeat in ((0,) if args.stage == "quality" else rounds):
                    order = modes if repeat == 0 else alternating_modes(modes, repeat)
                    for mode in order:
                        selected = packages["baseline" if mode == "baseline" else "current"]
                        target = directory / args.stage / case / (mode if repeat == 0 else f"round-{repeat}-{mode}")
                        spec = {"project": str(project), "source": selected["source"], "dll": selected["dll"],
                                "fingerprints": selected, "case": case, "mode": mode, "repeat": repeat,
                                "start": start, "duration": 8, "scene": compiled["scene"],
                                "scene_sha256": compiled["scene_sha256"], "audio": str(audio), "inputs": inputs,
                                "audio_sha256": fingerprint(audio), "ffmpeg": ffmpeg, "ffprobe": ffprobe,
                                "video": str(target / "output.mp4"), "result": str(target / "result.json")}
                        if args.stage == "quality" and mode != "baseline":
                            reference_result = directory / "quality" / case / "baseline/result.json"
                            reference = json.loads(reference_result.read_text(encoding="utf-8"))
                            if reference["status"] != "complete" or reference["scene_sha256"] != compiled["scene_sha256"]:
                                raise RuntimeError("Run matching baseline quality before candidate quality")
                            spec["reference_raw"] = reference["raw_reference"]
                        result = run_worker(args.stage, spec)
                        result.update(audio_sha256=spec["audio_sha256"], video=spec["video"],
                                      project_sha256=compiled["project_sha256"])
                        update_result(report, args.stage, result)
                        summarize(report)
                        write_report(destination, report)
                        print(json.dumps({"stage": args.stage, "case": case, "mode": mode,
                                          "repeat": repeat, "pipeline_seconds": result.get("pipeline_seconds"),
                                          "exact_raw_bytes_equal": result.get("exact_raw_bytes_equal")}), flush=True)
        elif args.stage == "full":
            if not report["gate"]["full_song_gate_passed"]:
                raise RuntimeError("Full export requires reviewed exact quality and passing complete local gate")
            selected_mode = report["gate"]["selected_mode"]
            if args.selected_batch and selected_mode != f"batch{args.selected_batch}":
                raise ValueError("--selected-batch must match the measured gate selection")
            modes = tuple(args.mode or ("baseline", selected_mode))
            if any(mode not in ("baseline", selected_mode) for mode in modes):
                raise ValueError("Full modes must be baseline and the gate-selected candidate")
            report.setdefault("full_projects", [str(path.resolve()) for path in DEFAULT_PROJECTS])
            for project in projects:
                if str(project) not in report["full_projects"]:
                    report["full_projects"].append(str(project))
            for project in projects:
                inputs = project_fingerprints(project)
                previous = report.setdefault("full_inputs", {}).get(str(project))
                if previous not in (None, inputs):
                    raise ValueError("Saved full project or audio changed; use a new --output")
                report["full_inputs"][str(project)] = inputs
                for repeat in rounds:
                    for mode in alternating_modes(modes, repeat):
                        selected = packages["baseline" if mode == "baseline" else "current"]
                        target = directory / "full" / project.stem / f"round-{repeat}-{mode}"
                        spec = {"project": str(project), "source": selected["source"], "dll": selected["dll"],
                                "fingerprints": selected, "case": project.stem, "mode": mode, "repeat": repeat,
                                "inputs": inputs,
                                "ffmpeg": ffmpeg, "ffprobe": ffprobe, "video": str(target / "output.mp4"),
                                "result": str(target / "result.json")}
                        result = run_worker("full", spec)
                        result["video"] = spec["video"]
                        update_result(report, "full", result)
                        summarize(report)
                        write_report(destination, report)
                        print(json.dumps({"stage": "full", "project": project.stem, "mode": mode,
                                          "repeat": repeat, "pipeline_seconds": result["pipeline_seconds"],
                                          "total_wall_seconds": result["total_wall_seconds"]}), flush=True)
        report["status"] = "complete-" + args.stage
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_report(destination, report)
    print(json.dumps({"report": str(destination), "gate": report["gate"], "full_gate": report["full_gate"]}), flush=True)


if __name__ == "__main__":
    main()
