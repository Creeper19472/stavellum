"""Desktop launcher and reproducible headless entrypoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stavellum.domain.models import ProjectDocument, load_document, save_document


def _add_inference_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--auto-staccato", action=argparse.BooleanOptionalAction, default=None,
                        help="保守识别跳音；未指定时沿用项目设置")
    parser.add_argument("--auto-grace", action=argparse.BooleanOptionalAction, default=None,
                        help="保守识别倚音；未指定时沿用项目设置")
    parser.add_argument("--auto-dynamics", action=argparse.BooleanOptionalAction, default=None,
                        help="识别 FL 音量自动化的渐强／渐弱；未指定时沿用项目设置")
    parser.add_argument("--auto-simplify-accidentals", action=argparse.BooleanOptionalAction, default=None,
                        help="保留调号并用等音拼写简化音符记法；未指定时沿用项目设置")
    parser.add_argument("--auto-ottava", action=argparse.BooleanOptionalAction, default=None,
                        help="用八度或双八度记号简化连续高低音区；未指定时沿用项目设置")


def _apply_inference_options(document: ProjectDocument, args: argparse.Namespace) -> None:
    for name in ("auto_staccato", "auto_grace", "auto_dynamics", "auto_simplify_accidentals",
                 "auto_ottava"):
        value = getattr(args, name)
        if value is not None:
            for mapping in document.mappings:
                setattr(mapping, name, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stavellum", description="FL Studio 工程转滚动五线谱；所有视频均逐帧离线渲染。")
    sub = parser.add_subparsers(dest="command")
    gui = sub.add_parser("gui", help="启动桌面界面（默认）")
    gui.add_argument("project", nargs="?")
    demo = sub.add_parser("demo", help="创建原创演示项目、MIDI 和 WAV")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--bars", type=int, default=16)
    demo.add_argument("--tail", type=float, default=1.25)
    demo.add_argument("--open", action="store_true")
    inspect = sub.add_parser("inspect", help="检查 FLP / MIDI 导入结果")
    inspect.add_argument("source")
    inspect.add_argument("--arrangement", type=int, default=0)
    convert = sub.add_parser("import", help="创建可校正的项目文件")
    convert.add_argument("source")
    convert.add_argument("--audio", default="")
    convert.add_argument("--output", required=True)
    convert.add_argument("--arrangement", type=int, default=0)
    convert.add_argument("--confirm-fixed-timing", action="store_true")
    convert.add_argument("--bpm", type=float)
    frame = sub.add_parser("frame", help="渲染一个确定时间的 PNG 帧")
    frame.add_argument("project")
    frame.add_argument("--time", type=float, default=0, help="从视频开头计时的展示秒数，包含开场停顿")
    frame.add_argument("--output", required=True)
    video = sub.add_parser("render", help="逐帧导出 MP4")
    video.add_argument("project")
    video.add_argument("--output", required=True)
    video.add_argument("--width", type=int)
    video.add_argument("--height", type=int)
    video.add_argument("--fps", type=int)
    video.add_argument("--preset", choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"))
    parts = sub.add_parser("parts", help="导出每种乐器的 MusicXML 和 PDF")
    parts.add_argument("project")
    parts.add_argument("--output", required=True)
    for command in (convert, frame, video, parts):
        _add_inference_options(command)
    for command in (frame, video):
        command.add_argument("--render-backend", choices=("auto", "cpu", "gpu"),
                             help="帧渲染后端；gpu 使用 RHI Vulkan，auto 优先 RHI Vulkan 并允许回退 CPU")
    video.add_argument("--video-encoder", choices=("auto", "libx264", "h264_nvenc"),
                       help="视频编码器；auto 优先 NVIDIA NVENC 并允许回退")
    video.add_argument("--nvenc-cq", type=int, help="NVENC CQ 质量（1–51，较小更清晰）")
    video.add_argument("--nvenc-preset", choices=tuple(f"p{i}" for i in range(1, 8)),
                       help="独立的 NVENC 编码预设（p1 最快，p7 最慢）")
    args = parser.parse_args(argv)
    try:
        if args.command in (None, "gui"):
            from stavellum.ui.gui import run_gui
            return run_gui(getattr(args, "project", None))
        if args.command == "demo":
            from .demo import create_demo_document
            document = create_demo_document(args.output, args.bars, args.tail)
            path = str(Path(args.output).resolve() / "demo.stproj")
            print(path)
            if args.open:
                from stavellum.ui.gui import run_gui
                return run_gui(path)
            return 0
        if args.command in ("inspect", "import"):
            from .importers import import_project
            project = import_project(args.source, args.arrangement)
            if args.command == "inspect":
                from dataclasses import asdict
                summary = {"name": project.name, "source_version": project.source_version, "notes": len(project.notes), "tracks": [asdict(t) for t in project.tracks], "bpm": project.bpm, "meter": f"{project.numerator}/{project.denominator}", "arrangements": project.arrangement_names, "timing_confirmed": project.timing_confirmed, "diagnostics": [asdict(d) for d in project.diagnostics]}
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return 0
            from stavellum.domain.mapping import suggest_mappings
            from stavellum.domain.models import Metadata
            if args.bpm is not None:
                project.bpm = args.bpm
            if args.confirm_fixed_timing:
                project.timing_confirmed = True
            document = ProjectDocument(project, suggest_mappings(project), str(Path(args.audio).resolve()) if args.audio else "", metadata=Metadata(project.name))
            _apply_inference_options(document, args)
            save_document(document, args.output)
            print(str(Path(args.output).resolve()))
            return 0
        document = load_document(args.project)
        _apply_inference_options(document, args)
        if args.command in ("frame", "render") and args.render_backend is not None:
            document.settings.render_backend = args.render_backend
        if args.command == "frame":
            from stavellum.graphics.qt import prepare_render_app
            from stavellum.presentation.scene import compile_scene
            from stavellum.rendering.render import FrameRenderer
            document.validate()
            prepare_render_app(document.settings)
            path = Path(args.output).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            renderer = FrameRenderer(compile_scene(document))
            try:
                if not renderer.render_frame(args.time).save(str(path)):
                    raise RuntimeError("无法保存 PNG。")
            finally:
                renderer.close()
            print(path)
        elif args.command == "render":
            from stavellum.exporting.export import export_video
            from stavellum.graphics.qt import prepare_render_app
            for attr in ("width", "height", "fps", "preset", "video_encoder", "nvenc_cq", "nvenc_preset"):
                if getattr(args, attr) is not None:
                    setattr(document.settings, attr, getattr(args, attr))
            document.validate()
            prepare_render_app(document.settings)
            print(export_video(document, args.output, _progress))
        elif args.command == "parts":
            from stavellum.engraving.notation import export_parts
            for path in export_parts(document, args.output, _progress):
                print(path)
        return 0
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


def _progress(fraction: float, message: str) -> None:
    print(f"{fraction:6.1%} {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
