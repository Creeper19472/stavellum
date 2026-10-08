"""Local read-only compatibility verification; no private input path is hardcoded.

Example: python scripts/verify_flp.py INPUT.flp --compile --confirm-fixed-timing
Optionally compare pitch/start/end against a fixed-tempo reference MIDI exported
from the same Song arrangement. Reports contain counts and differences, not files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from time import perf_counter

from stavellum.domain.mapping import suggest_mappings
from stavellum.domain.models import Metadata, ProjectDocument, ProjectIR
from stavellum.importers import ImportFailure, import_project


def _musical_events(project: ProjectIR) -> Counter:
    return Counter((note.pitch, Fraction(note.start_tick, project.ppq),
                    Fraction(note.end_tick, project.ppq)) for note in project.notes)


def _reference_comparison(source: ProjectIR, reference: ProjectIR) -> dict:
    expected, actual = _musical_events(reference), _musical_events(source)
    missing, extra = expected - actual, actual - expected

    def examples(events: Counter) -> list[dict]:
        return [{"pitch": pitch, "start_beat": str(start), "end_beat": str(end),
                 "count": count} for (pitch, start, end), count in sorted(events.items())[:20]]

    return {
        "exact_pitch_and_timing_match": not missing and not extra,
        "reference_notes": len(reference.notes),
        "missing_count": sum(missing.values()), "extra_count": sum(extra.values()),
        "missing_examples": examples(missing), "extra_examples": examples(extra),
        "comparison": "音高及精确拍位起止；保留重复事件数量，不按音轨名字猜测对应关系。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="只读核对真实 FLP 导入、映射与制谱，不修改输入文件。")
    parser.add_argument("source", type=Path)
    parser.add_argument("--arrangement", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="进一步编译 MusicXML 与连续 SVG")
    parser.add_argument("--confirm-fixed-timing", action="store_true", help="明确确认工程无变速（仅影响本次验证）")
    parser.add_argument("--reference-midi", type=Path, help="同 Song 编曲的固定速度参考 MIDI")
    parser.add_argument("--output", type=Path, help="可选 JSON 报告目标；不保存工程或音频")
    args = parser.parse_args(argv)
    if args.source.suffix.casefold() != ".flp":
        parser.error("source 必须为 .flp 工程。")
    report: dict = {}
    try:
        original_digest = hashlib.sha256(args.source.read_bytes()).digest()
        begin = perf_counter()
        project = import_project(args.source, args.arrangement)
        mappings = suggest_mappings(project)
        report.update({
            "source_version": project.source_version, "source_type": project.source_type,
            "source_unchanged": True, "ppq": project.ppq, "bpm": project.bpm,
            "meter": f"{project.numerator}/{project.denominator}",
            "notes": len(project.notes), "musical_tracks": len(project.tracks),
            "duration_seconds": project.duration_seconds,
            "arrangements": project.arrangement_names, "selected_arrangement": project.arrangement_index,
            "timing_confirmed_by_importer": project.timing_confirmed,
            "import_seconds": round(perf_counter() - begin, 3),
            "mappings": [asdict(mapping) for mapping in mappings],
            "diagnostics": [asdict(diagnostic) for diagnostic in project.diagnostics],
        })
        if args.confirm_fixed_timing:
            project.timing_confirmed = True
        if args.compile:
            from stavellum.engraving.notation import build_notation

            begin = perf_counter()
            document = ProjectDocument(project, mappings, metadata=Metadata(project.name))
            notation = build_notation(document)
            report["notation"] = {
                "compile_seconds": round(perf_counter() - begin, 3),
                "logical_parts": len([mapping for mapping in mappings if mapping.enabled]),
                "staves": len(notation.staff_part_ids),
                "quantized_notes": len(notation.quantized_events),
                "timing_anchors": len(notation.anchors),
                "svg_bytes": len(notation.display_svg.encode("utf-8")),
                "musicxml_bytes": len(notation.musicxml.encode("utf-8")),
                "diagnostics": [asdict(diagnostic) for diagnostic in notation.diagnostics],
            }
        if args.reference_midi:
            report["reference"] = _reference_comparison(project, import_project(args.reference_midi))
        if hashlib.sha256(args.source.read_bytes()).digest() != original_digest:
            report["source_unchanged"] = False
            raise ValueError("验证期间源文件发生变化；请关闭外部保存操作后重试。")
        result = 0 if report.get("reference", {}).get("exact_pitch_and_timing_match", True) else 1
    except (ImportFailure, ValueError, OSError) as exc:
        message = str(exc).replace(str(args.source.resolve()), args.source.name)
        if args.reference_midi:
            message = message.replace(str(args.reference_midi.resolve()), args.reference_midi.name)
        report["error"] = message
        if isinstance(exc, ImportFailure):
            report["diagnostics"] = [asdict(diagnostic) for diagnostic in exc.diagnostics]
        result = 1
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
