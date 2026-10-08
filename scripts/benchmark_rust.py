"""Measure real Vulkan ABI frame production, keeping readback in the timing."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Quad(ctypes.Structure):
    _fields_ = [("texture_id", ctypes.c_uint64), ("values", ctypes.c_float * 12)]


class Frame(ctypes.Structure):
    _fields_ = [("owner", ctypes.c_void_p), ("pixels", ctypes.c_void_p),
                ("size", ctypes.c_size_t)]


class Batch(ctypes.Structure):
    _fields_ = [("quads", ctypes.POINTER(Quad)), ("count", ctypes.c_size_t)]


def bind(path):
    dll = ctypes.CDLL(str(path))
    pointer = ctypes.c_void_p
    for name, result, arguments in (
        ("sprhi_create", pointer, (ctypes.c_int32, ctypes.c_int32, ctypes.c_uint64, ctypes.c_char_p)),
        ("sprhi_abi_version", ctypes.c_uint32, ()),
        ("sprhi_submit_batch_owned", ctypes.c_int,
         (pointer, ctypes.POINTER(Batch), ctypes.c_size_t, ctypes.POINTER(Frame))),
        ("sprhi_release_frame", None, (pointer,)),
        ("sprhi_report", ctypes.c_char_p, (pointer,)),
        ("sprhi_last_error", ctypes.c_char_p, (pointer,)),
        ("sprhi_close", ctypes.c_int, (pointer,)),
    ):
        function = getattr(dll, name)
        function.restype, function.argtypes = result, arguments
    return dll


def measure(dll, width, height, count, quads, batch_size):
    handle = dll.sprhi_create(width, height, 64 * 1024 * 1024, b"vulkan")
    if not handle:
        raise RuntimeError(dll.sprhi_last_error(None).decode())
    commands = (Quad * quads)(*(
        Quad(0, (ctypes.c_float * 12)(
            (i * 19) % width, (i * 31) % height, 20, 30,
            0, 0, 1, 1, .2, .1, .05, .25)) for i in range(quads)))
    samples = []
    try:
        for offset in range(0, count + 16, batch_size):
            size = min(batch_size, count + 16 - offset)
            frames = (Frame * size)()
            items = (Batch * size)(*(Batch(commands, quads) for _ in range(size)))
            started = time.perf_counter()
            status = dll.sprhi_submit_batch_owned(handle, items, size, frames)
            elapsed = time.perf_counter() - started
            if status:
                raise RuntimeError(dll.sprhi_last_error(handle).decode())
            try:
                assert all(f.size == width * height * 4 and f.owner and f.pixels for f in frames)
            finally:
                for frame in frames:
                    dll.sprhi_release_frame(frame.owner)
            if offset >= 16:
                samples.extend([elapsed / size] * size)
        report = json.loads(dll.sprhi_report(handle))
        return {"batch_size": batch_size, "frames": len(samples),
                "fps": len(samples) / sum(samples),
                "median_frame_ms": statistics.median(samples) * 1000,
                "p95_frame_ms": sorted(samples)[max(0, int(len(samples) * .95) - 1)] * 1000,
                "backend_report": report}
    finally:
        if dll.sprhi_close(handle):
            raise RuntimeError(dll.sprhi_last_error(handle).decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--quads", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/rust-benchmark.json")
    args = parser.parse_args()
    if min(args.width, args.height, args.frames, args.quads) <= 0:
        parser.error("dimensions, frames and quads must be positive")
    path = args.dll or Path(os.environ.get("STAVELLUM_RHI_DLL", str(
        ROOT / "src/stavellum/native/rhi/stavellum_rust.dll")))
    dll = bind(path)
    assert dll.sprhi_abi_version() == 3
    result = {"dll": str(path.resolve()), "width": args.width, "height": args.height,
              "quads": args.quads, "limitations": "synthetic solid quads; excludes notation, assets and encoder; batch latency amortized per frame",
              "results": [measure(dll, args.width, args.height, args.frames, args.quads, size)
                          for size in (1, 2, 4, 8)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
