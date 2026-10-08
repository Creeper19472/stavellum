# Rust Vulkan renderer

The native compositor lives in `native/rust-renderer`. It is written directly
against Vulkan through `ash` — no Qt RHI, no wgpu, no C++ — and keeps the exact
`sprhi_*` C ABI (version 3) that `src/stavellum/_rhi.py` loads. The
application still uses Python/PySide6 for imports, notation, layout and UI;
this crate is the first completed piece of the full Rust migration.

## Architecture

- **Instanced quads**: one 48-byte instance per quad. The six corner vertices
  are expanded in the vertex shader from `gl_VertexIndex`, replacing the old
  6 × 32 CPU-expanded vertices per quad (75% less upload, no CPU transform).
- **Run merging**: adjacent quads sharing a texture collapse into one
  instanced draw call, preserving painter order (no global sorting).
- **Persistent host rings**: instance payloads, texture uploads and readbacks
  run through three persistently mapped host buffers that only ever grow.
  Uploads and instances are written in place; no per-frame staging copies.
- **HOST_CACHED readback**: the readback buffer deliberately uses a
  `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` memory type. Plain
  host-visible memory on discrete GPUs is mapped write-combined, and reading
  pixels back through it is an order of magnitude slower.
- **Tight rows**: `bufferRowLength = width` copies need no CPU depitching.
- **One submission per batch**: up to eight frames share one command buffer,
  one queue submission and one fence wait; pending texture uploads ride the
  same submission through explicit barriers.
- **Recycled CPU frames**: returned frames are pooled and reused, avoiding
  multi-megabyte heap commit churn per frame under the GNU allocator.
- **Build-time shaders**: `src/shaders/quad.wgsl` is compiled to SPIR-V by
  `build.rs` using naga and embedded into the DLL; no runtime translation,
  no Qt SDK, no QSB files.
- 4×MSAA with resolve, premultiplied source-over blending, linear sampling
  with clamp-to-edge, opaque black clear — bit-identical output to the
  historical Qt backend for every pixel-level test in the suite.
- SPIR-V pushes pixels with Vulkan's y-down NDC; the WGSL shader compensates
  once because naga preserves WGSL's y-up convention.

## Build and select

```powershell
uv run python scripts/build_rust.py            # build target/release/stavellum_rust.dll
uv run python scripts/build_rust.py --install  # copy into src/stavellum/native/rhi
$env:STAVELLUM_RHI_DLL = "$PWD/target/release/stavellum_rust.dll"   # explicit override
```

The script uses MSVC when Visual Studio C++ tools are installed and otherwise
falls back to a preinstalled `x86_64-pc-windows-gnu` Rust toolchain. The
MinGW gcc is discovered per invocation (`STAVELLUM_MINGW`, `gcc.exe` on PATH,
or a gitignored `.cargo/config.local.toml`) and exported through
`CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER`; its directory also provides
`dlltool` for crate build scripts. `libgcc` is linked statically (pinned in
`.cargo/config.toml`), so the DLL depends only on system libraries.

An installed `stavellum_rust.dll` is preferred automatically by
`_rhi.library_path()`; the legacy Qt DLL remains selectable through
`STAVELLUM_RHI_DLL`.

## Measured performance (RTX 5070 Laptop GPU, synchronous ABI, readback included)

Synthetic (`scripts/benchmark_rust.py`, 1920×1080, 1000 quads/frame):

| batch | fps | median frame |
| --- | --- | --- |
| 1 | 532 | 1.72 ms |
| 4 | 600 | 1.63 ms |
| 8 | 666 | 1.50 ms |

Real demo scene (`artifacts/bench/bench_scene.py`, 300 frames, batch 4):
CPU raster 234 fps → Vulkan stream **538 fps (2.3×)**; at 3840×2160 the CPU
backend manages 78 fps → **149 fps (1.9×)**. The remaining per-frame budget is
roughly one fence wait (~0.7 ms) plus one full-frame copy into the caller's
owned buffer (~1.0 ms at 1080p) — both inherent to the synchronous owned-frame
ABI. Further gains need an async ABI (version 4) so the next batch renders
while FFmpeg consumes the previous one.

## Validation

```powershell
cargo test -p stavellum-renderer          # ABI layout, swizzle, null handling
uv run pytest -q tests/test_rust_renderer.py tests/test_rhi_owned.py \
    tests/test_rhi_batch_native.py tests/test_gpu_readback.py tests/test_rhi.py -m integration
uv run python scripts/benchmark_rust.py --width 1920 --height 1080 --frames 240
uv run pytest -q --basetemp=artifacts/pytest-tmp
```

The integration suite covers pixel-exact texture sampling and blending,
RGBA/BGRA readback parity, empty frames, batch identity across sizes 1–8,
pre-submission error rejection with exact messages, cross-thread release,
random-access reproducibility and CPU-reference comparison on production
scenes.

See `RUST_GLM_COLLABORATION.md` for the migration log and ownership record.
