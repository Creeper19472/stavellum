# Rust Vulkan renderer

The native compositor lives in `native/rust-renderer`. It is written directly
against Vulkan through `ash` — no Qt RHI, no wgpu, no C++ — and keeps the exact
`sprhi_*` C ABI (version 3) that `src/stavellum/_rhi.py` loads. The
application still uses Python/PySide6 for imports, notation, layout and UI.

The required `native/rust-core` library exposes `spcore_*` ABI version 1
through `src/stavellum/_core.py`. A renderer-owned `FrameEvaluator` serializes
compiled curves and notes once, then evaluates camera position, layout rows,
activity and tile-plan geometry with one native call per uncached time point.
CPU and Vulkan rendering share the same evaluator, including GPU-to-CPU recovery.
Its five-state cache reuses batch lookahead and drawing results without retaining
images. Python still compiles engraving, camera parameters and layout curves;
its tile residency selection continues to enforce the actual cache budget.
`FrameState` owns immutable Python rows and activity values, independently of
native buffers. Native handles belong to renderers, never `CompiledScene` or
the disk compilation cache. Rebuilding compiled settings creates a new evaluator;
temporary metadata display overrides do not rebuild it.

Renderer closure releases the shared evaluator even if graphics cleanup fails.
The Vulkan device and instance have explicit native owners, including partial
initialization failures; ash handles alone do not release those Vulkan objects.

Scene compute errors propagate independently of graphics errors. Missing or
incompatible core DLLs fail rendering even in CPU mode; there is no Python scene
compute fallback. Ordinary imports and project inspection need no native core.
Rebuild and install both libraries before running rendering tests or packaging.
Wheels require and include the core DLL; editable installation can precede building.
Only the explicit `STAVELLUM_CORE_DLL` override or the installed package resource
is selected; development Cargo outputs are never selected implicitly.

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
uv run python scripts/build_rust.py --install  # install renderer and scene core DLLs
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

To build and select the legacy Qt backend explicitly:

```powershell
uv run --with py7zr python scripts/build_rhi.py --prepare --install
$env:STAVELLUM_RHI_DLL = "$PWD/src/stavellum/native/rhi/stavellum_rhi.dll"
```

Remove `STAVELLUM_RHI_DLL` to restore automatic backend selection.
`STAVELLUM_CORE_DLL` overrides the required scene core for all rendering paths.
An invalid path or ABI is an error, rather than a request to use Python.

## Historical performance (RTX 5070 Laptop GPU, synchronous ABI, readback included)

These figures were reported during the original PR #1 migration. They have
not been reproduced as part of this extraction and are not current validation
or a comparison with the legacy Qt Vulkan backend. The demo harness referenced
below is a local artifact, rather than a shipped benchmark script.

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
cargo test --workspace --locked          # renderer ABI and scene core unit tests
uv run pytest -q tests/test_core_native.py tests/test_rust_renderer.py tests/test_rhi_owned.py tests/test_rhi_batch_native.py tests/test_gpu_readback.py tests/test_rhi.py -m integration
uv run python scripts/benchmark_rust.py --width 1920 --height 1080 --frames 240
uv run pytest -q --basetemp=artifacts/pytest-tmp
```

The integration suite covers pixel-exact texture sampling and blending,
RGBA/BGRA readback parity, empty frames, batch identity across sizes 1–8,
pre-submission error rejection with exact messages, cross-thread release,
random-access reproducibility and CPU-reference comparison on production
scenes.

Vulkan tests may skip when no usable graphics device is available. Required core
tests fail when the core DLL is missing or incompatible. Report those skips separately from
passing checks.

## Scene evaluation verification and performance

The completed migration measurements and their limits are recorded in
[the core evaluation report](core-evaluation-performance.md).

The independent pre-migration Python arithmetic lives only in test references.
Numerical parity, pixel parity against an explicit Git baseline, random seeking,
cache residency and lifecycle behavior are covered separately. Runtime reports
include `scene_compute_backend`, the selected library and ABI, initialization
cost, `scene_evaluation_count`, `scene_evaluation_seconds` and cache hits.
Export stream counters are frozen deltas for that stream.
Camera integration retains low-order floating terms when summing segments,
matching Python `math.fsum`; even a one-ulp coordinate difference can change
antialiased pixels. Float checks use both relative and absolute tolerances of
`1e-9`; pixel bytes, tile indices, budgets and residency sets remain exact.

```powershell
uv run python scripts/benchmark_core.py --baseline-ref <revision> --repeats 5
```

This diagnostic runs materialized frame evaluation plus CPU/Vulkan frame
production at 1080p and 4K, with cold and warm tile caches and actual FFmpeg
video encoding. Compilation is excluded from throughput. Pixel comparisons
and report serialization run outside timing. The baseline reference is explicit;
no Python reference evaluator is shipped as a selectable production backend.
The report records library fingerprints, initialization cost and native versus
materialized evaluation timings. Quality checks include random and repeated
times, plus export batch sizes 1/2/4 (subject to the existing output memory cap).
Use `--frames 600 --phases warm --cases dense --sizes 1920x1080 --backends cpu`
to investigate a noisy warm-cache measurement with longer alternating samples.
An evaluation speedup below 3× in dense cases or any median regression above
5% prevents the report's `acceptance_met` flag from passing.
