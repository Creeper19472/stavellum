# Python package architecture

Stavellum keeps its `src` layout and organizes Python modules by responsibility.
The package root contains only `__init__.py`, `__main__.py`, `cli.py`, and `demo.py`.

| Package | Responsibility | Allowed dependencies within Stavellum |
| --- | --- | --- |
| `domain` | Serializable application contracts, part mapping, progress | None |
| `graphics` | Qt lifetime, fonts, SVG, icons, branding | `domain` |
| `importers` | FLP/MIDI adapters and actionable failures | `domain` |
| `engraving` | Notation inference, engraving, printable parts | `domain`, `graphics` |
| `presentation` | Scene compilation, cache, timelines, native frame evaluation | `domain`, `graphics`, `engraving` |
| `rendering` | CPU assets, GPU composition, backend selection, bounded frame streams | `domain`, `graphics`, `presentation` |
| `exporting` | Audio inputs, encoding, offline video export | `domain`, `graphics`, `presentation`, `rendering` |
| `ui` | Desktop workflows and spawned background jobs | All the above, plus the root demo generator |

CLI and UI modules compose the workflows. Lower-level packages do not import the
CLI or UI. Within each package use relative imports; across packages use explicit
absolute imports. New subpackage initializers contain only a docstring. The
existing `importers` initializer retains `import_project` and `ImportFailure`.

## Shared contracts and dependency direction

Presentation data flows through independent modules:

```text
scene -> visibility -> layout -> types -> timeline -> curves
scene -> compilation_cache -> types / timeline / camera / axis
types -> camera -> axis
```

`types` owns scene and engraving records; `timeline` owns layout records and their
sampling methods; `curves` owns polynomial curves. The cache decodes these records
without importing the compiler. `visibility` compiles and restores visibility.
These records retain their fields and methods, including the concrete imports
needed by `get_type_hints` during cache decoding and project validation.

Rendering uses this direction:

```text
render -> rhi -> raster -> shared
render -> raster
```

`raster` owns CPU assets, tile plans and `ExportFrameStream`. `render` owns the
`FrameRenderer` that selects CPU/GPU behavior. The frame stream calls the renderer's
internal `_render_export` and `_record_export_frame` methods, so it need not import
or inspect the higher-level renderer class. CPU rendering counts its produced
frames; the backend-selecting renderer counts delivered export frames, including
GPU batches and fallback. Frame ownership, cancellation and memory bounds remain
shared by preview and export.

`font_registry` owns font registration state and bundled font families. Qt
initialization calls it after application creation; typography depends on Qt and
the registry. Font registration never creates an application recursively.
Import adapters depend on `importers.errors` rather than their dispatch initializer.

## Import migration and resources

Old flat Python import paths have been removed. For example:

```python
from stavellum.domain.models import ProjectDocument
from stavellum.presentation.scene import compile_scene
from stavellum.presentation.types import CompiledScene
from stavellum.rendering.render import FrameRenderer
from stavellum.rendering.raster import RasterFrameRenderer
from stavellum.graphics.qt import ensure_app
from stavellum.ui.gui import MainWindow
```

CLI commands and `.stproj` JSON are unchanged. Restart running applications and
workers after updating: serialized in-flight Python objects use their new module
names. Root `assets/`, `fonts/`, `icons/`, and `native/` resource directories stay
in place and are resolved through `importlib.resources.files("stavellum")`.
Native DLL overrides, Rust-before-legacy preference and development fallback stay
available. Wheels remain unpacked Windows x64 installations.

The compilation-cache fingerprint includes the relocated sources and extracted
contracts, curves and font registry. Old cache entries naturally miss because the
fingerprint changed; the record format remains version 1. Include any new
compilation input in `presentation.compilation_cache.engine_fingerprint`.

Diagnostic scripts use `scripts/_baseline.py` to resolve an archive's own flat or
subpackage layout. Absolute imports and resource anchors are isolated under its
historical namespace. Only an explicit `current_native=True` shares the current
native renderer bridge with that namespace.

## Validation

```powershell
uv run pytest -q -ra
uv run ruff check src tests scripts
uv build --wheel
```

`tests/test_package_structure.py` enforces package boundaries and an acyclic module
graph, including lazy imports and `TYPE_CHECKING` references. Existing tests cover
warm/cold caches, Windows spawn, frame ownership, batch streams, GPU fallback,
font embedding and video output. Installed-wheel checks must run outside the
repository to exercise shipped resources rather than the editable source tree.
