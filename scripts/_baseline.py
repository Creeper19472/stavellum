"""Load an explicit Git revision for diagnostics without shipping old backends."""

from __future__ import annotations

import importlib
import importlib.util
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_package(ref, directory, *, current_native=False):
    """Keep historical imports and resources isolated from production modules."""
    commit = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
                            cwd=ROOT, check=True, capture_output=True, text=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
    archive = subprocess.run(["git", "archive", "--format=zip", commit, "src"],
                             cwd=ROOT, check=True, capture_output=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # A distinct namespace per output keeps simultaneous revisions apart.
    name = "stavellum._benchmark_baseline_" + commit[:12] + "_" + str(abs(hash(str(directory))))
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        roots = [Path(name).parent for name in source.namelist()
                 if len(Path(name).parts) == 3 and name.endswith("/__init__.py")]
        if len(roots) != 1:
            raise RuntimeError("Git baseline must contain exactly one Python package under src")
        package_root = roots[0]
        for item in source.infolist():
            item_path = Path(item.filename)
            if not item_path.is_relative_to(package_root):
                continue
            relative = item_path.relative_to(package_root)
            destination = (directory / relative).resolve()
            if not destination.is_relative_to(directory):
                raise RuntimeError("Git baseline contains an invalid resource path")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = source.read(item)
                if item_path.suffix == ".py":
                    # Isolate absolute imports and resource anchors as well as relative imports.
                    # DLL filenames and diagnostic environment variables retain their names.
                    text = data.decode("utf-8")
                    text = re.sub(r"\b(from|import) " + re.escape(package_root.name) + r"(?=[.\s])",
                                  lambda match: match[1] + " " + name, text)
                    text = re.sub(r"(['\"])" + re.escape(package_root.name) + r"\1",
                                  lambda match: match[1] + name + match[1], text)
                    data = text.replace(package_root.name.upper(), "STAVELLUM").encode("utf-8")
                destination.write_bytes(data)
    spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    if current_native:
        native_name = "rendering._rhi" if (directory / "rendering/_rhi.py").is_file() else "_rhi"
        if "." in native_name:
            importlib.import_module(f"{name}.rendering")
        sys.modules[f"{name}.{native_name}"] = importlib.import_module("stavellum.rendering._rhi")
    return package, commit


def module(package, name):
    """Resolve logical modules from an archive's own layout, without masking import errors."""
    moved = {
        "scene": "presentation.scene", "layout": "presentation.layout",
        "models": "domain.models", "render": "rendering.render",
        "raster": "rendering.raster", "rhi": "rendering.rhi",
    }
    directory = Path(package.__file__).parent
    candidate = moved.get(name, name)
    if not directory.joinpath(*candidate.split(".")).with_suffix(".py").is_file():
        candidate = "render" if name == "raster" else name
    return importlib.import_module(f"{package.__name__}.{candidate}")


def renderer(package, *, rhi=False):
    implementation = module(package, "rhi" if rhi else "render")
    return implementation.RhiFrameRenderer if rhi else implementation.FrameRenderer
