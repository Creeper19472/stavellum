"""Build the Rust native libraries (Vulkan renderer and scene core)."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mingw_linker() -> Path | None:
    """A usable MinGW gcc for the GNU target, wherever it can be found."""
    override = os.environ.get("STAVELLUM_MINGW")
    if override:
        gcc = Path(override) / "gcc.exe"
        return gcc if gcc.is_file() else None
    on_path = shutil.which("gcc.exe")
    if on_path:
        return Path(on_path)
    for config in (ROOT / ".cargo" / "config.local.toml", ROOT / ".cargo" / "config.toml"):
        if not config.is_file():
            continue
        match = re.search(r'linker\s*=\s*"([^"]+)"', config.read_text(encoding="utf-8"))
        if match and Path(match.group(1)).is_file():
            return Path(match.group(1))
    return None


def build_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["CARGO_TARGET_DIR"] = str(ROOT / "target")
    environment.setdefault("CARGO_BUILD_JOBS", "2")
    mingw = os.environ.get("STAVELLUM_MINGW")
    if mingw:
        linker = Path(mingw) / "gcc.exe"
        if not linker.is_file():
            raise RuntimeError("STAVELLUM_MINGW must point to a MinGW bin directory")
        environment["PATH"] = str(linker.parent) + os.pathsep + environment["PATH"]
        environment["CARGO_BUILD_TARGET"] = "x86_64-pc-windows-gnu"
        environment["CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER"] = str(linker)
        return environment
    if sys.platform != "win32":
        return environment
    locator = (Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
               / "Microsoft Visual Studio/Installer/vswhere.exe")
    if locator.is_file():
        discovery = subprocess.run(
            [str(locator), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            check=True, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout.strip()
        if discovery:
            # A real Visual Studio toolchain: run inside its environment.
            vcvars = Path(discovery) / "VC/Auxiliary/Build/vcvars64.bat"
            query = subprocess.run(
                f'cmd.exe /d /s /c ""{vcvars}" >nul && set"',
                check=True, capture_output=True, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for line in query.stdout.splitlines():
                if "=" in line and not line.startswith("="):
                    key, value = line.split("=", 1)
                    environment[key] = value
            return environment
    # No MSVC link.exe: fall back to a preinstalled windows-gnu Rust toolchain.
    # Its linker is already pinned by .cargo/config.toml; the same directory
    # must be on PATH because crate build scripts invoke dlltool.
    try:
        toolchains = subprocess.run(["rustup", "toolchain", "list"], check=True,
                                    capture_output=True, text=True,
                                    creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("No MSVC link.exe and rustup unavailable: "
                           f"{error}") from error
    gnu = next((line.split()[0] for line in toolchains.splitlines()
                if line.split()[0].endswith("x86_64-pc-windows-gnu")), None)
    if not gnu:
        raise RuntimeError("Install Visual Studio C++ Build Tools or a "
                           "x86_64-pc-windows-gnu Rust toolchain")
    environment["RUSTUP_TOOLCHAIN"] = gnu
    linker = mingw_linker()
    if linker:
        environment["CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER"] = str(linker)
        environment["PATH"] = str(linker.parent) + os.pathsep + environment["PATH"]
    if not shutil.which("dlltool.exe", path=environment["PATH"]):
        raise RuntimeError("MinGW bin directory (dlltool.exe) is required on PATH "
                           "for the GNU target; set STAVELLUM_MINGW or install gcc")
    return environment


def build(*, install: bool = False, destination: Path | None = None) -> Path:
    if sys.platform != "win32":
        raise RuntimeError("Application packaging currently requires Windows x64")
    if shutil.which("cargo") is None:
        raise RuntimeError("Install the stable Rust MSVC toolchain")
    environment = build_environment()
    subprocess.run(["cargo", "build", "--release", "--locked", "--workspace"],
                   cwd=ROOT, env=environment, check=True)
    target_root = ROOT / "target"
    if environment.get("CARGO_BUILD_TARGET"):
        target_root /= environment["CARGO_BUILD_TARGET"]
    release = target_root / "release"
    libraries = [release / "stavellum_rust.dll", release / "stavellum_core.dll"]
    if install:
        destination = destination or ROOT / "src/stavellum/native/rhi"
        destination.mkdir(parents=True, exist_ok=True)
        core_destination = destination.parent / "core"
        core_destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(libraries[0], destination / libraries[0].name)
        shutil.copy2(libraries[1], core_destination / libraries[1].name)
    return libraries[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--install-dir", type=Path)
    args = parser.parse_args()
    print(build(install=args.install, destination=args.install_dir))


if __name__ == "__main__":
    main()
