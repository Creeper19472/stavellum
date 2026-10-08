"""Build the Windows QRhi Vulkan backend against the pinned PySide6 runtime.

Preparation: uv run --with py7zr python scripts/build_rhi.py --prepare
Build again: uv run python scripts/build_rhi.py
Install wheel resources: uv run python scripts/build_rhi.py --install
Only SDK files under .cache/rhi are downloaded; Qt runtime DLLs are not installed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import PySide6

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "rhi"
VERSION = "6.11.2"
REPOSITORY = (
    "https://download.qt.io/online/qtsdkrepository/windows_x86/desktop/"
    "qt6_6112/qt6_6112_msvc2022_64/"
)


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print("Downloading", url, flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)


def prepare() -> Path:
    import py7zr

    metadata = CACHE / "Updates.xml"
    if not metadata.exists():
        fetch(REPOSITORY + "Updates.xml", metadata)
    package = next(
        item for item in ET.parse(metadata).getroot().findall("PackageUpdate")
        if item.findtext("Name") == "qt.qt6.6112.win64_msvc2022_64"
    )
    archives = package.findtext("DownloadableArchives", "").split(",")
    archive = next(item.strip() for item in archives if item.strip().startswith("qtbase-"))
    filename = package.findtext("Version", "") + archive
    url = REPOSITORY + package.findtext("Name", "") + "/" + filename
    downloaded = CACHE / "downloads" / filename
    if not downloaded.exists():
        fetch(url, downloaded)
    expected = urllib.request.urlopen(url + ".sha1", timeout=30).read().decode().split()[0]
    with downloaded.open("rb") as archive_file:
        actual = hashlib.file_digest(archive_file, "sha1").hexdigest()
    if actual != expected:
        raise RuntimeError("Qt SDK archive checksum mismatch")
    sdk = CACHE / "sdk"
    if not (sdk / "lib" / "cmake" / "Qt6" / "Qt6Config.cmake").exists():
        print("Extracting Qt development SDK", flush=True)
        with py7zr.SevenZipFile(downloaded) as source:
            sdk_root = (CACHE / "sdk").resolve()
            for name in source.getnames():
                if not (sdk_root / name).resolve().is_relative_to(sdk_root):
                    raise RuntimeError("Invalid path in Qt SDK archive")
            source.extractall(CACHE / "sdk")
    headers = CACHE / "Vulkan-Headers-1.4.341" / "include"
    if not (headers / "vulkan" / "vulkan.h").exists():
        archive_zip = CACHE / "downloads" / "Vulkan-Headers-1.4.341.zip"
        if not archive_zip.exists():
            fetch("https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v1.4.341.zip", archive_zip)
        with zipfile.ZipFile(archive_zip) as source:
            cache_root = CACHE.resolve()
            for member in source.namelist():
                if not (cache_root / member).resolve().is_relative_to(cache_root):
                    raise RuntimeError("Invalid path in Vulkan-Headers archive")
            source.extractall(CACHE)
    return sdk


def build(sdk: Path) -> Path:
    if sys.platform != "win32" or PySide6.__version__ != VERSION:
        raise RuntimeError("RHI Vulkan requires Windows x64 and PySide6 " + VERSION)
    configured_vs = os.environ.get("STAVELLUM_VS")
    if configured_vs:
        vs = Path(configured_vs)
    else:
        locator = (Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
                   / "Microsoft Visual Studio/Installer/vswhere.exe")
        if not locator.is_file():
            raise RuntimeError("Install Visual Studio C++ build tools or set STAVELLUM_VS")
        discovery = subprocess.run(
            [str(locator), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            check=True, capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout.strip()
        if not discovery:
            raise RuntimeError("No Visual Studio C++ toolchain found; set STAVELLUM_VS")
        vs = Path(discovery)
    cmake = vs / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
    ninja = vs / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
    vcvars = vs / "VC/Auxiliary/Build/vcvars64.bat"
    query = subprocess.run(
        f'cmd.exe /d /s /c ""{vcvars}" >nul && set"',
        check=True, capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    environment = dict(os.environ)
    for line in query.stdout.splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            environment[key] = value
    output = CACHE / "build"
    qsb = Path(PySide6.__file__).parent / "qsb.exe"
    commands = [
        [str(cmake), "--fresh", "-S", str(ROOT / "native/rhi"), "-B", str(output), "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_MAKE_PROGRAM=" + str(ninja),
         "-DCMAKE_PREFIX_PATH=" + str(sdk), "-DQSB_EXECUTABLE=" + str(qsb),
         "-DVULKAN_INCLUDE_DIR=" + str(CACHE / "Vulkan-Headers-1.4.341/include")],
        [str(cmake), "--build", str(output)],
    ]
    for command in commands:
        result = subprocess.run(command, env=environment, capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        result.check_returncode()
    result = output / "stavellum_rhi.dll"
    print(result, flush=True)
    return result


def install_resources(library: Path, destination: Path) -> None:
    """Bundle only our DLL and shaders; Qt runtime remains supplied by PySide6."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in (library.name, "quad.vert.qsb", "quad.frag.qsb"):
        shutil.copy2(library.parent / name, destination / name)
    print("Installed RHI resources:", destination.resolve(), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rust", "qt"), default="qt",
                        help="Select rust for the new compositor; qt retains the verified backend")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--sdk", type=Path, default=CACHE / "sdk")
    parser.add_argument("--install", action="store_true",
                        help="Copy DLL/shaders into the Python package for wheel builds")
    parser.add_argument("--install-dir", type=Path, default=ROOT / "src/stavellum/native/rhi",
                        help="Package resource destination used with --install")
    args = parser.parse_args()
    if args.backend == "rust":
        from build_rust import build as build_rust
        print(build_rust(install=args.install, destination=args.install_dir))
        return
    sdk = prepare() if args.prepare else args.sdk
    library = build(sdk)
    if args.install:
        install_resources(library, args.install_dir)


if __name__ == "__main__":
    main()
