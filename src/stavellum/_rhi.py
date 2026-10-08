"""ctypes ABI for the Rust Vulkan compositor and historical QRhi library."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
import weakref
from collections import OrderedDict
from pathlib import Path

import PySide6
from PySide6.QtGui import QImage

from .gpu import GpuBackendError


class Quad(ctypes.Structure):
    _fields_ = [("texture_id", ctypes.c_uint64)] + [
        (name, ctypes.c_float) for name in
        ("x", "y", "w", "h", "u0", "v0", "u1", "v1", "r", "g", "b", "a")
    ]


class _OwnedFrame(ctypes.Structure):
    _fields_ = [("owner", ctypes.c_void_p), ("pixels", ctypes.c_void_p),
                ("size", ctypes.c_size_t)]


class _BatchItem(ctypes.Structure):
    _fields_ = [("quads", ctypes.POINTER(Quad)), ("count", ctypes.c_size_t)]


def library_path() -> Path:
    """Prefer shipped resources; allow an explicit DLL or development build."""
    override = os.environ.get("STAVELLUM_RHI_DLL")
    if override:
        return Path(override).resolve()
    root = Path(__file__).resolve()
    # Installed Rust builds are preferred. A development build is selected only
    # explicitly, so incomplete or stale Cargo outputs cannot change production.
    rust_library = root.parent / "native/rhi/stavellum_rust.dll"
    if rust_library.is_file():
        return rust_library
    package = Path(__file__).resolve().parent / "native/rhi/stavellum_rhi.dll"
    if package.is_file():
        return package
    return Path(__file__).resolve().parents[2] / ".cache/rhi/build/stavellum_rhi.dll"


class RhiTarget:
    """GPU textures never cross the ABI; each returned QImage owns its bytes."""

    def __init__(self, width: int, height: int, cache_megabytes: int, api: str):
        if api != "vulkan":
            raise ValueError("RHI API must be vulkan")
        if sys.platform != "win32":
            raise GpuBackendError("Vulkan 渲染需要 Windows x64。")
        # Only the legacy Qt RHI DLL shares PySide6's private ABI; the Rust
        # backend pins nothing beyond a working Vulkan loader.
        if library_path().name == "stavellum_rhi.dll" and PySide6.__version__ != "6.11.2":
            raise GpuBackendError("传统 Qt 渲染后端要求 PySide6 6.11.2。")
        self._thread = threading.get_ident()
        self.width, self.height = width, height
        self.budget = cache_megabytes * 1024 * 1024
        self._textures: OrderedDict[int, int] = OrderedDict()
        self._bytes = 0
        self.texture_cache_hits = 0
        self.texture_cache_misses = 0
        self.texture_cache_evictions = 0
        self.command_pack_seconds = 0.0
        # Retain the original copy path for measured comparisons and diagnosis.
        self._copy_readback = os.environ.get("STAVELLUM_RHI_COPY_READBACK") not in (None, "", "0")
        self._handle = None
        self._directory = os.add_dll_directory(str(Path(PySide6.__file__).parent))
        path = library_path()
        try:
            self._dll = ctypes.CDLL(str(path))
            self._dll.sprhi_abi_version.restype = ctypes.c_uint32
            self._dll.sprhi_abi_version.argtypes = ()
            if (self._dll.sprhi_abi_version() != 3 or ctypes.sizeof(Quad) != 56
                    or ctypes.sizeof(_OwnedFrame) != 24 or ctypes.sizeof(_BatchItem) != 16):
                raise GpuBackendError("Vulkan 渲染库 ABI 不匹配；请重新构建 DLL。")
            self._bind()
            self._handle = self._dll.sprhi_create(width, height, self.budget, api.encode("ascii"))
            if not self._handle:
                self._error()
        except OSError as error:
            self._directory.close()
            raise GpuBackendError("无法加载 Vulkan 渲染库；请运行 scripts/build_rust.py --install："
                                  + str(error)) from error
        except BaseException:
            self._directory.close()
            raise
        self._last_report = self.report()

    def _bind(self):
        pointer = ctypes.c_void_p
        signatures = (
            ("sprhi_abi_version", ctypes.c_uint32, ()),
            ("sprhi_create", pointer, (ctypes.c_int32, ctypes.c_int32, ctypes.c_uint64, ctypes.c_char_p)),
            ("sprhi_upload", ctypes.c_int, (pointer, ctypes.c_uint64, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, pointer)),
            ("sprhi_remove", ctypes.c_int, (pointer, ctypes.c_uint64)),
            ("sprhi_submit", ctypes.c_int, (pointer, ctypes.POINTER(Quad), ctypes.c_size_t, pointer, ctypes.c_size_t)),
            ("sprhi_submit_owned", ctypes.c_int, (pointer, ctypes.POINTER(Quad), ctypes.c_size_t,
                                                 ctypes.POINTER(_OwnedFrame))),
            ("sprhi_submit_batch_owned", ctypes.c_int,
             (pointer, ctypes.POINTER(_BatchItem), ctypes.c_size_t, ctypes.POINTER(_OwnedFrame))),
            ("sprhi_release_frame", None, (pointer,)),
            ("sprhi_report", ctypes.c_char_p, (pointer,)),
            ("sprhi_last_error", ctypes.c_char_p, (pointer,)),
            ("sprhi_close", ctypes.c_int, (pointer,)),
        )
        for name, result, arguments in signatures:
            function = getattr(self._dll, name)
            function.restype = result
            function.argtypes = arguments

    def _check(self):
        if threading.get_ident() != self._thread:
            raise GpuBackendError("RHI renderer must be used on its creating thread")
        if not self._handle:
            raise RuntimeError("RHI renderer is closed")

    def _error(self):
        message = self._dll.sprhi_last_error(self._handle)
        raise GpuBackendError(message.decode("utf-8", errors="replace") if message else "Native RHI operation failed")

    def texture(self, image: QImage) -> int:
        self._check()
        key = image.cacheKey()
        if key in self._textures:
            self.texture_cache_hits += 1
            self._textures.move_to_end(key)
            return key
        pixels = image.convertToFormat(QImage.Format.Format_RGBA8888_Premultiplied)
        view = pixels.bits()
        buffer = (ctypes.c_uint8 * pixels.sizeInBytes()).from_buffer(view)
        if self._dll.sprhi_upload(self._handle, key, pixels.width(), pixels.height(), pixels.bytesPerLine(), buffer):
            self._error()
        size = pixels.width() * pixels.height() * 4
        self._textures[key] = size
        self._bytes += size
        self.texture_cache_misses += 1
        return key

    def render(self, commands: list[Quad], *, evict: bool = True) -> QImage:
        self._check()
        started = time.perf_counter()
        quads = (Quad * len(commands))(*commands)
        self.command_pack_seconds += time.perf_counter() - started
        if self._copy_readback:
            image = QImage(self.width, self.height, QImage.Format.Format_ARGB32)
            if image.isNull():
                raise GpuBackendError("Cannot allocate RHI output image")
            view = image.bits()
            buffer = (ctypes.c_uint8 * image.sizeInBytes()).from_buffer(view)
            if self._dll.sprhi_submit(self._handle, quads, len(commands), buffer, image.sizeInBytes()):
                self._error()
        else:
            frame = _OwnedFrame()
            if self._dll.sprhi_submit_owned(self._handle, quads, len(commands), ctypes.byref(frame)):
                self._error()
            image = self._owned_image(frame)
        if evict:
            self._evict()
        return image

    def render_batch(self, commands_by_frame: list[list[Quad]]) -> list[QImage]:
        """Complete one bounded submission before publishing any owned images."""
        self._check()
        count = len(commands_by_frame)
        if not 1 <= count <= 8:
            raise ValueError("RHI batches must contain between one and eight frames")
        if self._copy_readback:
            # The historical copy diagnostic still exercises its original ABI.
            images = [self.render(commands, evict=False) for commands in commands_by_frame]
            self._evict()
            return images
        started = time.perf_counter()
        arrays = [(Quad * len(commands))(*commands) for commands in commands_by_frame]
        items = (_BatchItem * count)(*(_BatchItem(quads, len(quads)) for quads in arrays))
        self.command_pack_seconds += time.perf_counter() - started
        frames = (_OwnedFrame * count)()
        images = []
        try:
            if self._dll.sprhi_submit_batch_owned(self._handle, items, count, frames):
                self._error()
            for frame in frames:
                # _owned_image consumes this owner even if wrapping fails. Clear
                # the output slot first so the cleanup below cannot free it twice.
                owned = _OwnedFrame(frame.owner, frame.pixels, frame.size)
                frame.owner = frame.pixels = None
                frame.size = 0
                images.append(self._owned_image(owned))
            self._evict()
            return images
        except BaseException:
            images.clear()
            raise
        finally:
            for frame in frames:
                if frame.owner:
                    self._dll.sprhi_release_frame(frame.owner)
                    frame.owner = None

    def _evict(self):
        # endOffscreenFrame has completed the entire batch before eviction.
        while self._textures and self._bytes > self.budget:
            key, size = next(iter(self._textures.items()))
            if self._dll.sprhi_remove(self._handle, key):
                self._error()
            self._textures.pop(key)
            self._bytes -= size
            self.texture_cache_evictions += 1

    def _owned_image(self, frame: _OwnedFrame) -> QImage:
        release = None
        try:
            if not frame.owner or not frame.pixels or frame.size != self.width * self.height * 4:
                raise GpuBackendError("Invalid owned RHI frame")
            buffer = (ctypes.c_uint8 * frame.size).from_address(frame.pixels)
            release = weakref.finalize(buffer, self._dll.sprhi_release_frame, frame.owner)
            # At interpreter shutdown live images need no explicit free; the OS
            # releases them. Runtime collection follows the buffer's lifetime.
            release.atexit = False
            view = memoryview(buffer).cast("B")
            # PySide retains this Python buffer in QImage's shared data, also
            # across shallow QImage copies. Release frees only a QByteArray;
            # it needs neither the renderer nor its thread/graphics resources.
            image = QImage(view, self.width, self.height, self.width * 4,
                           QImage.Format.Format_ARGB32)
            if image.isNull():
                raise GpuBackendError("Cannot wrap owned RHI pixels")
            return image
        except BaseException:
            if release is not None:
                release()
            elif frame.owner:
                self._dll.sprhi_release_frame(frame.owner)
            raise

    def report(self) -> dict:
        self._check()
        value = self._dll.sprhi_report(self._handle)
        if not value:
            self._error()
        self._last_report = json.loads(value)
        self._last_report.update(
            gpu_texture_cache_hits=self.texture_cache_hits,
            gpu_texture_cache_misses=self.texture_cache_misses,
            gpu_texture_cache_evictions=self.texture_cache_evictions,
            command_pack_seconds=self.command_pack_seconds,
        )
        return self._last_report.copy()

    def close(self):
        if self._handle is None:
            return
        self._check()
        self._last_report = self.report()
        if self._dll.sprhi_close(self._handle):
            self._error()
        self._handle = None
        self._textures.clear()
        self._directory.close()
