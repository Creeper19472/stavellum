"""Choose shared CPU/GPU rendering for preview and offline frame output."""

from __future__ import annotations

import time as clock

from PySide6.QtGui import QImage

from stavellum.presentation.types import CompiledScene

from .gpu import GpuBackendError
from .raster import RasterFrameRenderer


class FrameRenderer(RasterFrameRenderer):
    """Use Vulkan for auto/GPU settings and Raster for explicit CPU or fallback."""

    def __init__(self, scene: CompiledScene):
        super().__init__(scene)
        self._gpu = None
        self._last_gpu_report = {}
        if self.requested_backend != "cpu":
            from .rhi import RhiFrameRenderer

            try:
                self._gpu = RhiFrameRenderer(scene, assets=self)
            except GpuBackendError as exc:
                self.render_backend = "cpu"
                if self.requested_backend == "gpu":
                    self.close()
                    raise GpuBackendError(f"已选择 Vulkan GPU 渲染，但 Vulkan 不可用：{exc}") from exc
                self.fallback_reasons.append(str(exc))
            except BaseException:
                super().close()
                raise
            else:
                self.render_backend = "gpu"
                native = self._gpu.backend_report()
                self.gpu_info = dict(native.get("gpu_info", {}))
                self.gpu_texture_cache_budget_bytes = native.get(
                    "gpu_texture_cache_budget_bytes", self.cache_limit)

    def backend_report(self) -> dict:
        native = self._last_gpu_report if self._gpu is None else self._gpu.backend_report()
        base = super().backend_report()
        report = {**base, **native}
        # The shared core continues evaluating CPU frames after GPU recovery.
        report.update(self._evaluator.report())
        for key in ("requested_render_backend", "render_backend", "render_fallback_reasons",
                    "render_seconds", "rendered_frame_count", "readback_mode",
                    "readback_mode_history", "readback_fallback_reasons"):
            report[key] = base[key]
        report["memory_copy_seconds"] = (native.get("memory_copy_seconds", 0.0)
                                         + base.get("cpu_format_conversion_seconds", 0.0))
        report["memory_copy_bytes"] = (native.get("memory_copy_bytes", 0)
                                       + base.get("cpu_format_conversion_bytes", 0))
        return {**report, "graphics_api": "vulkan" if self.render_backend == "gpu" else "raster",
                "readback_seconds": native.get("readback_seconds", 0.0),
                "requested_readback_mode": (native.get("requested_readback_mode", "rhi-sync")
                                            if self.requested_backend != "cpu" else "cpu")}

    def _gpu_failure(self, error: GpuBackendError) -> None:
        gpu, self._gpu = self._gpu, None
        if gpu is not None:
            self._last_gpu_report = gpu.backend_report()
            try:
                gpu.close()
            except GpuBackendError as cleanup_error:
                error.add_note(f"Vulkan 清理失败：{cleanup_error}")
        if self.requested_backend == "gpu":
            self.close()
            raise error
        self.render_backend = "cpu"
        self.fallback_reasons.append(str(error))

    def _render(self, seconds: float, *, native: bool, count: bool = True) -> QImage:
        if self._closed:
            raise RuntimeError("逐帧绘图器已经关闭。")
        started = clock.perf_counter()
        try:
            if self._gpu is not None:
                try:
                    image = (self._gpu._render(seconds) if native
                             else self._gpu.render_frame(seconds))
                except GpuBackendError as error:
                    self._gpu_failure(error)
                    image = self._render_cpu(seconds)
            else:
                image = self._render_cpu(seconds)
            if count:
                self.frame_count += 1
            return image
        finally:
            self.render_seconds += clock.perf_counter() - started

    def render_frame(self, time: float) -> QImage:
        return self._render(time, native=False)

    def _render_export(self, time: float) -> QImage:
        return self._render(time, native=True, count=False)

    def _record_export_frame(self) -> None:
        self.frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._export_stream is not None:
                self._export_stream.close()
        finally:
            gpu = self._gpu
            self._gpu = None
            try:
                if gpu is not None:
                    try:
                        self._last_gpu_report = gpu.backend_report()
                    finally:
                        gpu.close()
            finally:
                super().close()


def render_frame(scene: CompiledScene, time: float) -> QImage:
    """Convenience entrypoint. Reuse FrameRenderer for preview and video loops."""
    with FrameRenderer(scene) as renderer:
        return renderer.render_frame(time)
