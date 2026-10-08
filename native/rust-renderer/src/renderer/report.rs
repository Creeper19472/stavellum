use std::ffi::{c_char, CString};

use super::{readback::swizzle_path, Renderer};

impl Renderer {
    pub(crate) fn report(&mut self) -> *const c_char {
        let statistics = &self.stats;
        let histogram: serde_json::Map<String, serde_json::Value> = statistics
            .histogram
            .iter()
            .enumerate()
            .map(|(index, count)| ((index + 1).to_string(), serde_json::json!(count)))
            .collect();
        let owned = statistics.owned_frames;
        let copied = statistics.copied_frames;
        let copy_path = if owned > 0 && copied > 0 {
            "mixed-owned-and-copy"
        } else if owned > 0 {
            if self.bgra {
                "bgra-owned-buffer"
            } else {
                "rgba-inplace-sse2-swizzle"
            }
        } else if copied > 0 {
            if self.bgra {
                "bgra-memcpy"
            } else {
                "rgba-sse2-swizzle"
            }
        } else {
            "not-submitted"
        };
        let value = serde_json::json!({
            "graphics_api": "vulkan",
            "renderer_implementation": "rust-ash",
            "native_language": "rust",
            "native_swizzle_path": swizzle_path(self.bgra),
            "qt_version": "rust-native",
            "gpu_info": {"renderer": self.gpu.name, "msaa_samples": 4},
            "gpu_texture_cache_budget_bytes": self.budget,
            "gpu_texture_cache_peak_bytes": statistics.texture_peak,
            "gpu_texture_cache_bytes": self.texture_bytes(),
            "gpu_texture_cache_measured": false,
            "texture_cache_accounting": "logical RGBA bytes; excludes driver allocations",
            "gpu_upload_seconds": statistics.upload_seconds,
            "gpu_upload_bytes": statistics.uploaded_bytes,
            "native_prepare_seconds": statistics.prepare_seconds,
            "native_owned_copy_seconds": statistics.owned_copy_seconds,
            "native_total_seconds": statistics.total_seconds,
            "native_begin_frame_seconds": statistics.begin_frame_seconds,
            "gpu_submit_seconds": statistics.submit_seconds,
            "synchronous_readback_seconds": statistics.readback_seconds,
            "memory_copy_seconds": statistics.copy_seconds,
            "memory_copy_bytes": copied * self.frame_bytes as u64,
            "inplace_format_conversion_seconds": statistics.inplace_seconds,
            "inplace_format_conversion_bytes":
                if self.bgra { 0 } else { owned * self.frame_bytes as u64 },
            "readback_seconds":
                statistics.readback_seconds + statistics.copy_seconds + statistics.inplace_seconds,
            "gpu_timestamp_enabled": false,
            "gpu_timestamp_samples": 0,
            "gpu_execution_seconds": serde_json::Value::Null,
            "gpu_submitted_frame_count": statistics.frames,
            "gpu_submission_count": statistics.submissions,
            "gpu_batch_size_histogram": histogram,
            "gpu_batch_peak_size": statistics.batch_peak,
            "readback_mode": if statistics.batch_peak > 1 { "rhi-batch-sync" } else { "rhi-sync" },
            "owned_readback_frame_count": owned,
            "copied_readback_frame_count": copied,
            "readback_format": if self.bgra { "BGRA8" } else { "RGBA8" },
            "readback_copy_path": copy_path,
            "readback_buffer_peak_bytes": statistics.cpu_readback_peak,
            "readback_output_peak_bytes": statistics.output_peak,
            "readback_staging_peak_bytes": statistics.staging_peak,
            "readback_buffer_accounting":
                "per-submit CPU output estimate; excludes retained caller frames",
            "readback_staging_accounting": "persistently mapped Vulkan readback buffer; tight rows",
            "framebuffer_estimated_bytes": self.width as u64 * self.height as u64 * 20,
            "gpu_draw_call_count": statistics.draws,
            "gpu_instance_count": statistics.instances,
            "instance_stride_bytes": 48,
            "instance_buffer_capacity_bytes": self.instances.capacity,
        });
        self.report = CString::new(value.to_string()).unwrap();
        self.report.as_ptr()
    }
}
