use std::collections::HashMap;
use std::ffi::CString;
use std::thread::{self, ThreadId};

use ash::vk;

use crate::{abi::MAX_BATCH, require};
use device::{Gpu, HostBuffer};

mod device;
mod init;
mod readback;
mod report;
mod submit;
mod textures;

pub(crate) use readback::recycle_frame;

#[derive(Default)]
struct Stats {
    owned_copy_seconds: f64,
    total_seconds: f64,
    frames: u64,
    submissions: u64,
    owned_frames: u64,
    copied_frames: u64,
    histogram: [u64; MAX_BATCH],
    batch_peak: usize,
    output_peak: u64,
    staging_peak: u64,
    cpu_readback_peak: u64,
    texture_peak: u64,
    uploaded_bytes: u64,
    upload_seconds: f64,
    prepare_seconds: f64,
    begin_frame_seconds: f64,
    submit_seconds: f64,
    readback_seconds: f64,
    copy_seconds: f64,
    inplace_seconds: f64,
    draws: u64,
    instances: u64,
}

struct Texture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    set: vk::DescriptorSet,
    pool: usize,
    bytes: u64,
}

struct PendingUpload {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

pub(crate) struct Renderer {
    gpu: Gpu,
    thread: ThreadId,
    width: u32,
    height: u32,
    frame_bytes: usize,
    budget: u64,
    bgra: bool,
    sampler: vk::Sampler,
    pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    layout: vk::DescriptorSetLayout,
    pools: Vec<vk::DescriptorPool>,
    sets_in_pool: u32,
    msaa: vk::Image,
    msaa_memory: vk::DeviceMemory,
    msaa_view: vk::ImageView,
    resolve: vk::Image,
    resolve_memory: vk::DeviceMemory,
    resolve_view: vk::ImageView,
    textures: HashMap<u64, Texture>,
    pending: Vec<(u64, PendingUpload)>,
    spare_sets: Vec<(usize, vk::DescriptorSet)>,
    instances: HostBuffer,
    uploads: HostBuffer,
    staging: HostBuffer,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    stats: Stats,
    report: CString,
    failed: bool,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Resources are freed explicitly; the device/instance owners destroy
        // their Vulkan handles afterwards in Gpu field order.
        unsafe {
            let device = &self.gpu.device;
            let _ = device.device_wait_idle();
            for texture in self.textures.values() {
                device.destroy_image_view(texture.view, None);
                device.destroy_image(texture.image, None);
                device.free_memory(texture.memory, None);
            }
            for pool in &self.pools {
                device.destroy_descriptor_pool(*pool, None);
            }
            device.destroy_descriptor_set_layout(self.layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.pass, None);
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.msaa_view, None);
            device.destroy_image_view(self.resolve_view, None);
            device.destroy_image(self.msaa, None);
            device.destroy_image(self.resolve, None);
            device.free_memory(self.msaa_memory, None);
            device.free_memory(self.resolve_memory, None);
            device.unmap_memory(self.instances.memory);
            device.destroy_buffer(self.instances.buffer, None);
            device.free_memory(self.instances.memory, None);
            device.unmap_memory(self.uploads.memory);
            device.destroy_buffer(self.uploads.buffer, None);
            device.free_memory(self.uploads.memory, None);
            device.unmap_memory(self.staging.memory);
            device.destroy_buffer(self.staging.buffer, None);
            device.free_memory(self.staging.memory, None);
            device.destroy_fence(self.fence, None);
            device.free_command_buffers(self.command_pool, &[self.command_buffer]);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl Renderer {
    fn check(&self) -> Result<(), String> {
        require(
            self.thread == thread::current().id(),
            "RHI objects must be used on their creating thread",
        )?;
        require(
            !self.failed,
            "RHI renderer failed; close it and create a new instance",
        )
    }

    pub(crate) fn frame_bytes(&self) -> usize {
        self.frame_bytes
    }

    pub(crate) fn thread_id(&self) -> ThreadId {
        self.thread
    }

    pub(crate) fn record_total_seconds(&mut self, seconds: f64) {
        self.stats.total_seconds += seconds;
    }
}
