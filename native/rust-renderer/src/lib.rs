//! Direct-Vulkan (ash) offscreen quad compositor behind the `sprhi_*` ABI v3.
//!
//! Performance model, in contrast to the historical Qt QRhi backend:
//! - one instance (48 bytes) per quad; the six corner vertices are expanded
//!   in the vertex shader from gl_VertexIndex instead of 6x32 CPU bytes,
//! - instances, texture uploads and readbacks flow through persistently
//!   mapped host buffers reused across submissions,
//! - readbacks use tight rows (bufferRowLength = width), so no CPU
//!   depitching pass is ever needed,
//! - up to eight frames share one command buffer, one queue submission and
//!   one fence wait; pending texture uploads ride the same submission,
//! - adjacent quads with one texture collapse into a single instanced draw.
//!
//! All entry points except `sprhi_release_frame` are confined to the creating
//! thread. Validation errors raised before any GPU work leaves the renderer
//! usable; failures after submission poison it, matching the C++ contract.

#![recursion_limit = "256"]

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ops::Deref;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::Mutex;
use std::thread::{self, ThreadId};
use std::time::Instant;

use ash::vk;

const MAX_BATCH: usize = 8;
const MAX_QUADS: usize = 1_000_000;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Quad {
    pub texture_id: u64,
    pub values: [f32; 12],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BatchItem {
    pub quads: *const Quad,
    pub count: usize,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct Frame {
    pub owner: *mut c_void,
    pub pixels: *mut u8,
    pub size: usize,
}

struct Run {
    texture: u64,
    first: u32,
    count: u32,
}

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

thread_local! {
    static ERROR: RefCell<CString> = RefCell::new(CString::default());
}

/// Reusable CPU frame allocations. Heap allocation of multi-megabyte frames
/// per submit costs far more than the copy itself (fresh commit + zeroing),
/// so released frames are recycled across every renderer and thread.
struct FramePool {
    spare: Vec<Vec<u8>>,
    bytes: usize,
    limit: usize,
}

static FRAME_POOL: Mutex<FramePool> = Mutex::new(FramePool {
    spare: Vec::new(),
    bytes: 0,
    limit: 384 * 1024 * 1024,
});

fn pooled_frame(size: usize) -> Vec<u8> {
    if let Ok(mut pool) = FRAME_POOL.lock() {
        if let Some(position) = pool
            .spare
            .iter()
            .rposition(|frame| frame.capacity() >= size)
        {
            let mut frame = pool.spare.swap_remove(position);
            pool.bytes -= frame.capacity();
            frame.clear();
            frame.resize(size, 0);
            return frame;
        }
    }
    vec![0u8; size]
}

fn recycle_frame(frame: Vec<u8>) {
    if let Ok(mut pool) = FRAME_POOL.lock() {
        if pool.bytes + frame.capacity() <= pool.limit && pool.spare.len() < 64 {
            pool.bytes += frame.capacity();
            pool.spare.push(frame);
        }
    }
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

fn guarded(operation: impl FnOnce() -> Result<(), String>) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(operation));
    let error = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(message)) => Some(message),
        Err(panic) => {
            let text = if let Some(message) = panic.downcast_ref::<String>() {
                message.clone()
            } else if let Some(message) = panic.downcast_ref::<&str>() {
                (*message).to_owned()
            } else {
                "Unexpected native renderer failure".to_owned()
            };
            Some(text)
        }
    };
    let status = i32::from(error.is_some());
    ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(error.unwrap_or_default().replace('\0', " ")).unwrap();
    });
    status
}

// ash handles do not destroy Vulkan objects when dropped. These owners also
// release partially initialized GPU state when creation returns an error.
struct DeviceOwner(ash::Device);

impl Deref for DeviceOwner {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for DeviceOwner {
    fn drop(&mut self) {
        unsafe { self.0.destroy_device(None) };
    }
}

struct InstanceOwner(ash::Instance);

impl Deref for InstanceOwner {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for InstanceOwner {
    fn drop(&mut self) {
        unsafe { self.0.destroy_instance(None) };
    }
}

// Drop the device before the instance, then the loader entry point.
struct Gpu {
    device: DeviceOwner,
    instance: InstanceOwner,
    _entry: ash::Entry,
    physical: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family: u32,
    name: String,
    device_local: u32,
    host_visible: u32,
    readback: u32,
    max_texture_dimension: u32,
}

impl Gpu {
    fn create() -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|error| format!("Cannot load the Vulkan loader: {error}"))?;
        let application = vk::ApplicationInfo::default()
            .application_name(c"stavellum")
            .api_version(vk::make_api_version(0, 1, 0, 0));
        let info = vk::InstanceCreateInfo::default().application_info(&application);
        let instance = InstanceOwner(
            unsafe { entry.create_instance(&info, None) }
                .map_err(|_| "Cannot create Vulkan instance".to_owned())?,
        );
        let adapters = unsafe { instance.enumerate_physical_devices() }
            .map_err(|_| "Cannot initialize requested RHI backend".to_owned())?;
        let preferred = std::env::var("STAVELLUM_RHI_GPU").unwrap_or_default();
        let describe = |adapter: vk::PhysicalDevice| unsafe {
            let properties = instance.get_physical_device_properties(adapter);
            (
                properties.device_type,
                CStr::from_ptr(properties.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        let mut chosen: Option<(vk::PhysicalDevice, String)> = None;
        for adapter in &adapters {
            let (kind, name) = describe(*adapter);
            if kind == vk::PhysicalDeviceType::CPU {
                continue;
            }
            if !preferred.is_empty() {
                if name.to_lowercase().contains(&preferred.to_lowercase()) {
                    chosen = Some((*adapter, name));
                    break;
                }
            } else if kind == vk::PhysicalDeviceType::DISCRETE_GPU {
                chosen = Some((*adapter, name));
                break;
            }
        }
        if chosen.is_none() && preferred.is_empty() {
            chosen = adapters
                .iter()
                .find(|adapter| describe(**adapter).0 != vk::PhysicalDeviceType::CPU)
                .map(|adapter| (*adapter, describe(*adapter).1));
        }
        let (physical, name) = chosen.ok_or_else(|| {
            if preferred.is_empty() {
                "Vulkan GPU uses a software device"
            } else {
                "Requested Vulkan GPU unavailable"
            }
            .to_owned()
        })?;
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_family = families
            .iter()
            .position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or_else(|| "Cannot initialize requested RHI backend".to_owned())?
            as u32;
        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities);
        let queue_infos = [queue_info];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
        let device = DeviceOwner(
            unsafe { instance.create_device(physical, &device_info, None) }
                .map_err(|_| "Cannot initialize requested RHI backend".to_owned())?,
        );
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let memory = unsafe { instance.get_physical_device_memory_properties(physical) };
        let find = |wanted: vk::MemoryPropertyFlags| {
            (0..memory.memory_type_count)
                .map(|index| (index, memory.memory_types[index as usize].property_flags))
                .find(|(_, flags)| flags.contains(wanted))
                .map(|(index, _)| index)
                .ok_or_else(|| "Cannot initialize requested RHI backend".to_owned())
        };
        Ok(Self {
            device,
            instance,
            _entry: entry,
            physical,
            queue,
            queue_family,
            name,
            device_local: find(vk::MemoryPropertyFlags::DEVICE_LOCAL)?,
            host_visible: find(
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?,
            // Discrete GPUs map plain host-visible memory write-combined;
            // reading pixels back through it crawls. Prefer a cached type.
            readback: (0..memory.memory_type_count)
                .map(|index| (index, memory.memory_types[index as usize].property_flags))
                .find(|(_, flags)| {
                    flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT
                            | vk::MemoryPropertyFlags::HOST_CACHED,
                    )
                })
                .map(|(index, _)| index)
                .unwrap_or_else(|| {
                    (0..memory.memory_type_count)
                        .map(|index| (index, memory.memory_types[index as usize].property_flags))
                        .find(|(_, flags)| {
                            flags.contains(
                                vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                            )
                        })
                        .map(|(index, _)| index)
                        .expect("host visible type already found")
                }),
            max_texture_dimension: properties.limits.max_image_dimension2_d,
        })
    }
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

/// A persistently mapped host buffer that only ever grows between submissions.
struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pointer: *mut u8,
    capacity: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_type: u32,
}

impl HostBuffer {
    fn new(
        gpu: &Gpu,
        usage: vk::BufferUsageFlags,
        capacity: vk::DeviceSize,
        memory_type: u32,
    ) -> Result<Self, String> {
        let (buffer, memory, pointer) = unsafe {
            let info = vk::BufferCreateInfo::default().size(capacity).usage(usage);
            let buffer = gpu
                .device
                .create_buffer(&info, None)
                .map_err(|_| "Cannot create RHI host buffer".to_owned())?;
            let requirements = gpu.device.get_buffer_memory_requirements(buffer);
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type);
            let memory = gpu
                .device
                .allocate_memory(&info, None)
                .map_err(|_| "Cannot create RHI host buffer".to_owned())?;
            gpu.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|_| "Cannot create RHI host buffer".to_owned())?;
            let pointer = gpu
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|_| "Cannot create RHI host buffer".to_owned())?;
            (buffer, memory, pointer.cast::<u8>())
        };
        Ok(Self {
            buffer,
            memory,
            pointer,
            capacity,
            usage,
            memory_type,
        })
    }

    fn ensure(&mut self, gpu: &Gpu, needed: vk::DeviceSize) -> Result<(), String> {
        if needed <= self.capacity {
            return Ok(());
        }
        // Submissions are synchronous end to end; the previous command buffer
        // has fully completed before any host buffer is replaced.
        let mut size = self.capacity.max(1);
        while size < needed {
            size = size.saturating_mul(2);
        }
        let replacement = HostBuffer::new(gpu, self.usage, size, self.memory_type)?;
        unsafe {
            gpu.device.destroy_buffer(self.buffer, None);
            gpu.device.unmap_memory(self.memory);
            gpu.device.free_memory(self.memory, None);
        }
        *self = replacement;
        Ok(())
    }
}

struct Renderer {
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

fn instance_attributes() -> [vk::VertexInputAttributeDescription; 3] {
    let attribute = |location, offset| {
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(location)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(offset)
    };
    [attribute(0, 0), attribute(1, 16), attribute(2, 32)]
}

impl Renderer {
    fn new(width: i32, height: i32, budget: u64) -> Result<Self, String> {
        require(
            width > 0 && height > 0 && width <= 32768 && height <= 32768,
            "Invalid frame dimensions",
        )?;
        let gpu = Gpu::create()?;
        let width = width as u32;
        let height = height as u32;
        let frame_bytes = width as usize * height as usize * 4;
        let device = &gpu.device;

        // The multisample target only renders; the resolve target only
        // transfers, so each is probed with exactly its own usage.
        let supports = |format, usage, samples| unsafe {
            gpu.instance
                .get_physical_device_image_format_properties(
                    gpu.physical,
                    format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    usage,
                    vk::ImageCreateFlags::empty(),
                )
                .map(|properties| properties.sample_counts.contains(samples))
                .unwrap_or(false)
        };
        // Any nonzero value enables the diagnostic path, matching the C++
        // qEnvironmentVariableIntValue semantics.
        let force_rgba = std::env::var("STAVELLUM_RHI_RGBA_READBACK")
            .ok()
            .and_then(|value| value.trim().parse::<i64>().ok())
            .is_some_and(|value| value != 0);
        let usable = |format| {
            supports(
                format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::SampleCountFlags::TYPE_4,
            ) && supports(
                format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
                vk::SampleCountFlags::TYPE_1,
            )
        };
        let bgra = !force_rgba && usable(vk::Format::B8G8R8A8_UNORM);
        require(
            bgra || usable(vk::Format::R8G8B8A8_UNORM),
            "Requested RHI backend cannot provide 4xMSAA",
        )?;
        let format = if bgra {
            vk::Format::B8G8R8A8_UNORM
        } else {
            vk::Format::R8G8B8A8_UNORM
        };

        let sampler = unsafe {
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            device.create_sampler(&info, None)
        }
        .map_err(|_| "Cannot create RHI sampler".to_owned())?;

        let attachment = |samples| {
            vk::AttachmentDescription::default()
                .format(format)
                .samples(samples)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        };
        let msaa_attachment =
            attachment(vk::SampleCountFlags::TYPE_4).load_op(vk::AttachmentLoadOp::CLEAR);
        let resolve_attachment =
            attachment(vk::SampleCountFlags::TYPE_1).store_op(vk::AttachmentStoreOp::STORE);
        let color_reference = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let resolve_reference = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_references = [color_reference];
        let resolve_references = [resolve_reference];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_references)
            .resolve_attachments(&resolve_references);
        let attachments = [msaa_attachment, resolve_attachment];
        let subpasses = [subpass];
        let pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);
        let pass = unsafe { device.create_render_pass(&pass_info, None) }
            .map_err(|_| "Cannot create RHI render pass".to_owned())?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|_| "Cannot create RHI descriptor layout".to_owned())?;

        let push = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(8);
        let push_ranges = [push];
        let set_layouts = [layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )
        }
        .map_err(|_| "Cannot create RHI pipeline layout".to_owned())?;

        const VERTEX: &[u32] = &include!(concat!(env!("OUT_DIR"), "/quad.vs_main.words"));
        const FRAGMENT: &[u32] = &include!(concat!(env!("OUT_DIR"), "/quad.fs_main.words"));
        let module = |words: &[u32]| unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
        };
        let vertex = module(VERTEX).map_err(|_| "Cannot create RHI vertex shader".to_owned())?;
        let fragment =
            module(FRAGMENT).map_err(|_| "Cannot create RHI fragment shader".to_owned())?;
        let vertex_entry = c"vs_main";
        let fragment_entry = c"fs_main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex)
                .name(vertex_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment)
                .name(fragment_entry),
        ];
        let instance_binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(48)
            .input_rate(vk::VertexInputRate::INSTANCE);
        let instance_bindings = [instance_binding];
        let attributes = instance_attributes();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&instance_bindings)
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::Viewport::default()
            .width(width as f32)
            .height(height as f32)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(vk::Extent2D { width, height });
        let viewports = [viewport];
        let scissors = [scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_4);
        let blend = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let blends = [blend];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(pipeline_layout)
            .render_pass(pass);
        let pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|_| "Cannot create RHI graphics pipeline".to_owned())?;
        let pipeline = pipelines[0];
        unsafe {
            device.destroy_shader_module(fragment, None);
            device.destroy_shader_module(vertex, None);
        }

        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let create_image = |samples, usage| unsafe {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(samples)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            device.create_image(&info, None)
        };
        let msaa = create_image(
            vk::SampleCountFlags::TYPE_4,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        )
        .map_err(|_| "Cannot allocate 4xMSAA color buffer".to_owned())?;
        let resolve = create_image(
            vk::SampleCountFlags::TYPE_1,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .map_err(|_| "Cannot allocate resolve texture".to_owned())?;
        let bind_image = |image| unsafe {
            let requirements = device.get_image_memory_requirements(image);
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(gpu.device_local);
            let memory = device
                .allocate_memory(&info, None)
                .map_err(|_| "Cannot allocate RHI render target".to_owned())?;
            device
                .bind_image_memory(image, memory, 0)
                .map_err(|_| "Cannot allocate RHI render target".to_owned())?;
            Ok::<vk::DeviceMemory, String>(memory)
        };
        let msaa_memory = bind_image(msaa)?;
        let resolve_memory = bind_image(resolve)?;
        let view = |image| unsafe {
            let subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(subresource),
                None,
            )
        };
        let msaa_view = view(msaa).map_err(|_| "Cannot create RHI render target".to_owned())?;
        let resolve_view =
            view(resolve).map_err(|_| "Cannot create RHI render target".to_owned())?;
        let framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(pass)
                    .attachments(&[msaa_view, resolve_view])
                    .width(width)
                    .height(height)
                    .layers(1),
                None,
            )
        }
        .map_err(|_| "Cannot create RHI render target".to_owned())?;

        let instances = HostBuffer::new(
            &gpu,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            64 * 1024,
            gpu.host_visible,
        )?;
        let uploads = HostBuffer::new(
            &gpu,
            vk::BufferUsageFlags::TRANSFER_SRC,
            64 * 1024,
            gpu.host_visible,
        )?;
        let staging = HostBuffer::new(
            &gpu,
            vk::BufferUsageFlags::TRANSFER_DST,
            frame_bytes as vk::DeviceSize,
            gpu.readback,
        )?;

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(gpu.queue_family),
                None,
            )
        }
        .map_err(|_| "Cannot create RHI command pool".to_owned())?;
        let command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|_| "Cannot create RHI command buffer".to_owned())?[0];
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|_| "Cannot create RHI fence".to_owned())?;

        let mut renderer = Self {
            gpu,
            thread: thread::current().id(),
            width,
            height,
            frame_bytes,
            budget,
            bgra,
            sampler,
            pass,
            framebuffer,
            pipeline_layout,
            pipeline,
            layout,
            pools: Vec::new(),
            sets_in_pool: 0,
            msaa,
            msaa_memory,
            msaa_view,
            resolve,
            resolve_memory,
            resolve_view,
            textures: HashMap::new(),
            pending: Vec::new(),
            spare_sets: Vec::new(),
            instances,
            uploads,
            staging,
            command_pool,
            command_buffer,
            fence,
            stats: Stats::default(),
            report: CString::default(),
            failed: false,
        };
        let white = [255u8, 255, 255, 255];
        renderer.upload(0, 1, 1, 4, white.as_ptr())?;
        renderer.stats.uploaded_bytes = 0;
        renderer.stats.upload_seconds = 0.0;
        Ok(renderer)
    }

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

    fn texture_bytes(&self) -> u64 {
        self.textures
            .values()
            .map(|texture| texture.bytes)
            .sum::<u64>()
            + self
                .pending
                .iter()
                .map(|(_, upload)| upload.data.len() as u64)
                .sum::<u64>()
    }

    fn upload(
        &mut self,
        id: u64,
        width: i32,
        height: i32,
        stride: i32,
        pixels: *const u8,
    ) -> Result<(), String> {
        self.check()?;
        require(
            !pixels.is_null()
                && width > 0
                && height > 0
                && width as u32 <= self.gpu.max_texture_dimension
                && height as u32 <= self.gpu.max_texture_dimension,
            "Invalid texture size",
        )?;
        require(stride as i64 >= width as i64 * 4, "Invalid texture stride")?;
        require(
            !self.textures.contains_key(&id) && !self.has_pending_texture(&id),
            "Texture ID already uploaded",
        )?;
        let started = Instant::now();
        let width = width as usize;
        let height = height as usize;
        let stride = stride as usize;
        let source = unsafe { slice::from_raw_parts(pixels, (height - 1) * stride + width * 4) };
        // Tight-pack rows now; the GPU copy at submit time stays a single
        // linear transfer regardless of the caller's row alignment.
        let mut data = Vec::with_capacity(width * height * 4);
        for row in 0..height {
            data.extend_from_slice(&source[row * stride..row * stride + width * 4]);
        }
        let bytes = (width * height * 4) as u64;
        self.stats.texture_peak = self.stats.texture_peak.max(self.texture_bytes() + bytes);
        self.stats.uploaded_bytes += bytes;
        self.pending.push((
            id,
            PendingUpload {
                width: width as u32,
                height: height as u32,
                data,
            },
        ));
        self.stats.upload_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    fn remove(&mut self, id: u64) -> Result<(), String> {
        self.check()?;
        require(id != 0, "Cannot delete the solid white texture")?;
        if let Some(position) = self.pending.iter().position(|(key, _)| *key == id) {
            self.pending.remove(position);
            return Ok(());
        }
        if let Some(texture) = self.textures.remove(&id) {
            unsafe {
                // The descriptor set stays allocated and is recycled above;
                // upload and eviction churn must not grow pools unbounded.
                self.spare_sets.push((texture.pool, texture.set));
                self.gpu.device.destroy_image_view(texture.view, None);
                self.gpu.device.destroy_image(texture.image, None);
                self.gpu.device.free_memory(texture.memory, None);
            }
        }
        Ok(())
    }

    /// Allocate one set per texture (sampled image + shared sampler) from a
    /// growing pool.
    fn allocate_set(&mut self, view: vk::ImageView) -> Result<(vk::DescriptorSet, usize), String> {
        const POOL_SETS: u32 = 64;
        let device = &self.gpu.device;
        if let Some((pool, set)) = self.spare_sets.pop() {
            self.write_texture_bindings(set, view);
            return Ok((set, pool));
        }
        if self.pools.is_empty() || self.sets_in_pool == POOL_SETS {
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLED_IMAGE)
                    .descriptor_count(POOL_SETS),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::SAMPLER)
                    .descriptor_count(POOL_SETS),
            ];
            let info = vk::DescriptorPoolCreateInfo::default()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .max_sets(POOL_SETS)
                .pool_sizes(&sizes);
            let pool = unsafe { device.create_descriptor_pool(&info, None) }
                .map_err(|_| "Cannot create RHI descriptor pool".to_owned())?;
            self.pools.push(pool);
            self.sets_in_pool = 0;
        }
        let pool = *self.pools.last().expect("pool just ensured");
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&[self.layout]),
            )
        }
        .map_err(|_| "Cannot create texture bindings".to_owned())?;
        self.sets_in_pool += 1;
        self.write_texture_bindings(sets[0], view);
        Ok((sets[0], self.pools.len() - 1))
    }

    fn write_texture_bindings(&self, set: vk::DescriptorSet, view: vk::ImageView) {
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let sampler_info = vk::DescriptorImageInfo::default().sampler(self.sampler);
        unsafe {
            self.gpu.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[image_info]),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&[sampler_info]),
                ],
                &[],
            );
        }
    }

    /// Create GPU images for pending uploads and stage their bytes.
    fn stage_pending(&mut self) -> Result<Vec<(u64, vk::DeviceSize, u32, u32)>, String> {
        let mut pending = std::mem::take(&mut self.pending);
        // Upload order is irrelevant (every texture is independent), but a
        // mid-batch failure must hand the not-yet-staged uploads back: their
        // sprhi_upload calls already reported success.
        let outcome = self.stage_pending_inner(&mut pending);
        if outcome.is_err() {
            self.pending.extend(pending);
        }
        outcome
    }

    fn stage_pending_inner(
        &mut self,
        pending: &mut Vec<(u64, PendingUpload)>,
    ) -> Result<Vec<(u64, vk::DeviceSize, u32, u32)>, String> {
        let mut placed = Vec::new();
        let mut offset = 0 as vk::DeviceSize;
        while let Some((id, upload)) = pending.pop() {
            let extent = vk::Extent3D {
                width: upload.width,
                height: upload.height,
                depth: 1,
            };
            let (image, memory, view) = {
                let device = &self.gpu.device;
                let info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .extent(extent)
                    .mip_levels(1)
                    .array_layers(1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                let image = unsafe { device.create_image(&info, None) }
                    .map_err(|_| "Cannot allocate RHI texture".to_owned())?;
                let memory = unsafe {
                    let requirements = device.get_image_memory_requirements(image);
                    let info = vk::MemoryAllocateInfo::default()
                        .allocation_size(requirements.size)
                        .memory_type_index(self.gpu.device_local);
                    let memory = device
                        .allocate_memory(&info, None)
                        .map_err(|_| "Cannot allocate RHI texture".to_owned())?;
                    device
                        .bind_image_memory(image, memory, 0)
                        .map_err(|_| "Cannot allocate RHI texture".to_owned())?;
                    memory
                };
                let view = unsafe {
                    let subresource = vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1);
                    device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::default()
                                .image(image)
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(vk::Format::R8G8B8A8_UNORM)
                                .subresource_range(subresource),
                            None,
                        )
                        .map_err(|_| "Cannot create RHI texture view".to_owned())?
                };
                (image, memory, view)
            };
            let (set, pool) = self.allocate_set(view)?;
            unsafe {
                ptr::copy_nonoverlapping(
                    upload.data.as_ptr(),
                    self.uploads.pointer.add(offset as usize),
                    upload.data.len(),
                );
            }
            placed.push((id, offset, upload.width, upload.height));
            offset += upload.data.len() as vk::DeviceSize;
            self.textures.insert(
                id,
                Texture {
                    image,
                    memory,
                    view,
                    set,
                    pool,
                    bytes: upload.data.len() as u64,
                },
            );
        }
        Ok(placed)
    }

    fn validate(&self, frames: &[&[Quad]]) -> Result<(), String> {
        let total = frames.iter().map(|frame| frame.len()).sum::<usize>();
        require(
            total <= MAX_QUADS * MAX_BATCH,
            "Batch vertex buffer is too large",
        )?;
        for frame in frames {
            require(frame.len() <= MAX_QUADS, "Too many frame commands")?;
            for quad in *frame {
                require(
                    self.textures.contains_key(&quad.texture_id)
                        || self.has_pending_texture(&quad.texture_id),
                    "Frame refers to missing texture",
                )?;
                let values = &quad.values;
                require(
                    values[2] >= 0.0
                        && values[3] >= 0.0
                        && values[0].is_finite()
                        && values[1].is_finite()
                        && values[2].is_finite()
                        && values[3].is_finite(),
                    "Invalid quad geometry",
                )?;
                require(
                    values[4..12].iter().all(|value| value.is_finite()),
                    "Invalid quad attribute",
                )?;
            }
        }
        Ok(())
    }

    fn has_pending_texture(&self, id: &u64) -> bool {
        self.pending.iter().any(|(key, _)| key == id)
    }

    /// Copy the 48-byte payload of every quad into the mapped instance ring
    /// and merge adjacent same-texture quads into instanced draw runs.
    fn pack(&mut self, frames: &[&[Quad]]) -> Vec<Vec<Run>> {
        let mut runs = Vec::with_capacity(frames.len());
        let mut cursor = 0usize;
        for frame in frames {
            let mut frame_runs: Vec<Run> = Vec::new();
            for quad in *frame {
                unsafe {
                    ptr::copy_nonoverlapping(
                        quad.values.as_ptr() as *const u8,
                        self.instances.pointer.add(cursor * 48),
                        48,
                    );
                }
                cursor += 1;
                match frame_runs.last_mut() {
                    Some(run) if run.texture == quad.texture_id => run.count += 1,
                    _ => frame_runs.push(Run {
                        texture: quad.texture_id,
                        first: (cursor - 1) as u32,
                        count: 1,
                    }),
                }
            }
            runs.push(frame_runs);
        }
        self.stats.instances += cursor as u64;
        self.stats.draws += runs.iter().map(|frame| frame.len() as u64).sum::<u64>();
        runs
    }

    /// Render up to eight frames with one command buffer and one fence wait.
    fn render_batch(&mut self, frames: &[&[Quad]]) -> Result<(), String> {
        self.check()?;
        let count = frames.len();
        let started = Instant::now();
        self.validate(frames)?;
        let total = frames.iter().map(|frame| frame.len()).sum::<usize>();
        let staging_bytes = count as u64 * self.frame_bytes as u64;
        let upload_bytes = self
            .pending
            .iter()
            .map(|(_, upload)| upload.data.len() as u64)
            .sum::<u64>();
        self.instances
            .ensure(&self.gpu, (total as u64 * 48).max(1))?;
        self.staging.ensure(&self.gpu, staging_bytes.max(1))?;
        self.uploads.ensure(&self.gpu, upload_bytes.max(1))?;
        let uploads = self.stage_pending()?;
        // Pack after buffer growth: growth replaces the mapped instance ring.
        let runs = self.pack(frames);
        self.stats.prepare_seconds += started.elapsed().as_secs_f64();

        // Any failure from here on leaves partially recorded or submitted GPU
        // work behind; reuse after that is refused exactly like the C++ RHI.
        self.failed = true;
        let started = Instant::now();
        let device = &self.gpu.device;
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(|_| "Cannot begin RHI command buffer".to_owned())?;
            device
                .begin_command_buffer(
                    self.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|_| "Cannot begin RHI command buffer".to_owned())?;
        }
        self.stats.begin_frame_seconds += started.elapsed().as_secs_f64();
        let started = Instant::now();
        unsafe {
            let command = self.command_buffer;
            let layers = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1);
            let range = || {
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1)
            };
            for (id, offset, width, height) in &uploads {
                let texture = &self.textures[id];
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(texture.image)
                        .src_access_mask(vk::AccessFlags::NONE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .subresource_range(range())],
                );
                device.cmd_copy_buffer_to_image(
                    command,
                    self.uploads.buffer,
                    texture.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[vk::BufferImageCopy::default()
                        .buffer_offset(*offset)
                        .buffer_row_length(0)
                        .buffer_image_height(0)
                        .image_subresource(layers)
                        .image_extent(vk::Extent3D {
                            width: *width,
                            height: *height,
                            depth: 1,
                        })],
                );
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(texture.image)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .subresource_range(range())],
                );
            }
            let size = [self.width as f32, self.height as f32];
            device.cmd_push_constants(
                command,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                slice::from_raw_parts(size.as_ptr() as *const u8, 8),
            );
            for (index, frame_runs) in runs.iter().enumerate() {
                let clear = vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                };
                device.cmd_begin_render_pass(
                    command,
                    &vk::RenderPassBeginInfo::default()
                        .render_pass(self.pass)
                        .framebuffer(self.framebuffer)
                        .render_area(vk::Rect2D::default().extent(vk::Extent2D {
                            width: self.width,
                            height: self.height,
                        }))
                        .clear_values(&[clear]),
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                device.cmd_bind_vertex_buffers(command, 0, &[self.instances.buffer], &[0]);
                for run in frame_runs {
                    device.cmd_bind_descriptor_sets(
                        command,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_layout,
                        0,
                        &[self.textures[&run.texture].set],
                        &[],
                    );
                    device.cmd_draw(command, 6, run.count, 0, run.first);
                }
                device.cmd_end_render_pass(command);
                // Make the resolve a transfer source, pull it into this
                // frame's tight staging slice, then hand it back for the next
                // pass's resolve write.
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(self.resolve)
                        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .subresource_range(range())],
                );
                device.cmd_copy_image_to_buffer(
                    command,
                    self.resolve,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.staging.buffer,
                    &[vk::BufferImageCopy::default()
                        .buffer_offset(index as u64 * self.frame_bytes as u64)
                        .buffer_row_length(self.width)
                        .buffer_image_height(self.height)
                        .image_subresource(layers)
                        .image_extent(vk::Extent3D {
                            width: self.width,
                            height: self.height,
                            depth: 1,
                        })],
                );
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .image(self.resolve)
                        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .subresource_range(range())],
                );
            }
            device
                .end_command_buffer(command)
                .map_err(|_| "Cannot finish RHI frame".to_owned())?;
            let commands = [command];
            let submit = vk::SubmitInfo::default().command_buffers(&commands);
            device
                .queue_submit(self.gpu.queue, &[submit], self.fence)
                .map_err(|_| "Cannot submit RHI frame".to_owned())?;
        }
        self.stats.submit_seconds += started.elapsed().as_secs_f64();
        let started = Instant::now();
        unsafe {
            device
                .wait_for_fences(&[self.fence], true, u64::MAX)
                .map_err(|_| "Cannot finish RHI frame".to_owned())?;
            device
                .reset_fences(&[self.fence])
                .map_err(|_| "Cannot finish RHI frame".to_owned())?;
        }
        self.stats.readback_seconds += started.elapsed().as_secs_f64();
        let batch_bytes = count as u64 * self.frame_bytes as u64;
        self.stats.output_peak = self.stats.output_peak.max(batch_bytes);
        self.stats.staging_peak = self.stats.staging_peak.max(batch_bytes);
        self.stats.cpu_readback_peak = self.stats.cpu_readback_peak.max(batch_bytes);
        self.stats.frames += count as u64;
        self.stats.submissions += 1;
        self.stats.histogram[count - 1] += 1;
        self.stats.batch_peak = self.stats.batch_peak.max(count);
        self.failed = false;
        Ok(())
    }

    fn owned_outputs(&mut self, count: usize) -> Result<Vec<Frame>, String> {
        let timing = Instant::now();
        let size = self.frame_bytes;
        let mut outputs = Vec::with_capacity(count);
        for index in 0..count {
            let mut pixels = pooled_frame(size);
            unsafe {
                ptr::copy_nonoverlapping(
                    self.staging.pointer.add(index * size),
                    pixels.as_mut_ptr(),
                    size,
                );
            }
            if !self.bgra {
                let started = Instant::now();
                swizzle_rgba_to_bgra(&mut pixels);
                self.stats.inplace_seconds += started.elapsed().as_secs_f64();
            }
            let mut owner = Box::new(pixels);
            outputs.push(Frame {
                pixels: owner.as_mut_ptr(),
                size: owner.len(),
                owner: (&mut *owner as *mut Vec<u8>).cast(),
            });
            let _ = Box::into_raw(owner);
        }
        self.stats.owned_frames += count as u64;
        self.stats.owned_copy_seconds += timing.elapsed().as_secs_f64();
        Ok(outputs)
    }

    fn copy_output(&mut self, pixels: *mut u8, capacity: usize) -> Result<(), String> {
        require(
            !pixels.is_null() && capacity >= self.frame_bytes,
            "Output pixel buffer is too small",
        )?;
        let size = self.frame_bytes;
        let started = Instant::now();
        let target = unsafe { slice::from_raw_parts_mut(pixels, size) };
        let source = unsafe { slice::from_raw_parts(self.staging.pointer, size) };
        target.copy_from_slice(source);
        if !self.bgra {
            swizzle_rgba_to_bgra(target);
        }
        self.stats.copy_seconds += started.elapsed().as_secs_f64();
        self.stats.copied_frames += 1;
        self.stats.cpu_readback_peak = self.stats.cpu_readback_peak.max(size as u64 * 2);
        Ok(())
    }

    fn report(&mut self) -> *const c_char {
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

#[cfg(target_arch = "x86_64")]
fn swizzle_rgba_to_bgra(pixels: &mut [u8]) {
    use std::arch::is_x86_feature_detected;
    let count = pixels.len() / 4;
    if count >= 8 && is_x86_feature_detected!("avx2") {
        unsafe { swizzle_avx2(pixels.as_mut_ptr(), pixels.len()) };
        return;
    }
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn swizzle_avx2(pointer: *mut u8, bytes: usize) {
    use std::arch::x86_64::{
        _mm256_loadu_si256, _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_storeu_si256,
    };
    let shuffle = _mm256_setr_epi8(
        2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11,
        14, 13, 12, 15,
    );
    let mut offset = 0;
    while offset + 32 <= bytes {
        let block = _mm256_loadu_si256(pointer.add(offset) as *const _);
        _mm256_storeu_si256(
            pointer.add(offset) as *mut _,
            _mm256_shuffle_epi8(block, shuffle),
        );
        offset += 32;
    }
    let tail = pointer.add(offset);
    let remaining = bytes - offset;
    for index in (0..remaining).step_by(4) {
        let one = tail.add(index);
        core::ptr::swap(one, one.add(2));
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn swizzle_rgba_to_bgra(pixels: &mut [u8]) {
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
    }
}

fn swizzle_path(bgra: bool) -> &'static str {
    if bgra {
        return "none";
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            "avx2-pshufb"
        } else {
            "scalar-swap"
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        "scalar-swap"
    }
}

struct Handle {
    thread: ThreadId,
    renderer: Mutex<Renderer>,
}

unsafe fn with_renderer(
    pointer: *mut c_void,
    action: impl FnOnce(&mut Renderer) -> Result<(), String>,
) -> Result<(), String> {
    require(!pointer.is_null(), "Null RHI handle")?;
    let handle = &*pointer.cast::<Handle>();
    require(
        handle.thread == thread::current().id(),
        "RHI objects must be used on their creating thread",
    )?;
    let mut renderer = handle
        .renderer
        .lock()
        .map_err(|_| "RHI renderer lock poisoned".to_owned())?;
    action(&mut renderer)
}

unsafe fn quad_slice<'a>(quads: *const Quad, count: usize) -> Result<&'a [Quad], String> {
    require(count <= MAX_QUADS, "Too many frame commands")?;
    require(count == 0 || !quads.is_null(), "Null quad array")?;
    Ok(if count == 0 {
        &[]
    } else {
        slice::from_raw_parts(quads, count)
    })
}

#[no_mangle]
pub extern "C" fn sprhi_abi_version() -> u32 {
    // The default panic hook writes through the CRT stderr of this DLL,
    // which is not initialized when loaded through ctypes; route panics
    // to the trace file instead.
    std::panic::set_hook(Box::new(|information| {
        // The GNU CRT stderr of this DLL is not initialized under ctypes;
        // the default hook would crash writing through it.
        let _ = information.to_string();
    }));
    3
}

/// # Safety
/// Pointer arguments must be valid for the duration of the call; the
/// renderer handle must be used on its creating thread.
#[no_mangle]
pub unsafe extern "C" fn sprhi_create(
    width: i32,
    height: i32,
    budget: u64,
    api: *const c_char,
) -> *mut c_void {
    let mut result = ptr::null_mut();
    guarded(|| {
        require(!api.is_null(), "Null API name")?;
        require(
            CStr::from_ptr(api).to_bytes() == b"vulkan",
            "RHI API must be vulkan",
        )?;
        let renderer = Renderer::new(width, height, budget)?;
        result = Box::into_raw(Box::new(Handle {
            thread: thread::current().id(),
            renderer: Mutex::new(renderer),
        }))
        .cast();
        Ok(())
    });
    result
}

/// # Safety
/// Pointer arguments must be valid for the duration of the call; the
/// renderer handle must be used on its creating thread.
#[no_mangle]
pub unsafe extern "C" fn sprhi_upload(
    pointer: *mut c_void,
    id: u64,
    width: i32,
    height: i32,
    stride: i32,
    pixels: *const u8,
) -> i32 {
    guarded(|| {
        with_renderer(pointer, |renderer| {
            renderer.upload(id, width, height, stride, pixels)
        })
    })
}

/// # Safety
/// Pointer arguments must be valid for the duration of the call; the
/// renderer handle must be used on its creating thread.
#[no_mangle]
pub unsafe extern "C" fn sprhi_remove(pointer: *mut c_void, id: u64) -> i32 {
    guarded(|| with_renderer(pointer, |renderer| renderer.remove(id)))
}

/// # Safety
/// Pointer arguments must be valid for the duration of the call; the
/// renderer handle must be used on its creating thread.
#[no_mangle]
pub unsafe extern "C" fn sprhi_submit(
    pointer: *mut c_void,
    quads: *const Quad,
    count: usize,
    pixels: *mut u8,
    capacity: usize,
) -> i32 {
    guarded(|| {
        with_renderer(pointer, |renderer| {
            require(
                !pixels.is_null() && capacity >= renderer.frame_bytes,
                "Output pixel buffer is too small",
            )?;
            let frame = quad_slice(quads, count)?;
            renderer.render_batch(&[frame])?;
            renderer.copy_output(pixels, capacity)
        })
    })
}

/// # Safety
/// Pointer arguments must be valid for the duration of the call; the
/// renderer handle must be used on its creating thread.
#[no_mangle]
pub unsafe extern "C" fn sprhi_submit_owned(
    pointer: *mut c_void,
    quads: *const Quad,
    count: usize,
    frame: *mut Frame,
) -> i32 {
    if !frame.is_null() {
        ptr::write(frame, Frame::default());
    }
    guarded(|| {
        require(!frame.is_null(), "Null output frame")?;
        with_renderer(pointer, |renderer| {
            let commands = quad_slice(quads, count)?;
            renderer.render_batch(&[commands])?;
            let outputs = renderer.owned_outputs(1)?;
            ptr::write(frame, outputs[0]);
            Ok(())
        })
    })
}

/// # Safety
/// Pointer arguments must be valid for the duration of the call; the
/// renderer handle must be used on its creating thread.
#[no_mangle]
pub unsafe extern "C" fn sprhi_submit_batch_owned(
    pointer: *mut c_void,
    items: *const BatchItem,
    count: usize,
    frames: *mut Frame,
) -> i32 {
    if !frames.is_null() && count <= MAX_BATCH {
        for index in 0..count {
            frames.add(index).write(Frame::default());
        }
    }
    guarded(|| {
        require(
            (1..=MAX_BATCH).contains(&count),
            "Batch frame count must be between 1 and 8",
        )?;
        require(
            !items.is_null() && !frames.is_null(),
            "Null batch item array",
        )?;
        with_renderer(pointer, |renderer| {
            let timing = Instant::now();
            let raw = slice::from_raw_parts(items, count);
            let commands = raw
                .iter()
                .map(|item| quad_slice(item.quads, item.count))
                .collect::<Result<Vec<_>, _>>()?;
            renderer.render_batch(&commands)?;
            let outputs = renderer.owned_outputs(count)?;
            renderer.stats.total_seconds += timing.elapsed().as_secs_f64();
            for (index, output) in outputs.into_iter().enumerate() {
                frames.add(index).write(output);
            }
            Ok(())
        })
    })
}

/// # Safety
/// `owner` must originate from a successful submit call and be released
/// exactly once, on any thread.
#[no_mangle]
pub unsafe extern "C" fn sprhi_release_frame(owner: *mut c_void) {
    if !owner.is_null() {
        recycle_frame(*Box::from_raw(owner.cast::<Vec<u8>>()));
    }
}

/// # Safety
/// Pointer arguments must be valid for the duration of the call; the
/// renderer handle must be used on its creating thread.
#[no_mangle]
pub unsafe extern "C" fn sprhi_report(pointer: *mut c_void) -> *const c_char {
    let mut result = ptr::null();
    guarded(|| {
        with_renderer(pointer, |renderer| {
            require(
                renderer.thread == thread::current().id(),
                "RHI report must use creating thread",
            )?;
            result = renderer.report();
            Ok(())
        })
    });
    result
}

#[no_mangle]
pub extern "C" fn sprhi_last_error(_: *mut c_void) -> *const c_char {
    ERROR.with(|slot| slot.borrow().as_ptr())
}

#[no_mangle]
/// # Safety
/// `api` must point to a NUL-terminated string; the returned handle must be
/// used from the creating thread only.
pub unsafe extern "C" fn sprhi_close(pointer: *mut c_void) -> i32 {
    guarded(|| {
        if !pointer.is_null() {
            // Taking ownership before the thread check would destroy the
            // renderer even when the caller is told the close was refused.
            let handle = &*pointer.cast::<Handle>();
            require(
                handle.thread == thread::current().id(),
                "Close RHI on creating thread",
            )?;
            drop(Box::from_raw(pointer.cast::<Handle>()));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_layout_matches_python() {
        assert_eq!(std::mem::size_of::<Quad>(), 56);
        assert_eq!(std::mem::size_of::<BatchItem>(), 16);
        assert_eq!(std::mem::size_of::<Frame>(), 24);
    }

    #[test]
    fn swizzle_converts_rgba_pairs() {
        let mut pixels = vec![1u8, 2, 3, 255, 10, 20, 30, 128];
        swizzle_rgba_to_bgra(&mut pixels);
        assert_eq!(pixels, vec![3, 2, 1, 255, 30, 20, 10, 128]);
        let mut large: Vec<u8> = (0..4096).map(|index| (index % 251) as u8).collect();
        let mut reference = large.clone();
        swizzle_rgba_to_bgra(&mut large);
        for pixel in reference.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
        }
        assert_eq!(large, reference);
    }

    #[test]
    fn null_quad_pointer_accepted_only_without_commands() {
        unsafe {
            assert!(quad_slice(ptr::null(), 0).unwrap().is_empty());
            assert!(quad_slice(ptr::null(), 1).is_err());
        }
    }
}
