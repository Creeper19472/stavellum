use std::{collections::HashMap, ffi::CString, thread};

use ash::vk;

use super::{
    device::{Gpu, HostBuffer},
    Renderer, Stats,
};
use crate::require;

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
    pub(crate) fn new(width: i32, height: i32, budget: u64) -> Result<Self, String> {
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
}
