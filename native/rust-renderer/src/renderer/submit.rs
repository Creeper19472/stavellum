use std::{ptr, slice, time::Instant};

use ash::vk;

use super::Renderer;
use crate::{
    abi::{Quad, MAX_BATCH, MAX_QUADS},
    require,
};

struct Run {
    texture: u64,
    first: u32,
    count: u32,
}

impl Renderer {
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
    pub(crate) fn render_batch(&mut self, frames: &[&[Quad]]) -> Result<(), String> {
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
}
