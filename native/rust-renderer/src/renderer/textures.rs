use std::{ptr, slice, time::Instant};

use ash::vk;

use super::{PendingUpload, Renderer, Texture};
use crate::require;

impl Renderer {
    pub(super) fn texture_bytes(&self) -> u64 {
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

    pub(crate) fn upload(
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

    pub(crate) fn remove(&mut self, id: u64) -> Result<(), String> {
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
    pub(super) fn stage_pending(&mut self) -> Result<Vec<(u64, vk::DeviceSize, u32, u32)>, String> {
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

    pub(super) fn has_pending_texture(&self, id: &u64) -> bool {
        self.pending.iter().any(|(key, _)| key == id)
    }
}
