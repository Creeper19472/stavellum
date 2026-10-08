use std::{ffi::CStr, ops::Deref};

use ash::vk;

// ash handles do not destroy Vulkan objects when dropped. These owners also
// release partially initialized GPU state when creation returns an error.
pub(super) struct DeviceOwner(ash::Device);

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

pub(super) struct InstanceOwner(ash::Instance);

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
pub(super) struct Gpu {
    pub(super) device: DeviceOwner,
    pub(super) instance: InstanceOwner,
    _entry: ash::Entry,
    pub(super) physical: vk::PhysicalDevice,
    pub(super) queue: vk::Queue,
    pub(super) queue_family: u32,
    pub(super) name: String,
    pub(super) device_local: u32,
    pub(super) host_visible: u32,
    pub(super) readback: u32,
    pub(super) max_texture_dimension: u32,
}

impl Gpu {
    pub(super) fn create() -> Result<Self, String> {
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

/// A persistently mapped host buffer that only ever grows between submissions.
pub(super) struct HostBuffer {
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) pointer: *mut u8,
    pub(super) capacity: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_type: u32,
}

impl HostBuffer {
    pub(super) fn new(
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

    pub(super) fn ensure(&mut self, gpu: &Gpu, needed: vk::DeviceSize) -> Result<(), String> {
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
