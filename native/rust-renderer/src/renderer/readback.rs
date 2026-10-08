use std::{ptr, slice, sync::Mutex, time::Instant};

use super::Renderer;
use crate::{abi::Frame, require};

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

pub(crate) fn recycle_frame(frame: Vec<u8>) {
    if let Ok(mut pool) = FRAME_POOL.lock() {
        if pool.bytes + frame.capacity() <= pool.limit && pool.spare.len() < 64 {
            pool.bytes += frame.capacity();
            pool.spare.push(frame);
        }
    }
}

impl Renderer {
    pub(crate) fn owned_outputs(&mut self, count: usize) -> Result<Vec<Frame>, String> {
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

    pub(crate) fn copy_output(&mut self, pixels: *mut u8, capacity: usize) -> Result<(), String> {
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

pub(super) fn swizzle_path(bgra: bool) -> &'static str {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
