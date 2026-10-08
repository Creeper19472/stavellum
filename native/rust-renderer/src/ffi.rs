use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{
    ptr, slice,
    sync::Mutex,
    thread::{self, ThreadId},
    time::Instant,
};

use crate::{
    abi::{BatchItem, Frame, Quad, MAX_BATCH, MAX_QUADS},
    renderer::{recycle_frame, Renderer},
    require,
};

thread_local! {
    static ERROR: RefCell<CString> = RefCell::new(CString::default());
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
                !pixels.is_null() && capacity >= renderer.frame_bytes(),
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
            renderer.record_total_seconds(timing.elapsed().as_secs_f64());
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
                renderer.thread_id() == thread::current().id(),
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
    fn null_quad_pointer_accepted_only_without_commands() {
        unsafe {
            assert!(quad_slice(ptr::null(), 0).unwrap().is_empty());
            assert!(quad_slice(ptr::null(), 1).is_err());
        }
    }
}
