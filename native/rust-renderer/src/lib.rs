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

mod abi;
mod ffi;
mod renderer;

pub use abi::{BatchItem, Frame, Quad};
pub use ffi::{
    sprhi_abi_version, sprhi_close, sprhi_create, sprhi_last_error, sprhi_release_frame,
    sprhi_remove, sprhi_report, sprhi_submit, sprhi_submit_batch_owned, sprhi_submit_owned,
    sprhi_upload,
};

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}
