//! Per-frame scene evaluation math behind the `spcore_*` C ABI (version 1).
//!
//! Python compiles the engraved scene once (engraving, layout solving and the
//! camera certificate stay in Python), serializes the resulting curves and
//! notes into this crate, and then evaluates every rendered frame with a
//! single FFI call: camera position, layout rows, zoom, activity envelopes
//! and the tile plan header. Compilation algorithms stay in `axis.py`,
//! `camera.py` and `layout.py`; independent test references verify the
//! native frame evaluation against the former Python render path.

mod abi;
mod axis;
mod camera;
mod ffi;
mod math;
mod scene;
mod track;

pub use abi::{CoreFrame, CoreRow, CurveKey, LayoutConsts, Note};
pub use axis::TimeAxis;
pub use camera::Camera;
pub use ffi::{spcore_abi_version, spcore_close, spcore_compile, spcore_frame, spcore_last_error};
pub use scene::CoreScene;
pub use track::Track;

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}
