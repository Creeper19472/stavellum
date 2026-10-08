use std::ffi::c_void;

pub(crate) const MAX_BATCH: usize = 8;
pub(crate) const MAX_QUADS: usize = 1_000_000;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_layout_matches_python() {
        assert_eq!(std::mem::size_of::<Quad>(), 56);
        assert_eq!(std::mem::size_of::<BatchItem>(), 16);
        assert_eq!(std::mem::size_of::<Frame>(), 24);
        assert_eq!(std::mem::align_of::<Quad>(), 8);
        assert_eq!(std::mem::offset_of!(Quad, texture_id), 0);
        // Python names the twelve f32 values x/y/w/h/u0/v0/u1/v1/r/g/b/a.
        // Rust stores that same payload as one contiguous array.
        assert_eq!(std::mem::offset_of!(Quad, values), 8);
        assert_eq!(std::mem::size_of::<[f32; 12]>(), 48);
        assert_eq!(std::mem::align_of::<BatchItem>(), 8);
        assert_eq!(std::mem::offset_of!(BatchItem, quads), 0);
        assert_eq!(std::mem::offset_of!(BatchItem, count), 8);
        assert_eq!(std::mem::align_of::<Frame>(), 8);
        assert_eq!(std::mem::offset_of!(Frame, owner), 0);
        assert_eq!(std::mem::offset_of!(Frame, pixels), 8);
        assert_eq!(std::mem::offset_of!(Frame, size), 16);
    }
}
