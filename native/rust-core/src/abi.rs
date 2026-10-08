#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CurveKey {
    pub time: f64,
    pub value: f64,
    pub velocity: f64,
    pub acceleration: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Note {
    pub start: f64,
    pub end: f64,
    pub velocity: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LayoutConsts {
    pub indicator_right: f64,
    pub indicator_source_width: f64,
    pub indicator_source_height: f64,
    pub icon_source_size: f64,
    pub icon_source_gap: f64,
    pub region_top: f64,
    pub region_bottom: f64,
    pub expanded_top: f64,
    pub expanded_bottom: f64,
    /// NaN when no announcement expansion exists.
    pub expansion_start: f64,
    pub expansion_duration: f64,
    pub tempo_padding: f64,
    pub tempo_exit_time: f64,
    pub scene_scale: f64,
    pub play_x: f64,
    pub body_left: f64,
    pub body_right: f64,
    pub cache_limit: f64,
    /// Part index owning the tempo decoration, or -1. Kept last so the
    /// struct matches the ctypes declaration field for field.
    pub tempo_owner: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CoreRow {
    pub top: f64,
    pub opacity: f64,
    pub indicator_x: f64,
    pub indicator_y: f64,
    pub indicator_w: f64,
    pub indicator_h: f64,
    pub bounds_top: f64,
    pub bounds_bottom: f64,
    pub icon_size: f64,
    pub activity_level: f64,
    pub activity_attack: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CoreFrame {
    // All doubles first, then the packed integers, matching the ctypes
    // declaration field for field.
    pub world_x: f64,
    pub scale: f64,
    pub region_top: f64,
    pub region_bottom: f64,
    pub bounds_top: f64,
    pub bounds_bottom: f64,
    pub camera_speed: f64,
    pub tile_raster_scale: f64,
    pub tile_working_bytes: f64,
    pub tile_level: i32,
    pub tile_first: i32,
    pub tile_last: i32,
    pub part_count: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_layout_matches_python() {
        assert_eq!(std::mem::size_of::<CurveKey>(), 32);
        assert_eq!(std::mem::align_of::<CurveKey>(), 8);
        assert_eq!(std::mem::offset_of!(CurveKey, time), 0);
        assert_eq!(std::mem::offset_of!(CurveKey, value), 8);
        assert_eq!(std::mem::offset_of!(CurveKey, velocity), 16);
        assert_eq!(std::mem::offset_of!(CurveKey, acceleration), 24);
        assert_eq!(std::mem::size_of::<Note>(), 24);
        assert_eq!(std::mem::align_of::<Note>(), 8);
        assert_eq!(std::mem::offset_of!(Note, start), 0);
        assert_eq!(std::mem::offset_of!(Note, end), 8);
        assert_eq!(std::mem::offset_of!(Note, velocity), 16);
        assert_eq!(std::mem::size_of::<LayoutConsts>(), 152);
        assert_eq!(std::mem::align_of::<LayoutConsts>(), 8);
        assert_eq!(std::mem::offset_of!(LayoutConsts, indicator_right), 0);
        assert_eq!(
            std::mem::offset_of!(LayoutConsts, indicator_source_width),
            8
        );
        assert_eq!(
            std::mem::offset_of!(LayoutConsts, indicator_source_height),
            16
        );
        assert_eq!(std::mem::offset_of!(LayoutConsts, icon_source_size), 24);
        assert_eq!(std::mem::offset_of!(LayoutConsts, icon_source_gap), 32);
        assert_eq!(std::mem::offset_of!(LayoutConsts, region_top), 40);
        assert_eq!(std::mem::offset_of!(LayoutConsts, region_bottom), 48);
        assert_eq!(std::mem::offset_of!(LayoutConsts, expanded_top), 56);
        assert_eq!(std::mem::offset_of!(LayoutConsts, expanded_bottom), 64);
        assert_eq!(std::mem::offset_of!(LayoutConsts, expansion_start), 72);
        assert_eq!(std::mem::offset_of!(LayoutConsts, expansion_duration), 80);
        assert_eq!(std::mem::offset_of!(LayoutConsts, tempo_padding), 88);
        assert_eq!(std::mem::offset_of!(LayoutConsts, tempo_exit_time), 96);
        assert_eq!(std::mem::offset_of!(LayoutConsts, scene_scale), 104);
        assert_eq!(std::mem::offset_of!(LayoutConsts, play_x), 112);
        assert_eq!(std::mem::offset_of!(LayoutConsts, body_left), 120);
        assert_eq!(std::mem::offset_of!(LayoutConsts, body_right), 128);
        assert_eq!(std::mem::offset_of!(LayoutConsts, cache_limit), 136);
        assert_eq!(std::mem::offset_of!(LayoutConsts, tempo_owner), 144);
        assert_eq!(std::mem::size_of::<CoreRow>(), 88);
        assert_eq!(std::mem::align_of::<CoreRow>(), 8);
        assert_eq!(std::mem::offset_of!(CoreRow, top), 0);
        assert_eq!(std::mem::offset_of!(CoreRow, opacity), 8);
        assert_eq!(std::mem::offset_of!(CoreRow, indicator_x), 16);
        assert_eq!(std::mem::offset_of!(CoreRow, indicator_y), 24);
        assert_eq!(std::mem::offset_of!(CoreRow, indicator_w), 32);
        assert_eq!(std::mem::offset_of!(CoreRow, indicator_h), 40);
        assert_eq!(std::mem::offset_of!(CoreRow, bounds_top), 48);
        assert_eq!(std::mem::offset_of!(CoreRow, bounds_bottom), 56);
        assert_eq!(std::mem::offset_of!(CoreRow, icon_size), 64);
        assert_eq!(std::mem::offset_of!(CoreRow, activity_level), 72);
        assert_eq!(std::mem::offset_of!(CoreRow, activity_attack), 80);
        assert_eq!(std::mem::size_of::<CoreFrame>(), 88);
        assert_eq!(std::mem::align_of::<CoreFrame>(), 8);
        assert_eq!(std::mem::offset_of!(CoreFrame, world_x), 0);
        assert_eq!(std::mem::offset_of!(CoreFrame, scale), 8);
        assert_eq!(std::mem::offset_of!(CoreFrame, region_top), 16);
        assert_eq!(std::mem::offset_of!(CoreFrame, region_bottom), 24);
        assert_eq!(std::mem::offset_of!(CoreFrame, bounds_top), 32);
        assert_eq!(std::mem::offset_of!(CoreFrame, bounds_bottom), 40);
        assert_eq!(std::mem::offset_of!(CoreFrame, camera_speed), 48);
        assert_eq!(std::mem::offset_of!(CoreFrame, tile_raster_scale), 56);
        assert_eq!(std::mem::offset_of!(CoreFrame, tile_working_bytes), 64);
        assert_eq!(std::mem::offset_of!(CoreFrame, tile_level), 72);
        assert_eq!(std::mem::offset_of!(CoreFrame, tile_first), 76);
        assert_eq!(std::mem::offset_of!(CoreFrame, tile_last), 80);
        assert_eq!(std::mem::offset_of!(CoreFrame, part_count), 84);
    }
}
