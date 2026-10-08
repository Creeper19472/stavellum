use std::cell::RefCell;
use std::ffi::{c_char, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use crate::{
    abi::{CoreFrame, CoreRow, CurveKey, LayoutConsts, Note},
    axis::TimeAxis,
    camera::Camera,
    require,
    scene::{CoreScene, Part},
    track::Track,
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
                "Unexpected native core failure".to_owned()
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

fn track_from(
    keys: &[CurveKey],
    counts: &[i64],
    index: usize,
    offset: &mut usize,
) -> Result<Track, String> {
    let count = *counts.get(index).ok_or("Track count array is too short")? as usize;
    require(
        *offset + count <= keys.len(),
        "Track key array is shorter than declared",
    )?;
    let track = Track::new(keys[*offset..*offset + count].to_vec())?;
    *offset += count;
    Ok(track)
}

fn total(counts: &[i64]) -> usize {
    counts.iter().map(|count| (*count).max(0) as usize).sum()
}

/// # Safety
/// All pointers must reference arrays of the declared lengths for the
/// duration of the call. The returned handle is used from any thread but
/// must be closed exactly once.
#[no_mangle]
pub unsafe extern "C" fn spcore_compile(
    beats: *const f64,
    xs: *const f64,
    axis_count: i64,
    camera_bps: f64,
    camera_offset: f64,
    camera_window: f64,
    part_count: i64,
    zoom_keys: *const CurveKey,
    zoom_count: i64,
    tops_keys: *const CurveKey,
    tops_counts: *const i64,
    tops_total: i64,
    opacity_keys: *const CurveKey,
    opacity_counts: *const i64,
    opacity_total: i64,
    part_dimensions: *const f64,
    notes: *const Note,
    note_counts: *const i64,
    consts: *const LayoutConsts,
) -> *mut c_void {
    let mut result = ptr::null_mut();
    guarded(|| {
        require(axis_count >= 2, "时间轴需要至少两个对应的拍点和谱面位置。")?;
        require(part_count >= 1, "Layout requires at least one part")?;
        require(
            !beats.is_null()
                && !xs.is_null()
                && !zoom_keys.is_null()
                && !tops_keys.is_null()
                && !tops_counts.is_null()
                && !opacity_keys.is_null()
                && !opacity_counts.is_null()
                && !part_dimensions.is_null()
                && !note_counts.is_null()
                && !consts.is_null(),
            "Null core scene input",
        )?;
        require(
            camera_bps.is_finite()
                && camera_bps > 0.0
                && camera_offset.is_finite()
                && camera_window.is_finite()
                && camera_window >= 0.0,
            "Invalid core camera parameters",
        )?;
        let constants = &*consts;
        let finite_constants = [
            constants.indicator_right,
            constants.indicator_source_width,
            constants.indicator_source_height,
            constants.icon_source_size,
            constants.icon_source_gap,
            constants.region_top,
            constants.region_bottom,
            constants.expanded_top,
            constants.expanded_bottom,
            constants.expansion_duration,
            constants.tempo_padding,
            constants.scene_scale,
            constants.play_x,
            constants.body_left,
            constants.body_right,
            constants.cache_limit,
        ];
        require(
            finite_constants.iter().all(|value| value.is_finite())
                && (constants.expansion_start.is_nan() || constants.expansion_start.is_finite())
                && !constants.tempo_exit_time.is_nan(),
            "Core layout constants must be finite",
        )?;
        require(
            constants.scene_scale.is_finite()
                && constants.scene_scale > 0.0
                && constants.expansion_duration.is_finite()
                && constants.expansion_duration > 0.0,
            "Invalid core layout scale or duration",
        )?;
        let beats = std::slice::from_raw_parts(beats, axis_count as usize).to_vec();
        let xs = std::slice::from_raw_parts(xs, axis_count as usize).to_vec();
        let axis = TimeAxis::new(beats, xs)?;
        let camera = Camera::new(axis, camera_bps, camera_offset, camera_window);
        require(
            zoom_count >= 1,
            "Layout zoom track requires at least one key",
        )?;
        let zoom = Track::new(std::slice::from_raw_parts(zoom_keys, zoom_count as usize).to_vec())?;
        let part_count = part_count as usize;
        let tops_counts = std::slice::from_raw_parts(tops_counts, part_count);
        let opacity_counts = std::slice::from_raw_parts(opacity_counts, part_count);
        let note_counts = std::slice::from_raw_parts(note_counts, part_count);
        // The declared totals must cover exactly the per-part counts before
        // any raw slice is formed from them.
        require(
            total(tops_counts) == tops_total.max(0) as usize,
            "Tops key total does not match the per-part counts",
        )?;
        require(
            total(opacity_counts) == opacity_total.max(0) as usize,
            "Opacity key total does not match the per-part counts",
        )?;
        let dimensions = std::slice::from_raw_parts(part_dimensions, part_count * 2);
        require(
            dimensions.iter().all(|value| value.is_finite()),
            "Core part dimensions must be finite",
        )?;
        let tops_total = tops_total.max(0) as usize;
        let opacity_total = opacity_total.max(0) as usize;
        let all_tops = std::slice::from_raw_parts(tops_keys, tops_total);
        let all_opacities = std::slice::from_raw_parts(opacity_keys, opacity_total);
        let notes_total = total(note_counts);
        require(notes_total == 0 || !notes.is_null(), "Null core notes")?;
        let all_notes = if notes_total > 0 {
            std::slice::from_raw_parts(notes, notes_total)
        } else {
            &[]
        };
        require(
            all_notes
                .iter()
                .all(|note| note.start.is_finite() && note.end.is_finite()),
            "Core note times must be finite",
        )?;
        let mut tops_offset = 0usize;
        let mut opacity_offset = 0usize;
        let mut notes_offset = 0usize;
        let mut parts = Vec::with_capacity(part_count);
        for index in 0..part_count {
            let tops = track_from(all_tops, tops_counts, index, &mut tops_offset)?;
            let opacities = track_from(all_opacities, opacity_counts, index, &mut opacity_offset)?;
            let note_count = note_counts[index].max(0) as usize;
            let part_notes = all_notes[notes_offset..notes_offset + note_count].to_vec();
            notes_offset += note_count;
            parts.push(Part::new(
                tops,
                opacities,
                dimensions[index * 2],
                dimensions[index * 2 + 1],
                part_notes,
            ));
        }
        let scene = CoreScene::new(camera, zoom, parts, *consts);
        result = Box::into_raw(Box::new(scene)).cast();
        Ok(())
    });
    result
}

/// # Safety
/// `rows` must have space for at least as many entries as the scene has
/// parts; `out` must be valid for writes.
#[no_mangle]
pub unsafe extern "C" fn spcore_frame(
    handle: *mut c_void,
    presentation_time: f64,
    audio_time: f64,
    out: *mut CoreFrame,
    rows: *mut CoreRow,
    rows_capacity: i32,
) -> i32 {
    guarded(|| {
        require(!handle.is_null(), "Null core handle")?;
        require(!out.is_null() && !rows.is_null(), "Null core output")?;
        require(
            presentation_time.is_finite() && audio_time.is_finite(),
            "Core frame times must be finite",
        )?;
        let scene = &*handle.cast::<CoreScene>();
        require(
            rows_capacity >= scene.part_count() as i32,
            "Core row buffer is too small",
        )?;
        let row_slice = std::slice::from_raw_parts_mut(rows, scene.part_count());
        let frame = scene.frame(presentation_time, audio_time, row_slice);
        require(
            frame.scale.is_finite()
                && frame.scale > 0.0
                && frame.tile_raster_scale.is_finite()
                && frame.tile_raster_scale > 0.0
                && frame.world_x.is_finite()
                && frame.camera_speed.is_finite()
                && frame.tile_working_bytes.is_finite()
                && frame.tile_first > i32::MIN
                && frame.tile_last < i32::MAX,
            "Core frame geometry is outside the supported finite range",
        )?;
        require(
            row_slice.iter().all(|row| {
                row.top.is_finite()
                    && row.opacity.is_finite()
                    && row.indicator_x.is_finite()
                    && row.indicator_y.is_finite()
                    && row.indicator_w.is_finite()
                    && row.indicator_h.is_finite()
                    && row.bounds_top.is_finite()
                    && row.bounds_bottom.is_finite()
                    && row.icon_size.is_finite()
                    && row.activity_level.is_finite()
                    && row.activity_attack.is_finite()
            }),
            "Core frame rows must be finite",
        )?;
        *out = frame;
        Ok(())
    })
}

/// # Safety
/// `handle` must originate from `spcore_compile` and be closed exactly once.
#[no_mangle]
pub unsafe extern "C" fn spcore_close(handle: *mut c_void) -> i32 {
    guarded(|| {
        if !handle.is_null() {
            drop(Box::from_raw(handle.cast::<CoreScene>()));
        }
        Ok(())
    })
}

#[no_mangle]
pub extern "C" fn spcore_abi_version() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn spcore_last_error() -> *const c_char {
    ERROR.with(|slot| slot.borrow().as_ptr())
}
