//! Per-frame scene evaluation math behind the `spcore_*` C ABI (version 1).
//!
//! Python compiles the engraved scene once (engraving, layout solving and the
//! camera certificate stay in Python), serializes the resulting curves and
//! notes into this crate, and then evaluates every rendered frame with a
//! single FFI call: camera position, layout rows, zoom, activity envelopes
//! and the tile plan header. The port is a faithful translation of
//! `axis.py`, `camera.py`, `layout.py::sample/at`, `scene.py::
//! activity_levels` and `render.py::_tile_plan/_raster_level`; golden-value
//! tests in `tests/test_core_native.py` keep both implementations honest.

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

const TILE_PIXELS: f64 = 1024.0;
const TILE_BLEED: f64 = 2.0;
const ACTIVITY_ATTACK_SECONDS: f64 = 0.1;
const ACTIVITY_RELEASE_SECONDS: f64 = 0.12;

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

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

fn ease(value: f64) -> f64 {
    // Clamped quintic with zero endpoint velocity and acceleration.
    let value = value.clamp(0.0, 1.0);
    value * value * value * (10.0 + value * (-15.0 + 6.0 * value))
}

/// Quintic Hermite coefficients in the segment's normalized 0..1 coordinate.
fn hermite(first: CurveKey, last: CurveKey) -> [f64; 6] {
    let duration = last.time - first.time;
    let first_velocity = first.velocity * duration;
    let first_acceleration = first.acceleration * duration * duration / 2.0;
    let displacement = last.value - first.value - first_velocity - first_acceleration;
    let velocity = last.velocity * duration - first_velocity - 2.0 * first_acceleration;
    let acceleration = last.acceleration * duration * duration - 2.0 * first_acceleration;
    [
        first.value,
        first_velocity,
        first_acceleration,
        10.0 * displacement - 4.0 * velocity + acceleration / 2.0,
        -15.0 * displacement + 7.0 * velocity - acceleration,
        6.0 * displacement - 3.0 * velocity + acceleration / 2.0,
    ]
}

fn polynomial(coefficients: &[f64; 6], value: f64) -> f64 {
    // Horner over indices 0..count, matching Python's tuple evaluation.
    let mut result = 0.0;
    for coefficient in coefficients.iter().rev() {
        result = result * value + coefficient;
    }
    result
}

fn derivative(coefficients: &[f64; 6]) -> [f64; 6] {
    let mut result = [0.0; 6];
    for index in 1..coefficients.len() {
        result[index - 1] = index as f64 * coefficients[index];
    }
    result
}

/// Compiled curve track: quintic segments keyed by time.
#[derive(Clone)]
pub struct Track {
    keys: Vec<CurveKey>,
}

impl Track {
    fn new(keys: Vec<CurveKey>) -> Result<Self, String> {
        require(!keys.is_empty(), "Layout track requires at least one key")?;
        for pair in keys.windows(2) {
            require(
                pair[1].time > pair[0].time,
                "Layout track keys must be sorted by time",
            )?;
        }
        if !keys
            .iter()
            .all(|key| key.time.is_finite() && key.value.is_finite())
        {
            return Err("Layout track keys must be finite".to_owned());
        }
        Ok(Self { keys })
    }

    fn sample(&self, time: f64) -> CurveKey {
        let index = match self.keys.binary_search_by(|key| {
            key.time
                .partial_cmp(&time)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(index) => index,
            Err(index) => index.saturating_sub(1),
        };
        let first = self.keys[index];
        let Some(last) = self.keys.get(index + 1) else {
            return CurveKey {
                time,
                value: first.value,
                velocity: 0.0,
                acceleration: 0.0,
            };
        };
        if time < first.time || last.time <= first.time {
            return CurveKey {
                time,
                value: first.value,
                velocity: 0.0,
                acceleration: 0.0,
            };
        }
        let duration = last.time - first.time;
        let fraction = ((time - first.time) / duration).clamp(0.0, 1.0);
        let coefficients = hermite(first, *last);
        let first_derivative = derivative(&coefficients);
        let second_derivative = derivative(&first_derivative);
        CurveKey {
            time,
            value: polynomial(&coefficients, fraction),
            velocity: polynomial(&first_derivative, fraction) / duration,
            acceleration: polynomial(&second_derivative, fraction) / (duration * duration),
        }
    }
}

/// Monotone PCHIP-style score axis (exact port of `axis.py::TimeAxis`).
pub struct TimeAxis {
    beats: Vec<f64>,
    xs: Vec<f64>,
    derivatives: Vec<f64>,
}

impl TimeAxis {
    pub fn new(beats: Vec<f64>, xs: Vec<f64>) -> Result<Self, String> {
        require(
            beats.len() == xs.len() && beats.len() >= 2,
            "时间轴需要至少两个对应的拍点和谱面位置。",
        )?;
        for values in [&beats, &xs] {
            if !values.iter().all(|value| value.is_finite()) {
                return Err("时间轴锚点必须为有限数值。".to_owned());
            }
            for pair in values.windows(2) {
                if pair[1] <= pair[0] {
                    return Err("时间轴锚点必须严格递增。".to_owned());
                }
            }
        }
        let mut intervals = Vec::with_capacity(beats.len() - 1);
        let mut slopes = Vec::with_capacity(beats.len() - 1);
        for index in 0..beats.len() - 1 {
            let interval = beats[index + 1] - beats[index];
            let slope = (xs[index + 1] - xs[index]) / interval;
            if !interval.is_finite() || !slope.is_finite() || interval <= 0.0 || slope <= 0.0 {
                return Err("时间轴锚点间距和速度必须为有限正数。".to_owned());
            }
            intervals.push(interval);
            slopes.push(slope);
        }
        let mut derivatives = vec![slopes[0]];
        for index in 1..beats.len() - 1 {
            let (before, after) = (intervals[index - 1], intervals[index]);
            let (previous, following) = (slopes[index - 1], slopes[index]);
            let largest_interval = before.max(after);
            let before = before / largest_interval;
            let after = after / largest_interval;
            let first_weight = 2.0 * after + before;
            let second_weight = after + 2.0 * before;
            let smallest_slope = previous.min(following);
            let value = smallest_slope
                * ((first_weight + second_weight)
                    / (first_weight * (smallest_slope / previous)
                        + second_weight * (smallest_slope / following)));
            // A stricter PCHIP limit prevents near-stops within long beats.
            derivatives.push(value.min(2.0 * previous.min(following)));
        }
        derivatives.push(*slopes.last().expect("non-empty"));
        Ok(Self {
            beats,
            xs,
            derivatives,
        })
    }

    fn segment(&self, index: usize) -> (f64, f64, f64, f64, f64) {
        let interval = self.beats[index + 1] - self.beats[index];
        (
            interval,
            self.xs[index],
            self.xs[index + 1],
            self.derivatives[index],
            self.derivatives[index + 1],
        )
    }

    fn segment_x(&self, index: usize, fraction: f64) -> f64 {
        let (interval, start, end, first, last) = self.segment(index);
        let square = fraction * fraction;
        let cube = square * fraction;
        start
            + (-2.0 * cube + 3.0 * square) * (end - start)
            + (cube - 2.0 * square + fraction) * interval * first
            + (cube - square) * interval * last
    }

    /// Cubic coefficients in the segment's normalized coordinate.
    fn coefficients(&self, index: usize) -> (f64, f64, f64, f64) {
        let (interval, start, end, first, last) = self.segment(index);
        let first = first * interval;
        let last = last * interval;
        let distance = end - start;
        (
            start,
            first,
            3.0 * distance - 2.0 * first - last,
            -2.0 * distance + first + last,
        )
    }

    fn search(&self, beat: f64) -> usize {
        match self.beats.binary_search_by(|value| {
            value
                .partial_cmp(&beat)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(index) => index.min(self.beats.len() - 2),
            Err(index) => index.saturating_sub(1).min(self.beats.len() - 2),
        }
    }

    pub fn x_at(&self, beat: f64) -> f64 {
        if beat <= self.beats[0] {
            return self.xs[0] + (beat - self.beats[0]) * self.derivatives[0];
        }
        if beat >= *self.beats.last().expect("non-empty") {
            let last = self.beats.len() - 1;
            return self.xs[last] + (beat - self.beats[last]) * self.derivatives[last];
        }
        let index = self.search(beat);
        let fraction = (beat - self.beats[index]) / (self.beats[index + 1] - self.beats[index]);
        self.segment_x(index, fraction)
    }

    /// Local Taylor polynomial in beat units, including linear extrapolation.
    pub fn polynomial_at(&self, beat: f64) -> (f64, f64, f64, f64) {
        if beat <= self.beats[0] {
            return (self.x_at(beat), self.derivatives[0], 0.0, 0.0);
        }
        if beat >= *self.beats.last().expect("non-empty") {
            let last = self.beats.len() - 1;
            return (self.x_at(beat), self.derivatives[last], 0.0, 0.0);
        }
        let index = self.search(beat);
        let interval = self.beats[index + 1] - self.beats[index];
        let fraction = (beat - self.beats[index]) / interval;
        let (_, first, second, third) = self.coefficients(index);
        (
            self.x_at(beat),
            (first + fraction * (2.0 * second + 3.0 * third * fraction)) / interval,
            (second + 3.0 * third * fraction) / (interval * interval),
            third / (interval * interval * interval),
        )
    }

    pub fn speed_at(&self, beat: f64) -> f64 {
        self.polynomial_at(beat).1
    }

    pub fn max_abs_acceleration(&self) -> f64 {
        let mut maximum = 0.0f64;
        for index in 0..self.beats.len() - 1 {
            let interval = self.beats[index + 1] - self.beats[index];
            let interval_squared = interval * interval;
            let (_, _, second, third) = self.coefficients(index);
            maximum = maximum
                .max((2.0 * second / interval_squared).abs())
                .max(((2.0 * second + 6.0 * third) / interval_squared).abs());
        }
        maximum
    }

    /// Integrate locally rather than subtracting large cumulative primitives.
    fn integrate(&self, mut start: f64, end: f64, difference: bool) -> f64 {
        if end < start {
            return -self.integrate(end, start, difference);
        }
        let mut pieces: Vec<f64> = Vec::new();
        while start < end {
            let (right, value) = if start < self.beats[0] {
                let right = end.min(self.beats[0]);
                let middle = (start + right) / 2.0;
                let value = if difference {
                    self.derivatives[0]
                } else {
                    self.x_at(middle)
                };
                (right, value)
            } else if start >= *self.beats.last().expect("non-empty") {
                let middle = (start + end) / 2.0;
                let value = if difference {
                    *self.derivatives.last().expect("non-empty")
                } else {
                    self.x_at(middle)
                };
                (end, value)
            } else {
                let index = self.search(start);
                let interval = self.beats[index + 1] - self.beats[index];
                let right = end.min(self.beats[index + 1]);
                let midpoint =
                    ((start - self.beats[index]) + (right - self.beats[index])) / (2.0 * interval);
                let radius = (right - start) / (2.0 * interval);
                let (first, second, third, fourth) = self.coefficients(index);
                let value = if difference {
                    (second
                        + 2.0 * third * midpoint
                        + 3.0 * fourth * (midpoint * midpoint + radius * radius / 3.0))
                        / interval
                } else {
                    first
                        + second * midpoint
                        + third * (midpoint * midpoint + radius * radius / 3.0)
                        + fourth * (midpoint * midpoint * midpoint + midpoint * radius * radius)
                };
                (right, value)
            };
            pieces.push((right - start) * value);
            start = right;
        }
        // Python accumulates with math.fsum; plain ordered summation stays
        // within a few ulps for the handful of segments involved.
        pieces.iter().sum()
    }

    pub fn integral(&self, start: f64, end: f64) -> f64 {
        self.integrate(start, end, false)
    }

    pub fn difference(&self, start: f64, end: f64) -> f64 {
        self.integrate(start, end, true)
    }

    pub fn beat_at(&self, x: f64) -> f64 {
        if x <= self.xs[0] {
            return self.beats[0] + (x - self.xs[0]) / self.derivatives[0];
        }
        let last = self.beats.len() - 1;
        if x >= self.xs[last] {
            return self.beats[last] + (x - self.xs[last]) / self.derivatives[last];
        }
        let index = match self
            .xs
            .binary_search_by(|value| value.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(index) => return self.beats[index],
            Err(index) => index.saturating_sub(1),
        };
        let (mut low, mut high) = (0.0f64, 1.0f64);
        for _ in 0..48 {
            let middle = (low + high) / 2.0;
            if self.segment_x(index, middle) < x {
                low = middle;
            } else {
                high = middle;
            }
        }
        self.beats[index] + (low + high) / 2.0 * (self.beats[index + 1] - self.beats[index])
    }
}

/// Absolute-time lookahead camera (exact port of the evaluation half of
/// `camera.py::CameraTimeline`).
pub struct Camera {
    axis: TimeAxis,
    beats_per_second: f64,
    score_offset: f64,
    half_window: f64,
}

impl Camera {
    fn beat(&self, audio_time: f64) -> f64 {
        (audio_time - self.score_offset) * self.beats_per_second
    }

    fn radius(&self) -> f64 {
        self.half_window * self.beats_per_second
    }

    pub fn x_at(&self, audio_time: f64) -> f64 {
        let beat = self.beat(audio_time);
        let radius = self.radius();
        if radius == 0.0 {
            return self.axis.x_at(beat);
        }
        let (left, right) = (beat - radius, beat + radius);
        if right == left {
            return self.axis.x_at(beat);
        }
        self.axis.integral(left, right) / (right - left)
    }

    pub fn speed_at(&self, audio_time: f64) -> f64 {
        let beat = self.beat(audio_time);
        let radius = self.radius();
        if radius == 0.0 {
            return self.axis.speed_at(beat) * self.beats_per_second;
        }
        let (left, right) = (beat - radius, beat + radius);
        if right == left {
            return self.axis.speed_at(beat) * self.beats_per_second;
        }
        self.axis.difference(left, right) / (right - left) * self.beats_per_second
    }

    pub fn acceleration_at(&self, audio_time: f64) -> f64 {
        let beat = self.beat(audio_time);
        let radius = self.radius();
        let (_, _, square, _) = if radius == 0.0 || beat - radius == beat + radius {
            self.axis.polynomial_at(beat)
        } else {
            let (left, right) = (beat - radius, beat + radius);
            return (self.axis.speed_at(right) - self.axis.speed_at(left)) / (right - left)
                * self.beats_per_second
                * self.beats_per_second;
        };
        2.0 * square * self.beats_per_second * self.beats_per_second
    }
}

struct Part {
    tops: Track,
    opacities: Track,
    height: f64,
    staff_midpoint: f64,
    notes: Vec<Note>,
}

/// One compiled scene: everything needed to evaluate a frame.
pub struct CoreScene {
    camera: Camera,
    zoom: Track,
    parts: Vec<Part>,
    consts: LayoutConsts,
}

fn activity_levels(notes: &[Note], time: f64) -> (f64, f64) {
    // Envelopes indicate MIDI events, not measured audio loudness.
    let mut level = 0.0f64;
    let mut attack = 0.0f64;
    for note in notes {
        if time < note.start || note.velocity <= 0 {
            continue;
        }
        let velocity = note.velocity as f64 / 127.0;
        let note_level = if time < note.end {
            velocity
        } else if time < note.end + ACTIVITY_RELEASE_SECONDS {
            velocity * (1.0 - ease((time - note.end) / ACTIVITY_RELEASE_SECONDS))
        } else {
            continue;
        };
        level = level.max(note_level);
        let age = time - note.start;
        if age < ACTIVITY_ATTACK_SECONDS {
            attack = attack.max(note_level * (1.0 - ease(age / ACTIVITY_ATTACK_SECONDS)));
        }
    }
    (level, attack)
}

impl CoreScene {
    fn region_at(&self, time: f64) -> (f64, f64) {
        let fraction = if self.consts.expansion_start.is_nan() {
            0.0
        } else {
            ease((time - self.consts.expansion_start) / self.consts.expansion_duration)
        };
        (
            self.consts.region_top + (self.consts.expanded_top - self.consts.region_top) * fraction,
            self.consts.region_bottom
                + (self.consts.expanded_bottom - self.consts.region_bottom) * fraction,
        )
    }

    fn padding_at(&self, part_index: usize, time: f64) -> f64 {
        if self.consts.tempo_owner == part_index as i32 && time < self.consts.tempo_exit_time {
            self.consts.tempo_padding
        } else {
            0.0
        }
    }

    /// The cheapest power-of-two raster that never needs upsampling.
    ///
    /// A degenerate zoom (exp underflow) would otherwise underflow the scale
    /// to zero and loop forever; the Python path raises instead. Returning
    /// level zero keeps native evaluation total for any finite input.
    fn raster_level(&self, display_scale: f64) -> i32 {
        let ratio = self.consts.scene_scale / display_scale;
        if !(display_scale.is_finite() && display_scale > 0.0 && ratio.is_finite() && ratio > 0.0) {
            return 0;
        }
        let mut level = (ratio.log2().floor().max(0.0) as i32).min(1023);
        // Guard roundoff at exact layer boundaries without rounding up into a
        // raster smaller than the requested display resolution.
        while level != 0 && self.consts.scene_scale / 2f64.powi(level) < display_scale {
            level -= 1;
        }
        while level < 1023 && self.consts.scene_scale / 2f64.powi(level + 1) >= display_scale {
            level += 1;
        }
        level
    }

    /// Evaluate one frame; rows are written in part order.
    pub fn frame(
        &self,
        presentation_time: f64,
        audio_time: f64,
        rows: &mut [CoreRow],
    ) -> CoreFrame {
        let scale = self.zoom.sample(presentation_time).value.exp();
        let (region_top, region_bottom) = self.region_at(presentation_time);
        let world_x = self.camera.x_at(audio_time);
        let mut visible_top = f64::INFINITY;
        let mut visible_bottom = f64::NEG_INFINITY;
        for (index, part) in self.parts.iter().enumerate() {
            let top = part.tops.sample(presentation_time).value;
            let alpha = part
                .opacities
                .sample(presentation_time)
                .value
                .clamp(0.0, 1.0);
            let indicator_height = self.consts.indicator_source_height * scale;
            let indicator_width = self.consts.indicator_source_width * scale;
            let indicator_top = top + part.staff_midpoint * scale - indicator_height / 2.0;
            let bounds_top =
                (top - self.padding_at(index, presentation_time) * scale).min(indicator_top);
            let bounds_bottom = (top + part.height * scale).max(indicator_top + indicator_height);
            let (level, attack) = activity_levels(&part.notes, audio_time);
            rows[index] = CoreRow {
                top,
                opacity: alpha,
                indicator_x: self.consts.indicator_right - indicator_width,
                indicator_y: indicator_top,
                indicator_w: indicator_width,
                indicator_h: indicator_height,
                bounds_top,
                bounds_bottom,
                icon_size: self.consts.icon_source_size * scale,
                activity_level: level,
                activity_attack: attack,
            };
            if alpha > 1e-9 {
                visible_top = visible_top.min(bounds_top);
                visible_bottom = visible_bottom.max(bounds_bottom);
            }
        }
        let center = (region_top + region_bottom) / 2.0;
        let (bounds_top, bounds_bottom) = if visible_top.is_finite() {
            (visible_top, visible_bottom)
        } else {
            (center, center)
        };
        let tile_level = self.raster_level(scale);
        let raster_scale = self.consts.scene_scale / 2f64.powi(tile_level);
        let left = world_x - (self.consts.play_x - self.consts.body_left) / scale;
        let right = world_x + (self.consts.body_right - self.consts.play_x) / scale;
        let tile_first = (left * raster_scale / TILE_PIXELS).floor() as i32;
        let tile_last = (right * raster_scale / TILE_PIXELS).floor() as i32;
        let mut working_bytes = 0.0f64;
        for (index, part) in self.parts.iter().enumerate() {
            if rows[index].opacity <= 1e-6 {
                continue;
            }
            let bytes = (TILE_PIXELS + 2.0 * TILE_BLEED)
                * (part.height * raster_scale).ceil().max(1.0)
                * 4.0;
            working_bytes += bytes * ((tile_last - tile_first + 1) as f64);
        }
        // camera_speed is the raw CameraTimeline.speed_at(audio_time):
        // unlike CompiledScene.camera_speed_at it does not zero the intro.
        CoreFrame {
            world_x,
            scale,
            region_top,
            region_bottom,
            bounds_top,
            bounds_bottom,
            camera_speed: self.camera.speed_at(audio_time),
            tile_level,
            tile_raster_scale: raster_scale,
            tile_first,
            tile_last,
            tile_working_bytes: working_bytes,
            part_count: self.parts.len() as i32,
        }
    }
}

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
        let beats = std::slice::from_raw_parts(beats, axis_count as usize).to_vec();
        let xs = std::slice::from_raw_parts(xs, axis_count as usize).to_vec();
        let axis = TimeAxis::new(beats, xs)?;
        let camera = Camera {
            axis,
            beats_per_second: camera_bps,
            score_offset: camera_offset,
            half_window: camera_window,
        };
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
        let tops_total = tops_total.max(0) as usize;
        let opacity_total = opacity_total.max(0) as usize;
        let all_tops = std::slice::from_raw_parts(tops_keys, tops_total);
        let all_opacities = std::slice::from_raw_parts(opacity_keys, opacity_total);
        let notes_total = total(note_counts);
        let all_notes = if notes_total > 0 {
            std::slice::from_raw_parts(notes, notes_total)
        } else {
            &[]
        };
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
            parts.push(Part {
                tops,
                opacities,
                height: dimensions[index * 2],
                staff_midpoint: dimensions[index * 2 + 1],
                notes: part_notes,
            });
        }
        let scene = CoreScene {
            camera,
            zoom,
            parts,
            consts: *consts,
        };
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
        let scene = &*handle.cast::<CoreScene>();
        require(
            rows_capacity >= scene.parts.len() as i32,
            "Core row buffer is too small",
        )?;
        let row_slice = std::slice::from_raw_parts_mut(rows, scene.parts.len());
        *out = scene.frame(presentation_time, audio_time, row_slice);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn key(time: f64, value: f64) -> CurveKey {
        CurveKey {
            time,
            value,
            velocity: 0.0,
            acceleration: 0.0,
        }
    }

    #[test]
    fn axis_matches_linear_and_cubic_expectations() {
        let axis = TimeAxis::new(vec![0.0, 1.0, 2.0], vec![0.0, 100.0, 260.0]).unwrap();
        assert!((axis.x_at(0.0) - 0.0).abs() < 1e-12);
        assert!((axis.x_at(1.0) - 100.0).abs() < 1e-12);
        assert!((axis.x_at(2.0) - 260.0).abs() < 1e-12);
        assert!((axis.x_at(-1.0) + axis.derivatives[0]).abs() < 1e-12);
        assert!(axis.x_at(3.0) > 260.0);
        let beat = axis.beat_at(axis.x_at(0.5));
        assert!((beat - 0.5).abs() < 1e-9, "{beat}");
    }

    #[test]
    fn track_interpolates_with_zero_endpoint_derivatives() {
        let track = Track::new(vec![key(0.0, 0.0), key(1.0, 1.0)]).unwrap();
        assert!((track.sample(0.0).value - 0.0).abs() < 1e-12);
        assert!((track.sample(1.0).value - 1.0).abs() < 1e-12);
        let middle = track.sample(0.5).value;
        assert!((middle - 0.5).abs() < 1e-3, "smoothstep midpoint {middle}");
        assert!(track.sample(2.0).value > 0.999);
    }

    #[test]
    fn activity_envelope_attacks_and_releases() {
        let notes = vec![Note {
            start: 1.0,
            end: 2.0,
            velocity: 127,
        }];
        let (level, _) = activity_levels(&notes, 1.5);
        assert!((level - 1.0).abs() < 1e-12);
        let (level, attack) = activity_levels(&notes, 1.05);
        assert!(level > 0.99 && attack > 0.1);
        let (level, attack) = activity_levels(&notes, 2.1);
        assert!(level < 1.0 && attack == 0.0);
        let (level, _) = activity_levels(&notes, 2.2);
        assert_eq!(level, 0.0);
    }

    #[test]
    fn raster_level_never_upsamples() {
        let scene = test_scene(0.1);
        assert_eq!(scene.raster_level(0.1), 0);
        assert_eq!(scene.raster_level(0.05), 1);
        assert_eq!(scene.raster_level(0.025), 2);
        // Just past a boundary the guard deliberately does not round up
        // into a raster finer than the display resolution.
        assert_eq!(scene.raster_level(0.05 + 1e-12), 0);
        assert_eq!(scene.raster_level(0.075), 0);
    }

    fn test_scene(scene_scale: f64) -> CoreScene {
        CoreScene {
            camera: Camera {
                axis: TimeAxis::new(vec![0.0, 1.0], vec![0.0, 1000.0]).unwrap(),
                beats_per_second: 2.0,
                score_offset: 0.0,
                half_window: 0.0,
            },
            zoom: Track::new(vec![key(0.0, 0.05f64.ln())]).unwrap(),
            parts: vec![Part {
                tops: Track::new(vec![key(0.0, 100.0)]).unwrap(),
                opacities: Track::new(vec![key(0.0, 1.0)]).unwrap(),
                height: 3600.0,
                staff_midpoint: 1800.0,
                notes: Vec::new(),
            }],
            consts: LayoutConsts {
                indicator_right: 300.0,
                indicator_source_width: 195.0,
                indicator_source_height: 720.0,
                icon_source_size: 450.0,
                icon_source_gap: 450.0,
                region_top: 100.0,
                region_bottom: 900.0,
                expanded_top: 50.0,
                expanded_bottom: 950.0,
                expansion_start: f64::NAN,
                expansion_duration: 1.0,
                tempo_padding: 495.0,
                tempo_exit_time: 5.0,
                tempo_owner: 0,
                scene_scale,
                play_x: 400.0,
                body_left: 330.0,
                body_right: 1800.0,
                cache_limit: 1e9,
            },
        }
    }

    #[test]
    fn frame_packs_rows_and_tile_plan() {
        let scene = test_scene(0.1);
        let mut rows = [CoreRow {
            top: 0.0,
            opacity: 0.0,
            indicator_x: 0.0,
            indicator_y: 0.0,
            indicator_w: 0.0,
            indicator_h: 0.0,
            bounds_top: 0.0,
            bounds_bottom: 0.0,
            icon_size: 0.0,
            activity_level: 0.0,
            activity_attack: 0.0,
        }];
        let frame = scene.frame(0.5, 1.0, &mut rows);
        assert!((frame.scale - 0.05).abs() < 1e-12);
        assert!((rows[0].top - 100.0).abs() < 1e-12);
        assert_eq!(frame.part_count, 1);
        assert!(frame.tile_first <= frame.tile_last);
        assert!(frame.tile_working_bytes > 0.0);
        assert!(frame.tile_raster_scale > 0.0);
        // The tempo owner reserves padding above its row.
        assert!(rows[0].bounds_top < rows[0].top);
    }
}
