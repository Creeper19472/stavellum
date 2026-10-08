use crate::{
    abi::{CoreFrame, CoreRow, LayoutConsts, Note},
    camera::Camera,
    math::ease,
    track::Track,
};

const TILE_PIXELS: f64 = 1024.0;
const TILE_BLEED: f64 = 2.0;
const ACTIVITY_ATTACK_SECONDS: f64 = 0.1;
const ACTIVITY_RELEASE_SECONDS: f64 = 0.12;

pub(crate) struct Part {
    tops: Track,
    opacities: Track,
    height: f64,
    staff_midpoint: f64,
    notes: Vec<Note>,
}

impl Part {
    pub(crate) fn new(
        tops: Track,
        opacities: Track,
        height: f64,
        staff_midpoint: f64,
        notes: Vec<Note>,
    ) -> Self {
        Self {
            tops,
            opacities,
            height,
            staff_midpoint,
            notes,
        }
    }
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
    pub(crate) fn new(camera: Camera, zoom: Track, parts: Vec<Part>, consts: LayoutConsts) -> Self {
        Self {
            camera,
            zoom,
            parts,
            consts,
        }
    }

    pub(crate) fn part_count(&self) -> usize {
        self.parts.len()
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{abi::CurveKey, axis::TimeAxis};

    fn key(time: f64, value: f64) -> CurveKey {
        CurveKey {
            time,
            value,
            velocity: 0.0,
            acceleration: 0.0,
        }
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
            camera: Camera::new(
                TimeAxis::new(vec![0.0, 1.0], vec![0.0, 1000.0]).unwrap(),
                2.0,
                0.0,
                0.0,
            ),
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
