use crate::axis::TimeAxis;

/// Absolute-time lookahead camera (exact port of the evaluation half of
/// `camera.py::CameraTimeline`).
pub struct Camera {
    axis: TimeAxis,
    beats_per_second: f64,
    score_offset: f64,
    half_window: f64,
}

impl Camera {
    pub(crate) fn new(
        axis: TimeAxis,
        beats_per_second: f64,
        score_offset: f64,
        half_window: f64,
    ) -> Self {
        Self {
            axis,
            beats_per_second,
            score_offset,
            half_window,
        }
    }

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
