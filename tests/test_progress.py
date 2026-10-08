"""Estimates follow completed work, resets and monotonic worker samples."""

import pickle
from dataclasses import replace

import pytest

from stavellum.domain.progress import ExportEstimator, ExportProgress, ProgressReporter


def frame(elapsed, completed, **kwargs):
    return ExportProgress("frames", "", elapsed, completed=completed, total=1000,
                          unit="frames", fps=60, **kwargs)


def test_progress_snapshots_are_picklable_and_reporter_bounds_updates():
    now = [10.0]
    received = []
    reporter = ProgressReporter(received.append, clock=lambda: now[0])
    reporter.emit("frames", completed=0, total=1000, unit="frames")
    now[0] += .1
    reporter.emit("frames", completed=5)
    assert len(received) == 1
    now[0] += .2
    reporter.emit("frames", completed=15)
    reporter.emit("finalize", "正在收尾")
    assert len(received) == 3
    assert received[1].completed == 15 and received[1].total == 1000
    assert received[1].elapsed_seconds == pytest.approx(.3)
    assert pickle.loads(pickle.dumps(received[-1])) == received[-1]


def test_video_estimate_warms_up_and_uses_last_five_seconds():
    estimator = ExportEstimator()
    assert estimator.update(frame(0, 0), 10).speed is None
    assert estimator.update(frame(.5, 50), 10.5).speed is None
    estimate = estimator.update(frame(1, 100), 11)
    assert estimate.speed == 100 and estimate.remaining_seconds == 9
    estimator.update(frame(5, 500), 15)
    estimate = estimator.update(frame(6, 550), 16)
    assert estimate.speed == 90
    assert estimate.remaining_seconds == 5
    assert estimator.estimate(21).speed is None


def test_retry_resets_estimate_but_worker_elapsed_continues():
    estimator = ExportEstimator()
    estimator.update(frame(1, 0), 1)
    estimator.update(frame(2, 100), 2)
    assert estimator.update(frame(10, 0, attempt=2), 10).speed is None
    estimate = estimator.update(frame(11, 50, attempt=2), 11)
    assert estimate.speed == 50 and estimate.remaining_seconds == 19
    assert estimator.update(ExportProgress("finalize", "收尾", 12), 12).speed is None


def test_worker_clock_prevents_batched_gui_delivery_from_infinite_speed():
    estimator = ExportEstimator()
    estimator.update(frame(0, 0), 50)
    estimate = estimator.update(frame(2, 100), 50)
    assert estimate.speed == 50
    assert estimate.remaining_seconds == 18


def test_same_timestamp_and_counter_regression_have_no_spurious_estimate():
    estimator = ExportEstimator()
    estimator.update(frame(0, 0), 0)
    assert estimator.update(frame(0, 5), 0).speed is None
    estimator.update(frame(2, 100), 2)
    assert estimator.update(frame(3, 0), 3).speed is None


def test_parts_estimate_starts_after_first_commit_and_tracks_page_activity():
    estimator = ExportEstimator("parts")
    initial = ExportProgress("xml", "", 3, unit="parts", total=3, part_name="Violin")
    assert estimator.update(initial, 3).speed is None
    estimate = estimator.update(replace(initial, phase="save", elapsed_seconds=13,
                                        completed=1), 13)
    assert estimate.speed == .1 and estimate.remaining_seconds == 20
    next_part = replace(initial, phase="pdf", elapsed_seconds=17, completed=1,
                        part_name="Piano", page=1, pages=10)
    assert estimator.update(next_part, 17).remaining_seconds is not None
    assert estimator.update(replace(next_part, elapsed_seconds=21, page=2), 21).speed > 0
    assert estimator.estimate(26).speed is None
    assert estimator.update(replace(next_part, phase="done", elapsed_seconds=30), 30).speed is None
