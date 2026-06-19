import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import epistemic_task as et  # noqa: E402
import gabor_video as gv  # noqa: E402


def test_display_frame_rows_repeat_event_labels_and_pulse_edges():
    task = et.Experiment1Config(n_trials=1, frames_per_trial=3, frame_rate_hz=2.0, seed=5)
    render = gv.RenderConfig(width=320, height=240, display_fps=10.0, event_hz=2.0,
                             gabor_size_px=100, photodiode_pulse_frames=2)
    log = et.generate_experiment(task)
    rows = gv._display_frame_rows(log, render)

    assert len(rows) == 15
    assert rows[0]["display_frame_id"] == 0
    assert rows[-1]["display_frame_id"] == 14
    assert [r["photodiode_pulse"] for r in rows[:5]] == [1, 1, 0, 0, 0]
    assert len({r["surprise_nats"] for r in rows[:5]}) == 1
    assert rows[5]["event_global_id"] == 1


def test_render_frame_shape_and_photodiode_levels():
    event = et.generate_experiment(
        et.Experiment1Config(n_trials=1, frames_per_trial=1, seed=1)
    ).events[0]
    render = gv.RenderConfig(width=320, height=240, gabor_size_px=100)
    on = gv.render_frame(event, render, pulse_on=True)
    off = gv.render_frame(event, render, pulse_on=False)

    assert on.shape == (240, 320, 3)
    assert int(on[0, 0, 0]) == render.photodiode_on_gray
    assert int(off[0, 0, 0]) == render.photodiode_off_gray
    assert on.min() < render.background_gray
    assert on.max() > render.background_gray


def test_write_display_frames(tmp_path):
    task = et.Experiment1Config(n_trials=1, frames_per_trial=2, frame_rate_hz=2.0, seed=5)
    render = gv.RenderConfig(width=320, height=240, display_fps=10.0, event_hz=2.0,
                             gabor_size_px=100)
    rows = gv._display_frame_rows(et.generate_experiment(task), render)
    out = tmp_path / "display_frames.csv"
    gv._write_display_frames(rows, str(out))

    with open(out, newline="", encoding="utf-8") as f:
        loaded = list(csv.DictReader(f))
    assert len(loaded) == 10
    assert loaded[0]["display_frame_id"] == "0"
    assert "expected_information_gain_nats" in loaded[0]
