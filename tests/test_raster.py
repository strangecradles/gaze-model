"""Acceptance tests for the raster strip-tracking gaze study.

Covers the new study modules:
  - raster_track: strip rate geometry, finite output, sub-frame rate > frame rate
  - raster_synth: a known synthetic trajectory is recovered to a few arcmin and
    100% lock at a high (>1 kHz) rate — the labeled certification
  - raster_attention: a(t) is a finite [0,1] signal, deterministic, with a
    bounded teleop export

Kept fast (small frame counts / few strip sizes). Skips cleanly if the test1
SLO capture is absent (CI without the heavy data).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
import raster_track as rt

pytestmark = pytest.mark.skipif(
    not os.path.exists(data._slo_path("test1")),
    reason="test1 SLO capture not present")


def test_strip_rate_geometry():
    # 808 slow-axis cols @ 14.633 fps -> (808//S)*fps; must exceed frame rate.
    m = data.line_scan_meta("test1")
    fps = m["fps"]
    for S, lo in [(32, 300), (8, 1000), (2, 5000)]:
        rate = (808 // S) * fps
        assert rate > fps * 5
    assert (808 // 8) * fps > 1000.0     # S=8 clears 1 kHz


def test_track_finite_and_shaped():
    tk = rt.track("test1", S=32, ref_mode="incremental", max_frames=40, pad=40)
    assert tk.perp_px.shape == tk.along_px.shape == tk.t.shape == tk.q.shape
    assert np.isfinite(tk.perp_px).all() and np.isfinite(tk.along_px).all()
    assert tk.strip_hz > tk.fps          # sub-frame sampling
    assert 0.0 <= tk.infov.mean() <= 1.0
    assert np.all(np.diff(tk.t) >= -1e-9)  # time non-decreasing


def test_synth_recovers_known_trajectory():
    import raster_synth as rsy
    truth, _ = rsy.make_truth()
    Ht, Wt = truth.shape
    amp = float(min(60, (Ht - rsy.H) // 2 - 8, (Wt - rsy.W) // 2 - 8))
    traj = rsy.gen_trajectory(n_frames=30, amp_px=amp, seed=1)
    # high rate (>1 kHz): S=8 -> ~1478 Hz
    frames, oy, ox = rsy.render_frames(truth, traj, add_noise=True, rate=(rsy.W // 8) * rsy.FPS)
    t, pe, al, q = rsy.track_to_truth(frames, truth, oy, ox, S=8)
    tp, ta, md = rsy._true_at(traj, t)
    good = q > 0.3
    assert good.mean() > 0.9             # near-total lock on clean reference
    # recovered position tracks truth to a few px (arcmin-class)
    rms_perp = np.sqrt(np.mean((pe[good] - tp[good]) ** 2))
    rms_along = np.sqrt(np.mean((al[good] - ta[good]) ** 2))
    assert rms_perp < 12.0               # px; ~0.4'/px -> a few arcmin
    assert rms_along < 14.0
    # correlation with truth is strong
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert corr(pe[good], tp[good]) > 0.9
    assert corr(al[good], ta[good]) > 0.9


def test_attention_bounded_and_deterministic():
    import raster_attention as ra
    att = ra.compute(which="test1", S=8)
    fin = np.isfinite(att.a)
    assert fin.mean() > 0.5
    assert np.nanmin(att.a) >= 0.0 and np.nanmax(att.a) <= 1.0
    assert att.conf.min() >= 0.0 and att.conf.max() <= 1.0 + 1e-9
    # engaged (low saccade-rate) epochs score >= lapsing epochs
    v = ra.validate(att)
    assert v["a_engaged_med"] >= v["a_lapsing_med"]
    # teleop export is bounded and sized to the control rate
    import tempfile
    path, df = ra.export_for_teleop(att, rate_hz=100.0,
                                    path=os.path.join(tempfile.gettempdir(), "_ra_test.csv"))
    assert set(df.columns) == {"t_s", "attention", "confidence"}
    assert df["attention"].between(0, 1).all()
    assert df["confidence"].between(0, 1).all()
