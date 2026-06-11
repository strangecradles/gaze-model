"""G1 acceptance tests: shape / rate / finite invariants for every asset.

Derived series (along-shift, coarse-perp, frame-truth) run on small capped
slices so the suite stays fast; the loaders themselves enforce the full-shape
invariants via asserts.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402


# ---- (a) atlas ----

def test_atlas_shapes_and_finite():
    a = data.load_atlas()
    assert a.frames.ndim == 3
    assert a.frames.shape[1:] == (600, 1200)
    assert a.frames.shape[0] >= 20
    assert a.ref_map.shape == (600, 1200)
    assert a.phase.shape == (a.frames.shape[0], 1200)
    assert np.isfinite(a.ref_map).all()
    assert np.isfinite(a.frames).all()
    # perp = rows, along = cols
    assert a.rows_perp == 600 and a.cols_along == 1200


def test_atlas_bilinear_sample_in_range():
    a = data.load_atlas()
    v = a.at(300.5, 600.5)
    assert np.isfinite(v)
    lo, hi = a.ref_map.min(), a.ref_map.max()
    assert lo - 1e-3 <= v <= hi + 1e-3


# ---- (b) line-scan ----

def test_line_scan_meta_rate():
    for which in ("test2", "Athton1"):
        m = data.line_scan_meta(which)
        assert m["along_len"] == 1000
        assert 8000 < m["line_rate_hz"] < 16000
        assert m["n_total"] > 100000


def test_line_scan_load_slice():
    ls = data.load_line_scan("test2", max_sweeps=2000)
    assert ls.sweeps.shape == (2000, 1000)
    assert ls.sweeps.dtype == np.float32
    assert np.isfinite(ls.sweeps).all()
    assert 8000 < ls.line_rate_hz < 16000
    assert ls.capture_type == "xscan"


# ---- (c) along-shift ----

def test_along_shift_series():
    s = data.along_shift("test2", max_sweeps=3000)
    assert s.ndim == 1 and s.shape[0] == 3000
    assert np.isfinite(s).all()
    assert abs(float(s.mean())) < 1e-3  # mean-removed
    assert s.std() > 0  # the eye actually moves


# ---- (d) coarse-perp ----

def test_coarse_perp_series():
    a = data.load_atlas()
    c = data.coarse_perp("test2", atlas=a, block=32, max_sweeps=4000)
    assert c.ndim == 1 and c.shape[0] == 4000 // 32
    assert np.isfinite(c).all()
    # anchor lives within the atlas perp range
    assert (c >= 0).all() and (c <= 600).all()


# ---- (e) frame-truth ----

def test_frame_truth_raster():
    ft = data.frame_truth("test1", max_frames=40)
    assert ft.perp_px.shape == ft.along_px.shape == ft.t_s.shape
    assert ft.perp_px.shape[0] == 40
    assert np.isfinite(ft.perp_px).all() and np.isfinite(ft.along_px).all()
    assert 10 < ft.fps < 20
    # monotonic time
    assert np.all(np.diff(ft.t_s) > 0)


def test_frame_truth_rejects_xscan():
    with pytest.raises(ValueError):
        data.frame_truth("test2", max_frames=10)


# ---- (f) machine tracker ----

def test_machine_tracker_rate_and_shape():
    for which in ("test1", "test2", "Athton1"):
        t = data.load_tracker(which)
        assert t.right_x.shape == t.right_y.shape == t.t_s.shape
        assert 20 < t.rate_hz < 60
        assert np.all(np.diff(t.t_s) >= 0)


# ---- Units ----

def test_units_roundtrip():
    u = data.UNITS
    rows = np.array([0.0, 126.0, 252.0])
    arc = u.rows_to_arcmin(rows)
    back = u.arcmin_to_rows(arc)
    assert np.allclose(back, rows)
    assert abs(u.alias_spacing_rows - 126.0) < 1e-6


# ---- stimulus ----

def test_stimulus_degrees():
    s = data.load_stimulus("pursuit")
    assert s.x_deg.shape == s.x_px.shape
    assert np.isfinite(s.x_deg).all()
    assert np.abs(s.x_deg).max() < 30  # plausible visual angle
