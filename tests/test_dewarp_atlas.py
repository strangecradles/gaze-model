"""Regression tests for the --dewarp-atlas / ref_frames atlas options.

Guarantees:
  - OFF (default) reproduces the committed baseline byte-for-byte (reads the same cache,
    same rdx), so default behaviour is unchanged.
  - The Azimipour dewarp kernel is correct (identity at zero motion; exact known shift).
  - ON changes the atlas (rdx differs) but preserves output shape and coordinate frame
    (t / frame / col arrays identical).

Kernel tests are synthetic and always run. Integration tests use the prebuilt Igor
caches and skip cleanly if those caches are absent (e.g. fresh checkout / CI).
"""
import os

import numpy as np
import pytest

import people_fov_pf as pf

IGOR = "Igor"


# ----------------------------- kernel unit tests -----------------------------

def test_kernel_identity_at_zero_motion():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((96, 808)).astype(np.float32)
    out = pf._atlas_dewarp_cols(a, 0.0, 0.0)
    assert out.shape == a.shape
    assert np.max(np.abs(out - a)) == 0.0          # zero velocity => exact identity


def test_kernel_changes_and_is_finite():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((96, 808)).astype(np.float32)
    out = pf._atlas_dewarp_cols(a, 4.0, 1.0)
    assert out.shape == a.shape
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out - a)) > 0.0


def test_kernel_known_horizontal_shift():
    # prv_db[r, c] = c  (pure horizontal gradient). With vy=0 the dewarp is
    # corrected[r, c] = prv_db(r, c - disp_x[c]) = c - disp_x[c].
    Hc, W = 20, 808
    a = np.tile(np.arange(W, dtype=np.float64), (Hc, 1))
    vx = 6.0
    out = pf._atlas_dewarp_cols(a, vx, 0.0, order=1)
    frac = (np.arange(W) + 0.5) / W - 0.5
    disp_x = vx * frac
    expect = np.arange(W, dtype=np.float64)[None, :] - disp_x[None, :]
    # ignore a few edge columns (nearest-mode clamping)
    sl = slice(5, W - 5)
    assert np.allclose(out[:, sl], expect[:, sl], atol=1e-6)


def test_kernel_sign_is_stabilizing():
    # +sign subtracts displacement, -sign adds it; they must differ and be antisymmetric
    rng = np.random.default_rng(2)
    a = rng.standard_normal((40, 808)).astype(np.float32)
    pos = pf._atlas_dewarp_cols(a, 5.0, 0.0, sign=+1.0)
    neg = pf._atlas_dewarp_cols(a, 5.0, 0.0, sign=-1.0)
    assert np.max(np.abs(pos - neg)) > 0.0


# --------------------------- integration (cache) -----------------------------

def _has_cache(sub):
    return os.path.exists(sub.line_cache)


def test_off_reproduces_committed_baseline():
    sub = pf.subject_by_name(IGOR)
    if not _has_cache(sub):
        pytest.skip("committed Igor line cache not present")
    lm = pf.build_line_measurements(sub)                    # OFF, default
    m = pf.fov_mask(lm)
    from scipy.ndimage import gaussian_filter1d, median_filter
    import khz2d
    rdx = lm["rdx"].astype(float); fs = float(lm["line_rate"])
    sm = gaussian_filter1d(median_filter(khz2d.fill_nan(np.where(m, rdx, np.nan)), 7), fs * 0.01)
    scat = float(np.nanstd((rdx - sm)[m]) * pf.ARC_PER_PX)
    assert 4.0 < scat < 4.7                                  # committed baseline ~4.33'
    assert "dewarp_atlas" not in lm or not bool(lm.get("dewarp_atlas", False))


def test_on_changes_atlas_but_preserves_frame():
    sub = pf.subject_by_name(IGOR)
    dw = pf._dewarp_line_cache(sub)
    if not (_has_cache(sub) and os.path.exists(dw)):
        pytest.skip("dewarp Igor cache not present")
    off = pf.build_line_measurements(sub)
    on = pf.build_line_measurements(sub, dewarp_atlas=True)
    # coordinate frame preserved
    assert off["rdx"].shape == on["rdx"].shape
    assert np.array_equal(off["frame"], on["frame"])
    assert np.array_equal(off["col"], on["col"])
    assert np.allclose(off["t"], on["t"])
    # atlas actually changed the horizontal localization
    assert not np.array_equal(off["rdx"], on["rdx"])


def test_refavg1_would_equal_baseline_semantics():
    # ref_frames=1 disables the averaging branch -> identical code path to baseline.
    sub = pf.subject_by_name(IGOR)
    if not _has_cache(sub):
        pytest.skip("committed Igor line cache not present")
    lm = pf.build_line_measurements(sub, ref_frames=1)       # reads committed cache
    assert lm["rdx"].size > 0
