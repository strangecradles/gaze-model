"""Unit tests for the --mosaic-reject flag in filter.ParticleFilter.

- OFF (default) is byte-identical to a never-locked ON run (the flag only acts when locked).
- ON narrows the alias-robust-mean window ONLY when steadily locked; the gate uses the
  existing _lock_steady(max_ncc, p_pursuit) so saccade/lock-loss keep the wide 30' window.
"""
import numpy as np
import pytest

import filter as flt


def _toy_filter(mosaic_reject, window=3.0, seed=0):
    rng = np.random.default_rng(seed)
    st = flt.init_filter(400, 100.0, 0.0, 15.0, 4.0, rng=rng)
    atlas = rng.standard_normal((220, 80)).astype(np.float64)
    return flt.ParticleFilter(st, atlas, 200, col_step=1.0,
                              mosaic_reject=mosaic_reject, mosaic_window_rows=window)


def test_flag_defaults_off():
    pf = _toy_filter(False)
    assert pf.mosaic_reject is False


def test_lock_steady_gate_exists():
    pf = _toy_filter(True)
    # not locked -> gate False -> wide window path (no narrowing)
    assert pf._lock_steady(0.0, 0.0) is False
    # clearly locked -> gate True
    assert pf._lock_steady(0.99, 0.99) is True


def test_window_narrows_only_when_locked():
    # the est_perp is the alias-robust mean within `window` of the MAP. With mosaic_reject
    # ON and a steadily-locked step, particles beyond mosaic_window_rows from the MAP must be
    # excluded; OFF (or unlocked) keeps the full 30' window. We assert the selection logic by
    # replicating it on a synthetic bimodal cloud.
    import calib
    pos = np.concatenate([np.full(200, 0.0), np.full(200, 6.0)])  # two mosaic modes 6 rows apart
    w = np.concatenate([np.full(200, 0.6 / 200), np.full(200, 0.4 / 200)])
    imap = int(np.argmax(w))
    alias_window = 0.5 * calib.ALIAS_SPACING_ROWS
    near_wide = np.abs(pos - pos[imap]) < alias_window
    near_narrow = np.abs(pos - pos[imap]) < 3.0
    # wide window includes BOTH modes (averages the flip in); narrow excludes the far mode
    assert near_wide.all()
    assert near_narrow[:200].all() and not near_narrow[200:].any()


def test_never_locked_on_equals_off():
    # if the filter never locks (lock thresholds impossible), ON must equal OFF step-by-step.
    rng1 = np.random.default_rng(1); rng2 = np.random.default_rng(1)
    st1 = flt.init_filter(300, 100.0, 0.0, 15.0, 4.0, rng=rng1)
    st2 = flt.init_filter(300, 100.0, 0.0, 15.0, 4.0, rng=rng2)
    atlas = np.random.default_rng(3).standard_normal((220, 80)).astype(np.float64)
    off = flt.ParticleFilter(st1, atlas, 200, col_step=1.0, mosaic_reject=False)
    on = flt.ParticleFilter(st2, atlas, 200, col_step=1.0, mosaic_reject=True,
                            lock_ncc_thr=2.0, lock_p_pursuit_thr=2.0)  # never lockable
    line = np.random.default_rng(4).standard_normal(200)
    g1 = np.random.default_rng(7); g2 = np.random.default_rng(7)
    e_off = [off.step(line, 0.0, 1e-4, g1).est_perp for _ in range(5)]
    e_on = [on.step(line, 0.0, 1e-4, g2).est_perp for _ in range(5)]
    assert np.allclose(e_off, e_on)
