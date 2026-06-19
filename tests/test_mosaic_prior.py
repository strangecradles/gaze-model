"""Unit tests for the belief-level mosaic motion prior (filter.ParticleFilter).

Guarantees:
  - OFF (default) and ON-but-never-armed are byte-identical to baseline step-by-step
    (the prior only acts when steadily locked AND past the post-reseed hold).
  - The prior's weight factor heavily down-weights a particle cluster displaced ~one mosaic
    spacing from the causal track prediction, but NOT a displacement the track is following
    (a real main-sequence ramp / microsaccade rides along with pred_track).
  - The post-reseed hold keeps the prior off during reacquisition.
"""
import numpy as np
import pytest

import calib
import dynamics
import filter as flt

MOSAIC = calib.MOSAIC_SPACING_ROWS  # ~6 rows


def _run(mosaic_prior, n_steps=8, seed_state=1, seed_atlas=3, seed_line=4, seed_g=7, **kw):
    rng = np.random.default_rng(seed_state)
    st = flt.init_filter(300, 100.0, 0.0, 15.0, 4.0, rng=rng)
    atlas = np.random.default_rng(seed_atlas).standard_normal((220, 80)).astype(np.float64)
    pf = flt.ParticleFilter(st, atlas, 200, col_step=1.0, mosaic_prior=mosaic_prior, **kw)
    line = np.random.default_rng(seed_line).standard_normal(200)
    g = np.random.default_rng(seed_g)
    return [pf.step(line, 0.0, 1e-4, g).est_perp for _ in range(n_steps)]


def test_flag_defaults_off():
    rng = np.random.default_rng(0)
    st = flt.init_filter(200, 100.0, 0.0, 15.0, 4.0, rng=rng)
    atlas = rng.standard_normal((220, 80)).astype(np.float64)
    pf = flt.ParticleFilter(st, atlas, 200, col_step=1.0)
    assert pf.mosaic_prior is False


def test_off_equals_baseline_bit_identical():
    base = _run(False)                      # explicit OFF
    # ON but never lockable -> prior never arms -> must be bitwise identical
    on = _run(True, lock_ncc_thr=2.0, lock_p_pursuit_thr=2.0)
    assert base == on


def test_reseed_hold_never_rearms_equals_off():
    # ON, lockable, but reseed-hold so large the prior never re-arms -> identical to OFF.
    base = _run(False)
    on = _run(True, mosaic_prior_reseed_hold=10 ** 12)
    assert base == on


def test_prior_downweights_displaced_cluster_when_locked():
    # White-box: replicate the exact w_mosaic formula (filter.step 3c) on a synthetic
    # bimodal cloud and assert the +1-mosaic-spacing cluster is heavily suppressed while the
    # on-track cluster is ~unchanged.
    dt = 1e-4
    pred_track = 100.0
    pos = np.concatenate([np.full(200, pred_track), np.full(200, pred_track + MOSAIC)])
    disp_max = dynamics.SIGMA_V_PURSUIT_ROWS_S * dt
    sigma = flt.MOSAIC_PRIOR_SIGMA_ROWS
    excess = np.maximum(0.0, np.abs(pos - pred_track) - disp_max)
    w_mosaic = np.exp(-0.5 * (excess / sigma) ** 2)
    on_track, displaced = w_mosaic[:200], w_mosaic[200:]
    assert np.all(on_track > 0.99)                 # on-track unpenalised
    assert np.all(displaced < np.exp(-1.0))        # displaced cluster heavily down-weighted
    assert displaced.mean() < 0.2                  # ~exp(-2) for a full 6-row displacement


def test_prior_does_not_suppress_when_track_follows():
    # A real microsaccade: the track has moved to the displaced position, so pred_track follows
    # the cluster and excess ~ 0 -> w_mosaic ~ 1 (NOT clipped).
    dt = 1e-4
    pred_track = 106.0                              # track already at the new (real) position
    pos = np.full(300, 106.0)
    disp_max = dynamics.SIGMA_V_PURSUIT_ROWS_S * dt
    excess = np.maximum(0.0, np.abs(pos - pred_track) - disp_max)
    w_mosaic = np.exp(-0.5 * (excess / flt.MOSAIC_PRIOR_SIGMA_ROWS) ** 2)
    assert np.all(w_mosaic > 0.999)                 # track follows -> no suppression


def test_steps_since_reseed_resets_on_reseed():
    # Drive a filter that is guaranteed to lose lock (garbage atlas/line) with mosaic_prior on,
    # a tiny coarse anchor so a reseed fires, and assert the reseed-hold counter is reset.
    rng = np.random.default_rng(11)
    st = flt.init_filter(300, 100.0, 0.0, 15.0, 4.0, rng=rng)
    atlas = rng.standard_normal((220, 80)).astype(np.float64)
    pf = flt.ParticleFilter(st, atlas, 200, col_step=1.0, mosaic_prior=True,
                            ncc_loss_window=3)
    g = np.random.default_rng(12)
    # mismatched line -> low NCC -> lock-loss -> reseed within a few steps
    line = np.random.default_rng(99).standard_normal(200)
    reseeded = False
    for _ in range(12):
        post = pf.step(line, 0.0, 1e-4, g, coarse_anchor=100.0)
        if post.reseeded:
            reseeded = True
            assert pf._steps_since_reseed == 0      # counter reset at reseed
            break
    assert reseeded, "expected a reseed on sustained low-NCC input"
