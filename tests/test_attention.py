"""Acceptance tests for the scalar attention metric a(t) (attention.py).

a(t) is the goal's fallback deliverable: a single [0,1] scalar per timestep,
derived from the best 2D gaze reconstruction (M4 PF @ 11.8 kHz), for use as a
robot-teleop imitation-learning feature. These tests assert it behaves as an
intake/observability proxy — necessary, not sufficient:

  - a(t) is a finite [0,1] signal.
  - the intake gate falls monotonically with gaze speed (saccadic suppression).
  - a collapses on blink / out-of-FOV lines (unobservable gaze).
  - a is lower in-flight during saccades than during fixation.
  - a is lower for right/temporal gaze (right-eye SLO FOV loss) than left/centre.
  - the IL-alignment helpers (resample / for_timestamps) are correct + bounded.
  - the computation is deterministic.

Skips cleanly if the reconstruction cache is absent (CI without the heavy build).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import attention as A
import khz2d


pytestmark = pytest.mark.skipif(
    khz2d.load_method(A.BEST_METHOD) is None,
    reason=f"reconstruction cache '{A.BEST_METHOD}' not built")


@pytest.fixture(scope="module")
def att():
    return A.attention_signal()


@pytest.fixture(scope="module")
def vd(att):
    return A._validate(att)


def test_a_is_bounded_finite_signal(att):
    a = att.a
    assert a.shape == att.t.shape
    assert np.all(np.isfinite(a))
    assert a.min() >= 0.0 and a.max() <= 1.0
    # not degenerate (some attention, some suppression)
    assert 0.05 < a.mean() < 0.95
    assert np.mean(a > 0.5) > 0.1


def test_intake_gate_monotone_in_speed(vd):
    """The saccadic-suppression gate must fall as gaze speed rises."""
    means = [m for (_, _, m, n) in vd["band_a"] if np.isfinite(m)]
    assert len(means) >= 4
    # strictly non-increasing across ascending speed bands
    for lo, hi in zip(means, means[1:]):
        assert hi <= lo + 1e-6, f"intake gate not monotone: {means}"
    # slow band near 1, fast band near 0 — the gate actually discriminates
    assert means[0] > 0.85
    assert means[-1] < 0.2


def test_blink_and_out_of_fov_suppressed(vd):
    """Invalid lines (blink / out-of-FOV) carry essentially no attention."""
    assert vd["a_invalid"] < 0.15
    assert vd["a_invalid"] < 0.5 * vd["a_fix"]


def test_saccade_below_fixation(vd):
    """In-flight saccade lines score below fixation/pursuit (intake suppression)."""
    assert vd["a_sac"] < vd["a_fix"]


def test_fov_loss_lowers_attention(vd):
    """Right/temporal gaze (right-eye SLO FOV loss) scores below left/centre gaze."""
    assert vd["a_right"] < vd["a_leftc"]
    assert vd["a_valid"] > vd["a_invalid"]


def test_resample_bounded_and_sized(att):
    t_grid, a_grid = att.resample(100.0)
    assert a_grid.shape == t_grid.shape
    assert np.all(np.isfinite(a_grid))
    assert a_grid.min() >= 0.0 and a_grid.max() <= 1.0
    # 100 Hz over a ~70 s trace -> ~7000 steps
    assert 5000 < len(a_grid) < 9000


def test_for_timestamps_nearest_matches_native(att):
    """nearest-pool at the native line times returns the native a(t)."""
    idx = np.arange(0, len(att.t), 5000)
    got = att.for_timestamps(att.t[idx], pool="nearest")
    assert np.allclose(got, att.a[idx])


def test_for_timestamps_mean_pool_bounded(att):
    ts = np.arange(att.t[0] + 0.1, att.t[-1] - 0.1, 0.02)  # 50 Hz control grid
    a = att.for_timestamps(ts, pool="mean")
    assert a.shape == ts.shape
    assert np.all(np.isfinite(a))
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_deterministic(att):
    a2 = A.attention_signal()
    assert np.array_equal(att.a, a2.a)
    assert att.v_half == a2.v_half
