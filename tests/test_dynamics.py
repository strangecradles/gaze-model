"""G9 acceptance tests: the IMM dynamics prior (dynamics.py).

A pure-prior open-loop rollout (no observations) must reproduce real oculomotor
statistics, the load-bearing one being the saccade MAIN SEQUENCE (peak velocity
vs amplitude) — PLAN.md forbids a generic-smoothness prior precisely because it
would NOT reproduce the main sequence. These tests therefore measure, from the
rollout itself:

  * the saccade main-sequence log-log slope is positive and in a band around the
    measured along-channel slope (~0.37; see calib.main_seq_slope), peak velocity
    increases with amplitude, and large saccades approach VMAX;
  * the saccade rate from the transition matrix is ~0.5-4 /s;
  * fixation/pursuit velocity is band-limited (PSD low-pass) with bounded
    acceleration, and drift over a fixation is a few arcmin (not white-noise
    explosive, not frozen);
  * the saccade displacement distribution is main-sequence-consistent (peak vel
    tied to amplitude), NOT a Gaussian-process / random-walk;
  * determinism with a seeded rng; states finite; mode posterior sums to 1.

Tolerances are set from the measured rollout statistics (run
``python dynamics.py --report``): slope ~0.34-0.42 across seeds/rates (band
[0.2, 0.8]); rate ~2.2/s; drift p90 ~2'; peak-vel max / VMAX = 1.0.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dynamics  # noqa: E402


ARCMIN_PER_ROW = dynamics.ARCMIN_PER_ROW


@pytest.fixture(scope="module")
def ro():
    # 60 s @ 1 kHz pure-prior rollout (no observations); deterministic seed.
    return dynamics.rollout(duration_s=60.0, fs=1000.0, seed=0)


# ---------------------------------------------------------------------------
# helpers: extract saccade (mode==1) runs straight from the rollout arrays
# ---------------------------------------------------------------------------


def _saccade_runs(ro):
    """Return (amplitude_rows, peakvel_rows_s) per mode==1 run extracted from the
    rollout trajectory — exactly the measurement the DoD calls for."""
    m = ro.mode == 1
    amps, pvs = [], []
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            # displacement from the sample just before onset to the run end
            i0 = i - 1 if i > 0 else i
            disp = np.hypot(ro.pos_perp[j - 1] - ro.pos_perp[i0],
                            ro.pos_along[j - 1] - ro.pos_along[i0])
            pv = np.max(np.hypot(ro.vel_perp[i:j], ro.vel_along[i:j]))
            amps.append(disp)
            pvs.append(pv)
            i = j
        else:
            i += 1
    return np.asarray(amps), np.asarray(pvs)


# ---------------------------------------------------------------------------
# saccade main sequence reproduced by the pure-prior rollout
# ---------------------------------------------------------------------------


def test_main_sequence_slope_from_rollout(ro):
    amps, pvs = _saccade_runs(ro)
    good = (amps > 1.0) & (pvs > 0)            # > ~0.5 arcmin
    assert good.sum() >= 50, "need enough saccades to fit a main sequence"
    la = np.log10(amps[good])
    lp = np.log10(pvs[good])
    slope = np.polyfit(la, lp, 1)[0]
    # positive (bigger saccades faster) and in a band around the measured ~0.37
    assert 0.2 <= slope <= 0.8, f"main-sequence log-log slope {slope:.3f} out of band"


def test_peakvel_increases_with_amplitude(ro):
    amps, pvs = _saccade_runs(ro)
    good = (amps > 1.0) & (pvs > 0)
    # peak velocity rises with amplitude: strong positive rank/linear correlation
    r = np.corrcoef(amps[good], pvs[good])[0, 1]
    assert r > 0.5, f"peak-vel vs amplitude correlation {r:.3f} too weak"
    # binned means are monotone-ish: small saccades slower than large ones
    lo = pvs[good][amps[good] < np.median(amps[good])].mean()
    hi = pvs[good][amps[good] >= np.median(amps[good])].mean()
    assert hi > lo, "large saccades must be faster than small ones"


def test_large_saccades_approach_vmax(ro):
    amps, pvs = _saccade_runs(ro)
    # the fastest saccades approach the main-sequence asymptote VMAX
    assert pvs.max() / dynamics.VMAX_ROWS_S > 0.9
    assert pvs.max() <= dynamics.VMAX_ROWS_S * (1.0 + 1e-9)


def test_rollout_matches_ground_truth_main_sequence(ro):
    # the per-saccade commanded ground truth and the array-extracted runs agree
    slope_gt, _ = dynamics.main_sequence_fit(ro.sacc_amplitude, ro.sacc_peakvel)
    amps, pvs = _saccade_runs(ro)
    good = (amps > 1.0) & (pvs > 0)
    slope_arr = np.polyfit(np.log10(amps[good]), np.log10(pvs[good]), 1)[0]
    assert abs(slope_gt - slope_arr) < 0.1, \
        f"array slope {slope_arr:.3f} vs ground-truth {slope_gt:.3f} diverge"


# ---------------------------------------------------------------------------
# saccade rate from the transition matrix
# ---------------------------------------------------------------------------


def test_saccade_rate_in_band(ro):
    rate = len(ro.sacc_amplitude) / ro.t[-1]
    assert 0.5 <= rate <= 4.0, f"saccade rate {rate:.2f}/s out of ~0.5-4 band"


def test_transition_matrix_rows_sum_to_one():
    for dt in (1e-3, 5e-4, 1e-2):
        T = dynamics.transition_matrix(dt)
        assert np.allclose(T.sum(axis=1), 1.0)
        assert (T >= 0).all() and (T <= 1).all()
    # pursuit->saccade per-dt prob matches the declared hazard
    dt = 1e-3
    T = dynamics.transition_matrix(dt)
    expected = 1.0 - np.exp(-dynamics.HAZARD_PURSUIT_TO_SACCADE * dt)
    assert abs(T[0, 1] - expected) < 1e-12


# ---------------------------------------------------------------------------
# fixation / pursuit: band-limited, bounded acceleration, plausible drift
# ---------------------------------------------------------------------------


def test_pursuit_drift_plausible(ro):
    segs = dynamics.fixation_segments(ro, min_dur_s=0.2)
    assert len(segs) >= 10
    drift = np.array([np.ptp(ro.pos_perp[a:b]) for a, b in segs]) * ARCMIN_PER_ROW
    # a few arcmin: NOT frozen (some drift) and NOT white-noise explosive
    assert np.median(drift) > 0.05, "fixation must drift (not frozen)"
    assert np.percentile(drift, 90) < 10.0, \
        f"fixation drift p90 {np.percentile(drift,90):.2f}' too large (envelope-riding)"


def test_pursuit_acceleration_bounded(ro):
    segs = dynamics.fixation_segments(ro, min_dur_s=0.2)
    accs = [np.abs(np.diff(ro.vel_perp[a:b])) / ro.dt for a, b in segs]
    acc = np.concatenate(accs)
    # explicit cap is honored (the anti-runaway guarantee); accel is small
    assert acc.max() <= dynamics.ACCEL_CAP_ROWS_S2 + 1e-6
    assert np.percentile(acc, 99) < dynamics.ACCEL_CAP_ROWS_S2


def test_pursuit_velocity_band_limited(ro):
    segs = dynamics.fixation_segments(ro, min_dur_s=0.5)
    vp = np.concatenate([ro.vel_perp[a:b] for a, b in segs])
    f = np.fft.rfftfreq(len(vp), ro.dt)
    P = np.abs(np.fft.rfft(vp - vp.mean())) ** 2
    cum = np.cumsum(P) / P.sum()
    f90 = f[np.searchsorted(cum, 0.9)]
    nyq = 0.5 / ro.dt
    # OU pursuit velocity is low-pass: 90% of the power well below Nyquist
    assert f90 < 0.2 * nyq, f"pursuit velocity not band-limited (90% power to {f90:.1f} Hz)"


# ---------------------------------------------------------------------------
# NOT generic smoothness: saccade displacement is main-sequence-consistent
# ---------------------------------------------------------------------------


def test_saccade_kinematics_not_gaussian_process(ro):
    """A generic-smoothness / random-walk mode would have saccade peak velocity
    roughly INDEPENDENT of displacement (driven by a fixed process-noise sigma).
    The main-sequence prior ties them: V_peak is a tight, saturating function of
    amplitude. Assert that tie is strong (high correlation, low scatter about the
    fitted law) and that the law saturates (not linear-unbounded)."""
    amps, pvs = _saccade_runs(ro)
    good = (amps > 1.0) & (pvs > 0)
    a, v = amps[good], pvs[good]
    # 1) peak velocity is tightly tied to amplitude (a GP/random walk would not be)
    assert np.corrcoef(a, v)[0, 1] > 0.5
    # 2) the relationship saturates: doubling amplitude well above A0 does NOT
    #    double the peak velocity (a fixed-sigma random walk has no such ceiling)
    A0_rows = dynamics.A0_ROWS
    big = a > 4 * A0_rows
    if big.sum() >= 5:
        assert v[big].mean() > 0.8 * dynamics.VMAX_ROWS_S, \
            "large saccades must sit near the VMAX ceiling (saturating main sequence)"
    # 3) the velocities the saccade mode produces are ballistic (hundreds of
    #    deg/s), far above any pursuit velocity — there is no single sigma that
    #    explains both modes (the whole point of the IMM split)
    assert v.max() > 100.0 * dynamics.ROWS_PER_DEG  # > 100 deg/s


def test_v_peak_is_the_only_saccade_speed_law():
    # the documented invariant: saccade speed comes from the main sequence, not
    # process noise. v_peak is monotone and saturates toward VMAX.
    A = np.array([2.0, 8.0, 20.0, 200.0, 2000.0]) / ARCMIN_PER_ROW
    V = np.asarray(dynamics.v_peak(A))
    assert np.all(np.diff(V) > 0)
    assert V[-1] <= dynamics.VMAX_ROWS_S
    assert V[-1] > 0.95 * dynamics.VMAX_ROWS_S
    assert V[0] < 0.5 * dynamics.VMAX_ROWS_S       # small saccades slow


# ---------------------------------------------------------------------------
# determinism, finiteness, valid posterior
# ---------------------------------------------------------------------------


def test_deterministic_with_seed():
    a = dynamics.rollout(duration_s=10.0, fs=1000.0, seed=11)
    b = dynamics.rollout(duration_s=10.0, fs=1000.0, seed=11)
    assert np.array_equal(a.pos_perp, b.pos_perp)
    assert np.array_equal(a.pos_along, b.pos_along)
    assert np.array_equal(a.vel_perp, b.vel_perp)
    assert np.array_equal(a.mode, b.mode)
    # different seed -> different trajectory
    c = dynamics.rollout(duration_s=10.0, fs=1000.0, seed=12)
    assert not np.array_equal(a.pos_perp, c.pos_perp)


def test_states_finite(ro):
    for arr in (ro.pos_perp, ro.pos_along, ro.vel_perp, ro.vel_along, ro.t):
        assert np.isfinite(arr).all()
    assert set(np.unique(ro.mode)).issubset({0, 1})


def test_predict_advances_and_is_finite():
    rng = np.random.default_rng(0)
    st = dynamics.init_state(256, mode=0)
    for _ in range(100):
        st = dynamics.predict(st, 1e-3, rng)
        assert np.isfinite(st.pos_perp).all() and np.isfinite(st.vel_perp).all()
        assert np.isfinite(st.pos_along).all() and np.isfinite(st.vel_along).all()
    assert st.n == 256


def test_mode_posterior_sums_to_one():
    rng = np.random.default_rng(5)
    st = dynamics.init_state(1000, mode=0)
    for _ in range(80):
        st = dynamics.predict(st, 1e-3, rng)
    p_pursuit, p_saccade = dynamics.mode_posterior(st)
    assert abs((p_pursuit + p_saccade) - 1.0) < 1e-9
    assert 0.0 <= p_pursuit <= 1.0 and 0.0 <= p_saccade <= 1.0
    # most of the time the eye is in pursuit/fixation
    assert p_pursuit > 0.5
    # a degenerate (all-zero) weight vector still yields a valid distribution
    st.weight[:] = 0.0
    pp, ps = dynamics.mode_posterior(st)
    assert abs((pp + ps) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# amplitude distribution is heavy-tailed small (~2-40 arcmin)
# ---------------------------------------------------------------------------


def test_saccade_amplitudes_microsaccade_scale(ro):
    amp_arc = ro.sacc_amplitude * ARCMIN_PER_ROW
    # median in the small-saccade / microsaccade band; tail extends to tens of '
    assert 4.0 < np.median(amp_arc) < 20.0
    assert np.percentile(amp_arc, 90) < 50.0
    assert amp_arc.min() > 0.0
