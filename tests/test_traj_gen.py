"""G7 acceptance tests: label-bearing 2D gaze trajectory generator.

Asserts the generated trajectories reproduce real oculomotor statistics:

  * saccade main sequence -- log-log peak-velocity-vs-amplitude slope is positive
    and sits in a band around the measured ~0.371 (calib); peak velocity is
    monotone-increasing with amplitude (binned) and large saccades approach Vmax.
  * drift spectrum shape -- fixation-segment velocity PSD is LOW-PASS (negative
    log-log slope; negligible power at high frequency), i.e. band-limited, not
    white.
  * microsaccade rate -- in a plausible range (~0.2-3 /s).
  * trajectory invariants -- finite, both modes present, perp_rows/along_cols
    inside the atlas bounds [0,600]/[0,1200] for default settings, finite v_*.
  * determinism -- same seed -> identical trajectory.
  * schema -- the field units obey the documented contract (G9/G10 depend on it).

Tolerances were set by MEASURING the generator (see traj_gen.py docstring): at a
well-resolved rate the main-sequence slope is ~0.40-0.52 across seeds, the drift
velocity PSD slope is ~-2.1, and the microsaccade rate is ~0.35-1.35 /s. The
main-sequence slope is read from the gradient-measured velocity, so saccade
pulses must be sampled densely enough to resolve the peak; the slope tests run at
``_RES_RATE`` (5 kHz) where even microsaccades are resolved. Bound/finite/mode
invariants are rate-agnostic.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calib       # noqa: E402
import traj_gen    # noqa: E402

# rate at which short microsaccade pulses are well-resolved (gradient-measured
# peak velocity tracks the V_peak law); the main-sequence slope is read here.
_RES_RATE = 5000.0
_DUR = 20.0
_N_SEEDS = 12

ATLAS_ROWS = traj_gen.ATLAS_ROWS
ATLAS_COLS = traj_gen.ATLAS_COLS


@pytest.fixture(scope="module")
def traj():
    return traj_gen.sample_trajectory(duration=_DUR, rate=_RES_RATE, seed=0)


# ---- trajectory invariants ----

def test_schema_shapes_and_dtypes(traj):
    n = int(round(_DUR * _RES_RATE))
    for name in ("t", "perp_arcmin", "along_arcmin", "perp_rows", "along_cols",
                 "v_perp", "v_along", "mode"):
        arr = getattr(traj, name)
        assert arr.shape == (n,), f"{name} has wrong shape {arr.shape}"
    assert traj.mode.dtype == np.int8
    assert traj.rate == _RES_RATE


def test_all_finite(traj):
    for name in ("t", "perp_arcmin", "along_arcmin", "perp_rows", "along_cols",
                 "v_perp", "v_along"):
        assert np.all(np.isfinite(getattr(traj, name))), f"{name} non-finite"


def test_both_modes_present(traj):
    modes = set(np.unique(traj.mode).tolist())
    assert 0 in modes and 1 in modes


def test_inside_atlas_bounds(traj):
    assert traj.perp_rows.min() >= 0.0 and traj.perp_rows.max() <= ATLAS_ROWS
    assert traj.along_cols.min() >= 0.0 and traj.along_cols.max() <= ATLAS_COLS


def test_schema_unit_contract(traj):
    # perp_rows/along_cols are the arcmin motion mapped through ARCMIN_PER_ROW and
    # centred on the (default) atlas anchor; isotropic atlas-sampling for along.
    assert np.allclose(traj.perp_rows,
                       traj.anchor_row + traj.perp_arcmin / calib.ARCMIN_PER_ROW)
    assert np.allclose(traj.along_cols,
                       traj.anchor_col + traj.along_arcmin / calib.ARCMIN_PER_ROW)
    assert traj.anchor_row == traj_gen.DEFAULT_ANCHOR_ROW
    assert traj.anchor_col == traj_gen.DEFAULT_ANCHOR_COL
    # v_* are the atlas-unit time derivatives
    assert np.allclose(traj.v_perp, np.gradient(traj.perp_rows) * traj.rate)
    assert np.allclose(traj.v_along, np.gradient(traj.along_cols) * traj.rate)


def test_determinism():
    a = traj_gen.sample_trajectory(duration=10.0, rate=2000.0, seed=7)
    b = traj_gen.sample_trajectory(duration=10.0, rate=2000.0, seed=7)
    assert np.array_equal(a.perp_rows, b.perp_rows)
    assert np.array_equal(a.along_cols, b.along_cols)
    assert np.array_equal(a.v_perp, b.v_perp)
    assert np.array_equal(a.mode, b.mode)
    # a different seed gives a different trajectory
    c = traj_gen.sample_trajectory(duration=10.0, rate=2000.0, seed=8)
    assert not np.array_equal(a.perp_rows, c.perp_rows)


# ---- saccade main sequence (the real kinematics) ----

def _seed_slopes(rate=_RES_RATE, dur=_DUR, n_seeds=_N_SEEDS):
    out = []
    for s in range(n_seeds):
        tr = traj_gen.sample_trajectory(duration=dur, rate=rate, seed=s)
        amps, pvs = traj_gen.main_sequence(tr)
        if amps.size >= 8:
            out.append(traj_gen.main_sequence_slope(amps, pvs))
    return np.asarray(out)


def test_main_sequence_slope_in_band():
    slopes = _seed_slopes()
    assert slopes.size >= _N_SEEDS - 2
    # positive (bigger saccades are faster) and within a band around the measured
    # ~0.371 log-log slope. Measured generator range ~0.39-0.52 -> [0.2, 0.7].
    assert np.all(slopes > 0.0)
    assert np.all(slopes >= 0.2)
    assert np.all(slopes <= 0.7)
    # the calib-measured along-channel slope (the anchor) is itself in-band
    cal = calib.calibrate()
    assert 0.2 <= cal.main_seq_slope <= 0.7


def test_peak_velocity_monotone_with_amplitude():
    # pool many seeds for a dense main sequence, then bin by amplitude
    amps, pvs = [], []
    for s in range(_N_SEEDS):
        tr = traj_gen.sample_trajectory(duration=_DUR, rate=_RES_RATE, seed=s)
        a, p = traj_gen.main_sequence(tr)
        amps.append(a)
        pvs.append(p)
    amps = np.concatenate(amps)
    pvs = np.concatenate(pvs)
    assert amps.size > 50
    order = np.argsort(amps)
    a_sorted, p_sorted = amps[order], pvs[order]
    bins = np.array_split(np.arange(a_sorted.size), 6)
    med = np.array([np.median(p_sorted[b]) for b in bins])
    # binned median peak velocity strictly increases with amplitude
    assert np.all(np.diff(med) > 0)


def test_large_saccades_approach_vmax():
    amps, pvs = [], []
    for s in range(_N_SEEDS):
        tr = traj_gen.sample_trajectory(duration=_DUR, rate=_RES_RATE, seed=s)
        a, p = traj_gen.main_sequence(tr)
        amps.append(a)
        pvs.append(p)
    amps = np.concatenate(amps)
    pvs = np.concatenate(pvs)
    big = amps > 200.0  # > ~3.3 deg
    assert big.sum() >= 3
    # large saccades reach a substantial fraction of Vmax (saturating law)
    assert np.median(pvs[big]) / traj_gen.VMAX_ARCMIN_S > 0.5
    # but never exceed Vmax (with a small discretisation margin)
    assert pvs.max() <= traj_gen.VMAX_ARCMIN_S * 1.05


def test_peak_velocity_law_matches_measured_slope():
    # the V_peak law itself, sampled over a saccade amplitude range, reproduces
    # the measured ~0.371 small-amplitude log-log slope within band
    amps = np.logspace(np.log10(2.0), np.log10(720.0), 400)  # arcmin
    pvs = traj_gen.peak_velocity_arcmin_s(amps)
    slope = traj_gen.main_sequence_slope(amps, pvs)
    assert 0.2 <= slope <= 0.7


# ---- fixational drift spectrum (low-pass, band-limited) ----

def test_drift_spectrum_is_low_pass():
    from numpy.fft import rfft, rfftfreq
    rate = _RES_RATE
    slopes, hi_fracs = [], []
    for s in range(_N_SEEDS):
        tr = traj_gen.sample_trajectory(duration=_DUR, rate=rate, seed=s)
        v = (np.gradient(tr.perp_arcmin) * rate)[tr.mode == 0]
        v = v - v.mean()
        if v.size < 512:
            continue
        power = np.abs(rfft(v)) ** 2
        f = rfftfreq(v.size, 1.0 / rate)
        band = (f >= 5.0) & (f <= 300.0)
        slope = np.polyfit(np.log10(f[band]), np.log10(power[band] + 1e-15), 1)[0]
        slopes.append(slope)
        hi_fracs.append(power[f > 500.0].sum() / power[f > 1.0].sum())
    slopes = np.asarray(slopes)
    hi_fracs = np.asarray(hi_fracs)
    assert slopes.size >= _N_SEEDS - 2
    # velocity PSD falls with frequency (Brownian/OU ~ -2), strongly negative
    assert np.all(slopes < -0.5)
    assert np.median(slopes) < -1.0
    # negligible velocity power above 500 Hz (band-limited, not white)
    assert np.all(hi_fracs < 0.05)


# ---- microsaccade rate ----

def test_microsaccade_rate_plausible():
    rate = _RES_RATE
    rates = []
    for s in range(20):
        tr = traj_gen.sample_trajectory(duration=_DUR, rate=rate, seed=s)
        amps, _ = traj_gen.main_sequence(tr)
        n_micro = int(np.sum(amps < 30.0))  # microsaccades < ~0.5 deg
        rates.append(n_micro / tr.t[-1])
    rates = np.asarray(rates)
    # plausible microsaccade rate band ~0.2-3 /s
    assert 0.2 <= np.mean(rates) <= 3.0
    assert np.all(rates >= 0.1)
    assert np.all(rates <= 4.0)
