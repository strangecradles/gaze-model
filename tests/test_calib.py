"""G2 acceptance tests: row<->arcmin calibration.

Asserts the two independent perp-scale estimates agree within the justified
tolerance, ALIAS_SPACING_ROWS sits in the physically plausible [90, 160] band,
no reported quantity is NaN (the prior 344 Hz NaN path is gone), and
rows_to_arcmin / arcmin_to_rows are exact inverses. The calibration is cached
(cache/calib.json + the data.py intermediates) so the suite stays fast.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calib  # noqa: E402


@pytest.fixture(scope="module")
def cal():
    return calib.calibrate()


# ---- ALIAS_SPACING_ROWS in the plausible band ----

def test_alias_spacing_in_band(cal):
    assert 90.0 <= calib.ALIAS_SPACING_ROWS <= 160.0
    assert 90.0 <= cal.alias_spacing_rows <= 160.0
    # alias spacing is rows-per-1-degree by construction (== 60 arcmin)
    assert abs(calib.rows_to_arcmin(calib.ALIAS_SPACING_ROWS) - 60.0) < 1e-6


def test_rows_per_deg_matches_module_constant(cal):
    assert math.isclose(calib.ROWS_PER_DEG, cal.rows_per_deg, rel_tol=1e-9)
    assert math.isclose(calib.ALIAS_SPACING_ROWS, cal.rows_per_deg, rel_tol=1e-9)
    assert math.isclose(calib.ARCMIN_PER_ROW, 60.0 / cal.rows_per_deg, rel_tol=1e-9)


# ---- two independent estimates agree ----

def test_two_estimates_agree(cal):
    e1, e2 = cal.estimate1_rows_per_deg, cal.estimate2_rows_per_deg
    assert np.isfinite(e1) and np.isfinite(e2)
    assert e1 > 0 and e2 > 0
    residual = abs(e1 - e2) / (0.5 * (e1 + e2))
    assert math.isclose(residual, cal.cross_check_residual, rel_tol=1e-9)
    # measured residual ~12%; tolerance 35% (justified by the coarse-anchor
    # bootstrap spread). Must pass with margin.
    assert cal.cross_check_residual <= cal.agree_tol
    assert cal.estimates_agree is True
    # both estimates land in the plausible band
    assert 90.0 <= e1 <= 160.0
    assert 90.0 <= e2 <= 160.0


# ---- no NaN anywhere (prior NaN path is gone) ----

def test_no_nan_anywhere(cal):
    for k, v in cal.as_dict().items():
        if isinstance(v, float):
            assert np.isfinite(v), f"non-finite calibration quantity {k}={v}"
    # the along main sequence (the source of the old NaN) is finite
    assert np.isfinite(cal.main_seq_slope)
    assert cal.main_seq_n_events > 0
    assert np.isfinite(cal.deg_per_along_px)


# ---- rows_to_arcmin / arcmin_to_rows exact inverses ----

def test_roundtrip_exact():
    rows = np.array([0.0, 1.0, 63.0, 124.641, 600.0])
    back = calib.arcmin_to_rows(calib.rows_to_arcmin(rows))
    assert np.allclose(back, rows, rtol=0, atol=1e-9)
    arc = np.array([-30.0, 0.0, 21.0, 60.0, 120.0])
    back2 = calib.rows_to_arcmin(calib.arcmin_to_rows(arc))
    assert np.allclose(back2, arc, rtol=0, atol=1e-9)


def test_roundtrip_scalar():
    assert math.isclose(
        float(calib.arcmin_to_rows(calib.rows_to_arcmin(42.0))), 42.0, abs_tol=1e-9)
    # one row equals arcmin_per_row arcmin
    assert math.isclose(float(calib.rows_to_arcmin(1.0)), calib.ARCMIN_PER_ROW,
                        rel_tol=1e-12)


# ---- main sequence proves real 12 kHz kinematics ----

def test_main_sequence_finite_and_yields_events(cal):
    # enough supra-threshold saccades to fit a main sequence
    assert cal.main_seq_n_events >= 50
    # log-log peak-vel-vs-amplitude slope is finite and positive (bigger
    # saccades are faster)
    assert np.isfinite(cal.main_seq_slope)
    assert cal.main_seq_slope > 0
    # along channel grounds to a finite, sane angular scale
    assert np.isfinite(cal.deg_per_along_px)
    assert abs(cal.along_grounding_r) > 0.3
