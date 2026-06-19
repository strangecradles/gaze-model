import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from along_quality import AlongQualityModel, normalize_subject_quality  # noqa: E402
import calib  # noqa: E402
import data  # noqa: E402
import filter as flt  # noqa: E402
import synth_stream as ss  # noqa: E402


def test_qv_power_curve_is_monotonic():
    model = AlongQualityModel.qv_power(2.0, 18.0, 1.0, 0.2, 0.8)
    sigma = model.sigma(np.array([0.2, 0.4, 0.6, 0.8]), base_sigma=2.0)
    assert np.all(np.diff(sigma) <= 0.0)
    assert sigma[0] >= sigma[-1]


def test_degenerate_subject_quality_is_finite_and_neutral():
    qn = normalize_subject_quality(np.array([0.3, 0.3, np.nan]), 0.3, 0.3)
    assert np.all(np.isfinite(qn))
    assert np.allclose(qn, 0.5)
    model = AlongQualityModel.qv_power(2.0, 10.0, 1.0, 0.3, 0.3)
    sigma = model.sigma(np.array([0.3, np.nan]), base_sigma=2.0)
    assert np.all(np.isfinite(sigma))
    assert np.allclose(sigma, 6.0)


def test_constant_model_reproduces_default_pf_outputs():
    atlas = data.load_atlas()
    stream = ss.make_synthetic(0.08, 1000.0, 12, atlas, line_len=80)
    tr = stream.trajectory
    rng = np.random.default_rng(123)
    along = tr.along_cols + rng.normal(0.0, 0.5, len(tr.t))
    coarse = tr.perp_rows + rng.normal(0.0, flt.COARSE_SIGMA_ROWS, len(tr.t))
    common = dict(
        rate=stream.rate,
        atlas=atlas,
        init_perp=float(tr.perp_rows[0]),
        init_along=float(tr.along_cols[0]),
        n_particles=80,
        perp_spread=calib.ALIAS_SPACING_ROWS,
        along_spread=2.0,
        line_len=stream.line_len,
        coarse_anchor=coarse,
        seed=44,
    )
    base = flt.run(stream.lines, along, **common)
    const = flt.run(
        stream.lines, along, **common,
        along_quality=np.linspace(0.0, 1.0, len(tr.t)),
        quality_scaled_along=True,
        along_quality_model=AlongQualityModel.constant(),
    )
    assert np.array_equal(base.est_perp, const.est_perp)
    assert np.array_equal(base.est_along, const.est_along)
    assert np.allclose(const.along_sigma_eff, flt.SIGMA_ALONG)
    assert np.all(const.hyp_count == 1)
