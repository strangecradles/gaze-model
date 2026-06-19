import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from along_quality import AlongQualityModel  # noqa: E402
import people_fov_pf as pf  # noqa: E402


def _tiny_people_inputs():
    rng = np.random.default_rng(5)
    H, W = 360, 16
    frames = []
    base = rng.normal(120.0, 25.0, (H, W)).astype(np.float32)
    for f in range(3):
        frames.append((f, base.copy()))

    n_lines = 808 + W
    rate = 8080.0
    lm = dict(
        t=np.arange(n_lines, dtype=np.float64) / rate,
        qv=np.full(n_lines, 0.75, dtype=np.float32),
        qh=np.full(n_lines, 0.65, dtype=np.float32),
        lam_v=np.zeros(n_lines, dtype=np.float32),
        rdx=np.zeros(n_lines, dtype=np.float32),
        con=np.ones(n_lines, dtype=np.float32),
        line_rate=np.float64(rate),
    )
    ch = dict(
        t=np.arange(3, dtype=np.float64) / 10.0,
        fps=np.float64(10.0),
        ok=np.ones(3, dtype=bool),
        inc_x=np.zeros(3, dtype=np.float64),
        inc_y=np.zeros(3, dtype=np.float64),
        x=np.zeros(3, dtype=np.float64),
        y=np.zeros(3, dtype=np.float64),
    )
    return frames, lm, ch


def test_short_people_run_writes_along_quality_audit_fields(tmp_path):
    frames, lm, ch = _tiny_people_inputs()
    sub = pf.Subject("TinyAQ", "tiny_aq")
    cache_path = tmp_path / "m4_dpf_physics_aq_qv_power_s2_6_g1.npz"

    out = pf.run_m4(
        sub, lm, ch, cache_path=str(cache_path), rebuild=True,
        n_particles=24, padw=8, line_len=20,
        frame_reader=lambda: iter(frames),
        quality_scaled_along=True,
        along_quality_model=AlongQualityModel.qv_power(2.0, 6.0, 1.0, 0.2, 0.8),
        multi_hypothesis=True,
        hypothesis_top_k=3,
    )

    assert cache_path.name.startswith("m4_dpf_physics_aq_")
    assert cache_path.name != "m4_dpf_physics.npz"
    for key in ("along_quality", "along_sigma_eff", "hyp_count", "qv", "qh", "con"):
        assert key in out
        assert len(out[key]) == len(out["t"])
    for key in (
        "x_px_immediate",
        "y_px_immediate",
        "fixed_lag_resolved",
        "fixed_lag_hyp_index",
        "fixed_lag_hyp_rank",
        "fixed_lag_hyp_logp_gap",
        "fixed_lag_hyp_logp_margin",
        "fixed_lag_local_best_x_px",
        "fixed_lag_local_best_y_px",
        "fixed_lag_path_x_px",
        "fixed_lag_path_y_px",
        "fixed_lag_blended_immediate",
    ):
        assert key in out
        assert len(out[key]) == len(out["t"])
    assert int(out["fixed_lag_lines"]) > 0
    assert float(out["fixed_lag_ms"]) > 0.0
    for key in (
        "hypothesis_velocity_cost",
        "hypothesis_velocity_sigma_deg_s",
        "hypothesis_acceleration_cost",
        "hypothesis_acceleration_sigma_deg_s2",
        "hypothesis_blend_immediate",
        "hypothesis_blend_delta_rows",
        "hypothesis_blend_alpha",
        "hypothesis_blend_saccade_p",
        "motion_prior",
        "motion_prior_sigma_rows",
        "motion_prior_ncc_thr",
        "motion_prior_tau_s",
    ):
        assert key in out
    valid = out["valid"].astype(bool)
    assert np.any(valid)
    assert np.all(np.isfinite(out["along_sigma_eff"][valid]))
    assert np.nanmax(out["hyp_count"]) >= 1
    assert np.all(out["fixed_lag_resolved"][valid])
