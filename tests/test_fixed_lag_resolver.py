import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import filter as flt  # noqa: E402


def _posterior(perp, logp, p_sacc=0.0, v_perp=None):
    perp = np.asarray(perp, dtype=np.float64)
    logp = np.asarray(logp, dtype=np.float64)
    if v_perp is None:
        v_perp = np.zeros_like(perp)
    else:
        v_perp = np.asarray(v_perp, dtype=np.float64)
    w = np.exp(logp - logp.max())
    w = w / w.sum()
    j = int(np.argmax(logp))
    z = np.zeros_like(perp)
    return flt.StepPosterior(
        est_perp=float(perp[j]),
        est_along=0.0,
        est_v_perp=0.0,
        est_v_along=0.0,
        ess=float(1.0 / np.sum(w ** 2)),
        mode_posterior=(1.0 - p_sacc, p_sacc),
        resampled=False,
        reseeded=False,
        max_ncc=0.5,
        along_sigma_eff=2.0,
        hyp_perp=perp,
        hyp_along=z.copy(),
        hyp_v_perp=v_perp.copy(),
        hyp_v_along=z.copy(),
        hyp_logp=logp,
        pos_perp=perp.copy(),
        pos_along=z.copy(),
        weight=w,
    )


def test_streaming_fixed_lag_matches_batch_resolver():
    posts = [
        _posterior([0.0, 8.0], [0.0, -0.2]),
        _posterior([0.4, 8.2], [-0.1, 0.0]),
        _posterior([0.8, 8.4], [-0.2, 0.0]),
        _posterior([1.2, 8.6], [0.0, -0.5]),
        _posterior([1.6, 8.8], [0.0, -0.6]),
    ]
    rate = 1000.0
    lag_ms = 2.0
    sigma = 3.0
    batch = flt._fixed_lag_resolve(posts, rate, lag_ms, sigma)

    resolver = flt.FixedLagHypothesisResolver(
        rate, lag_ms=lag_ms, transition_sigma_rows=sigma)
    out = np.full(len(posts), np.nan)
    for p in posts:
        est = resolver.push(p)
        if est is not None:
            out[est.index] = est.est_perp
    for est in resolver.flush():
        out[est.index] = est.est_perp

    assert np.array_equal(out, batch[0])
    assert not np.isnan(out).any()


def test_fixed_lag_can_blend_large_pursuit_divergence_to_immediate():
    posts = [
        _posterior([0.0, 20.0], [0.0, -0.01]),
        _posterior([0.0, 20.0], [0.0, -0.01]),
    ]
    posts[0].est_perp = 10.0
    posts[1].est_perp = 10.0
    resolver = flt.FixedLagHypothesisResolver(
        1000.0,
        lag_ms=1.0,
        transition_sigma_rows=3.0,
        blend_immediate=True,
        blend_delta_rows=5.0,
        blend_alpha=0.5,
    )
    est = None
    for post in posts:
        est = resolver.push(post)
    assert est is not None
    assert est.blended_immediate
    assert est.est_perp == 5.0


def _resolve_perp(posts, *, slew_gate=False, slew_max_deg_s=50.0):
    resolver = flt.FixedLagHypothesisResolver(
        1000.0,
        lag_ms=1.0,
        transition_sigma_rows=30.0,
        obs_weight=4.0,
        slew_gate=slew_gate,
        slew_max_deg_s=slew_max_deg_s,
    )
    out = np.full(len(posts), np.nan)
    for p in posts:
        est = resolver.push(p)
        if est is not None:
            out[est.index] = est.est_perp
    for est in resolver.flush():
        out[est.index] = est.est_perp
    return out


def _resolve_perp_with_velocity_cost(posts, cost):
    resolver = flt.FixedLagHypothesisResolver(
        1000.0,
        lag_ms=1.0,
        transition_sigma_rows=30.0,
        obs_weight=4.0,
        velocity_cost=cost,
        velocity_sigma_deg_s=10.0,
    )
    out = np.full(len(posts), np.nan)
    for p in posts:
        est = resolver.push(p)
        if est is not None:
            out[est.index] = est.est_perp
    for est in resolver.flush():
            out[est.index] = est.est_perp
    return out


def _resolve_perp_with_acceleration_cost(posts, cost):
    resolver = flt.FixedLagHypothesisResolver(
        1000.0,
        lag_ms=1.0,
        transition_sigma_rows=30.0,
        obs_weight=4.0,
        acceleration_cost=cost,
        acceleration_sigma_deg_s2=100.0,
    )
    out = np.full(len(posts), np.nan)
    for p in posts:
        est = resolver.push(p)
        if est is not None:
            out[est.index] = est.est_perp
    for est in resolver.flush():
        out[est.index] = est.est_perp
    return out


def test_slew_gate_rejects_pursuit_jump_return_alias():
    posts = [
        _posterior([0.0], [0.0]),
        _posterior([0.0, 10.0], [-1.0, 0.0]),
        _posterior([0.0, 10.0], [0.0, -1.0]),
        _posterior([0.0], [0.0]),
    ]
    ungated = _resolve_perp(posts, slew_gate=False)
    gated = _resolve_perp(posts, slew_gate=True, slew_max_deg_s=50.0)

    assert ungated[1] == 10.0
    assert gated[1] == 0.0
    assert np.all(np.isfinite(gated))


def test_slew_gate_preserves_physiological_ramp():
    posts = [
        _posterior([0.0], [0.0]),
        _posterior([4.0], [0.0]),
        _posterior([8.0], [0.0]),
        _posterior([12.0], [0.0]),
    ]
    gated = _resolve_perp(posts, slew_gate=True, slew_max_deg_s=50.0)
    assert np.array_equal(gated, np.array([0.0, 4.0, 8.0, 12.0]))


def test_velocity_cost_discourages_commit_path_jerk():
    rows_per_s = 4.0 * 1000.0
    posts = [
        _posterior([0.0], [0.0], v_perp=[rows_per_s]),
        _posterior([4.0, 20.0], [-0.3, 0.0], v_perp=[rows_per_s, rows_per_s]),
        _posterior([8.0, 24.0], [-0.3, 0.0], v_perp=[rows_per_s, rows_per_s]),
    ]
    no_cost = _resolve_perp_with_velocity_cost(posts, 0.0)
    with_cost = _resolve_perp_with_velocity_cost(posts, 5.0)
    assert no_cost[1] == 20.0
    assert with_cost[1] == 4.0


def test_acceleration_cost_discourages_jump_return_alias():
    posts = [
        _posterior([0.0], [0.0]),
        _posterior([0.0, 20.0], [-0.2, 0.0]),
        _posterior([0.0, 20.0], [0.0, -0.4]),
        _posterior([0.0], [0.0]),
    ]
    no_cost = _resolve_perp_with_acceleration_cost(posts, 0.0)
    with_cost = _resolve_perp_with_acceleration_cost(posts, 1.0)
    assert no_cost[1] == 20.0
    assert np.array_equal(with_cost, np.zeros(len(posts)))
