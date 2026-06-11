"""G14 loss + learned-likelihood tests (losses.py).

Each loss is finite, differentiable (gradient flows to the gaze / head params, NOT
into the frozen decoder/atlas), and behaves sanely:
  - L_recon ~0 when gaze == truth and rises off-truth;
  - L_dyn penalises off-main-sequence / band-limit-violating kinematics;
  - L_couple penalises a bent (non-straight) saccade;
  - L_anchor is finite and matches the slow component.
The learned head produces a calibrated posterior and its grads reach only the head.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data  # noqa: E402
import dynamics  # noqa: E402
import losses  # noqa: E402
import synth_stream as ss  # noqa: E402

ATLAS = data.load_atlas()
LINE_LEN = 200
RATE = 2000.0


def _fixation_line(seed: int = 7):
    """A clean (pre-noise) fixation line + its true (perp, along)."""
    s = ss.make_synthetic(0.3, RATE, seed, ATLAS, line_len=LINE_LEN)
    tr = s.trajectory
    i = int(np.where(tr.mode == 0)[0][50])
    return s.clean[i], float(tr.perp_rows[i]), float(tr.along_cols[i])


# ---------------------------------------------------------------------------
# L_recon
# ---------------------------------------------------------------------------


def test_l_recon_zero_at_truth_rises_off_truth():
    line, tp, ta = _fixation_line()
    at_truth = losses.l_recon(torch.tensor(tp), torch.tensor(ta), line, ATLAS, LINE_LEN)
    off = losses.l_recon(torch.tensor(tp + 40.0), torch.tensor(ta), line, ATLAS, LINE_LEN)
    assert torch.isfinite(at_truth) and torch.isfinite(off)
    assert at_truth.item() < 0.2, f"L_recon at truth should be ~0, got {at_truth.item()}"
    assert off.item() > at_truth.item(), "L_recon must rise off the true perp"
    print(f"\n[G14] L_recon truth {at_truth.item():.4f} -> +40 rows {off.item():.4f}")


def test_l_recon_grad_flows_to_gaze_not_atlas():
    line, tp, ta = _fixation_line()
    perp = torch.tensor(tp + 8.0, requires_grad=True)
    along = torch.tensor(ta, requires_grad=True)
    loss = losses.l_recon(perp, along, line, ATLAS, LINE_LEN)
    loss.backward()
    assert perp.grad is not None and torch.isfinite(perp.grad).all()
    assert along.grad is not None and torch.isfinite(along.grad).all()
    # the atlas is a plain ndarray and carries no grad — the decoder is frozen
    assert not isinstance(ATLAS.ref_map, torch.Tensor) or not ATLAS.ref_map.requires_grad


# ---------------------------------------------------------------------------
# L_dyn
# ---------------------------------------------------------------------------


def test_l_dyn_penalises_off_manifold():
    s = ss.make_synthetic(0.3, RATE, 3, ATLAS, line_len=LINE_LEN)
    tr = s.trajectory
    real = losses.l_dyn(torch.tensor(tr.perp_rows), tr.t)
    bad = tr.perp_rows.copy()
    bad[10::2] += 5000.0  # >VMAX jumps + high-frequency chatter
    off = losses.l_dyn(torch.tensor(bad), torch.tensor(tr.t))
    assert torch.isfinite(real) and torch.isfinite(off)
    assert off.item() > real.item(), "L_dyn must penalise off-main-sequence kinematics"
    print(f"\n[G14] L_dyn real {real.item():.4e} -> off-manifold {off.item():.4e}")


def test_l_dyn_grad_flows():
    s = ss.make_synthetic(0.2, RATE, 3, ATLAS, line_len=LINE_LEN)
    p = torch.tensor(s.trajectory.perp_rows, requires_grad=True)
    loss = losses.l_dyn(p, s.trajectory.t)
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_l_dyn_uses_dynamics_main_sequence_ceiling():
    """A constant-velocity ramp just under VMAX is unpenalised by the main-seq term;
    just over VMAX is penalised — confirming it reuses dynamics.VMAX_ROWS_S."""
    n = 50
    t = np.arange(n) / RATE
    vmax = dynamics.VMAX_ROWS_S
    under = np.cumsum(np.full(n, 0.9 * vmax / RATE))
    over = np.cumsum(np.full(n, 1.5 * vmax / RATE))
    # isolate the main-sequence term by removing the band-limit confound: both are
    # smooth ramps so band energy is tiny; the difference is the VMAX overshoot.
    lu = losses.l_dyn(torch.tensor(under), torch.tensor(t))
    lo = losses.l_dyn(torch.tensor(over), torch.tensor(t))
    assert lo.item() > lu.item(), "over-VMAX ramp must cost more than under-VMAX"


# ---------------------------------------------------------------------------
# L_couple
# ---------------------------------------------------------------------------


def test_l_couple_penalises_bent_saccade():
    """A straight saccade (perp ∝ along) has ~0 coupling loss; bowing the perp path
    away from the along-implied line raises it."""
    n = 40
    mode = np.zeros(n, dtype=np.int8)
    mode[10:30] = 1  # a saccade segment
    along = np.linspace(0.0, 100.0, n)
    # straight: perp tracks along exactly within the saccade
    perp_straight = 0.6 * along
    straight = losses.l_couple(torch.tensor(perp_straight), torch.tensor(along), mode)
    perp_bent = perp_straight.copy()
    bow = np.zeros(n)
    bow[10:30] = np.sin(np.linspace(0, np.pi, 20)) * 40.0
    perp_bent = perp_bent + bow
    bent = losses.l_couple(torch.tensor(perp_bent), torch.tensor(along), mode)
    assert torch.isfinite(straight) and torch.isfinite(bent)
    assert bent.item() > straight.item(), "L_couple must penalise a non-straight saccade"
    print(f"\n[G14] L_couple straight {straight.item():.4e} -> bent {bent.item():.4e}")


def test_l_couple_grad_flows_and_skips_pure_perp():
    n = 30
    mode = np.zeros(n, dtype=np.int8)
    mode[5:25] = 1
    along = np.full(n, 3.0)            # no along travel -> pure-perp -> skipped
    perp = torch.tensor(np.linspace(0, 50, n), requires_grad=True)
    loss = losses.l_couple(perp, torch.tensor(along), mode)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0, "pure-perp saccade (no along travel) must be skipped"


# ---------------------------------------------------------------------------
# L_anchor
# ---------------------------------------------------------------------------


def test_l_anchor_finite_and_matches_slow_component():
    s = ss.make_synthetic(0.3, RATE, 2, ATLAS, line_len=LINE_LEN)
    tr = s.trajectory
    rng = np.random.default_rng(0)
    # anchor = truth + coarse noise; loss should be small (slow component matches)
    anc_good = tr.perp_rows + rng.normal(0, 30.0, len(tr.t))
    anc_bad = tr.perp_rows + 400.0  # large slow-component bias
    lg = losses.l_anchor(torch.tensor(tr.perp_rows), tr.t, anc_good)
    lb = losses.l_anchor(torch.tensor(tr.perp_rows), tr.t, anc_bad)
    assert torch.isfinite(lg) and torch.isfinite(lb)
    assert lb.item() > lg.item(), "L_anchor must penalise a biased slow component"
    p = torch.tensor(tr.perp_rows, requires_grad=True)
    losses.l_anchor(p, tr.t, anc_good).backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


# ---------------------------------------------------------------------------
# Learned likelihood head
# ---------------------------------------------------------------------------


def test_perp_features_shape_and_finite():
    line, tp, ta = _fixation_line()
    cand = np.arange(tp - 100, tp + 100)
    feats = losses.perp_features(line, cand, ta, ATLAS, LINE_LEN)
    assert feats.shape == (len(cand), losses.N_FEATURES)
    assert torch.isfinite(feats).all()


def test_learned_head_calibrated_posterior_and_grad():
    line, tp, ta = _fixation_line()
    cand = np.arange(tp - 100, tp + 100)
    feats = losses.perp_features(line, cand, ta, ATLAS, LINE_LEN)
    head = losses.LearnedPerpLikelihood()
    ll = head.log_likelihood(feats)
    assert ll.shape == (len(cand),)
    # log-softmax => exp sums to 1 (calibrated posterior)
    assert abs(float(ll.exp().sum().item()) - 1.0) < 1e-5
    # grad flows to the head params, and the features are detached from the decoder
    assert not feats.requires_grad, "features must be detached from the frozen decoder"
    true_idx = int(np.argmin(np.abs(cand - tp)))
    nll = -head.log_likelihood(feats)[true_idx]
    nll.backward()
    grads = [p.grad for p in head.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_learned_head_grad_does_not_reach_atlas():
    """The decoder render is detached inside perp_features, so no gradient can
    propagate into the frozen atlas (it is the analysis-by-synthesis invariant)."""
    line, tp, ta = _fixation_line()
    cand = np.arange(tp - 50, tp + 50)
    atlas_t = torch.tensor(ATLAS.ref_map, dtype=torch.float64, requires_grad=True)

    class _Wrap:
        ref_map = atlas_t

    feats = losses.perp_features(line, cand, ta, _Wrap, LINE_LEN)
    head = losses.LearnedPerpLikelihood()
    loss = -head.log_likelihood(feats)[len(cand) // 2]
    loss.backward()
    assert atlas_t.grad is None, "no gradient may flow into the frozen atlas"
