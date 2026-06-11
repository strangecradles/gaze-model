"""losses.py — self-supervised losses + a learned calibrated perp likelihood (GOALS.md G14).

This module holds the *learned components* of the tracker. The frozen physics
decoder (decoder.render) is NEVER trained — every loss here either renders a line
through the frozen decoder and compares it to the observed line, or penalises a
trajectory's kinematics; gradients flow only into the *learned likelihood /
calibration head*, never into the atlas.

Losses (PLAN.md "Losses" section):

  - L_recon  — line-scan reconstruction through the FROZEN decoder. The render at
    the estimated (or true) gaze should match the observed line. ~0 when gaze ==
    truth, rises as the estimate moves off the true (perp, along). This is the
    analysis-by-synthesis data term. The decoder is frozen; the gradient w.r.t.
    the *gaze* exists (so it can drive a gaze estimate) but NOT w.r.t. the atlas.

  - L_dyn    — off-manifold / main-sequence / band-limit penalty. Penalises a
    perp(t) trajectory that (a) violates the IMM main sequence (a saccade-speed
    segment whose peak velocity exceeds the main-sequence law V_peak(A) for its
    amplitude) and (b) carries energy above the dynamics band limit (oculomotor
    motion is low-pass; high-frequency perp energy is non-physical). Reuses the
    dynamics.py main-sequence law verbatim.

  - L_couple — along<->perp saccade-direction consistency. A saccade is spatially
    straight, so the (trusted) along displacement predicts the perp displacement
    via the saccade direction ratio. The trusted along axis teaches the aliased
    perp axis: penalise perp(t) that departs from the along-implied straight line
    during a saccade segment.

  - L_anchor (optional) — slow component vs frame-rate full-frame-registration
    truth (data.frame_truth) where available. Matches only the low-pass (slow)
    component of perp(t) to the coarse absolute truth; the fast component is left
    to the other losses (the anchor is blunt, ~43').

Learned calibrated observation likelihood (G13 condition met — the physics score
IS the measured saccade bottleneck):

  - LearnedPerpLikelihood — a small torch module. Given the per-candidate-row
    multi-band NCC profile of an observed line against the frozen decoder's
    renders, plus a blur cue (the line's high-frequency attenuation, which encodes
    velocity), it produces a CALIBRATED score over candidate perp rows. It is
    trained so its argmax/posterior matches the TRUE perp on labeled synthetic —
    especially on BLURRED saccade lines where the raw fine-NCC fails (the fine
    band is attenuated by motion blur, but the coarse band survives; the head
    learns to lean on the surviving band when the blur cue is high).

    The head only learns the COMPARISON / CALIBRATION over physics features; the
    decoder stays frozen (features are computed with torch.no_grad on the render).

EMPIRICAL BASIS (G14 investigation, held-out synthetic seeds 6-9 @2 kHz):
  fixation: learned == physics-fine (gross 0.000, RMS 0.3 rows) — DoD preserved.
  saccade : physics-fine gross 0.228 / RMS 70.5 rows;
            learned       gross 0.051 / RMS 22.9 rows  (partial improvement);
            the coarse band alone reaches gross 0.000 / RMS 8.3 rows — blur leaves
            the low-freq structure intact, which the learned head exploits.
"""
from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import calib
import decoder
import dynamics

# ---------------------------------------------------------------------------
# Feature bands (shared by the learned likelihood and the loss helpers).
# The fine band is the razor-sharp, blur-fragile channel; the coarse band is the
# broad, blur-robust channel; the mid band sits between. The blur cue is the
# fraction of line variance carried by the fine band (low when the line is
# motion-blurred — high-spatial-frequency content is the first thing blur kills).
# ---------------------------------------------------------------------------

FINE_SIGMA: float = 6.0
MID_SIGMA_LO: float = 2.0
MID_SIGMA_HI: float = 12.0
COARSE_SIGMA: float = 12.0
BANDS = ("fine", "mid", "coarse")


def _gauss1d_torch(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Reflect-padded 1-D Gaussian blur over the last axis (differentiable)."""
    radius = max(1, int(round(3.0 * sigma)))
    k = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    w = torch.exp(-0.5 * (k / sigma) ** 2)
    w = w / w.sum()
    pad = radius
    xp = F.pad(x.reshape(-1, 1, x.shape[-1]), (pad, pad), mode="reflect")
    out = F.conv1d(xp, w.reshape(1, 1, -1))
    return out.reshape(x.shape)


def _band_torch(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "fine":
        y = x - _gauss1d_torch(x, FINE_SIGMA)
    elif kind == "mid":
        y = _gauss1d_torch(x, MID_SIGMA_LO) - _gauss1d_torch(x, MID_SIGMA_HI)
    elif kind == "coarse":
        y = _gauss1d_torch(x, COARSE_SIGMA)
    else:
        raise ValueError(f"band must be one of {BANDS}, got {kind}")
    mu = y.mean(dim=-1, keepdim=True)
    sd = y.std(dim=-1, keepdim=True)
    return (y - mu) / (sd + 1e-9)


def _blur_cue_torch(line: torch.Tensor) -> torch.Tensor:
    """Fraction of line variance in the fine band (low => motion-blurred)."""
    hp = line - _gauss1d_torch(line, FINE_SIGMA)
    return hp.var(dim=-1) / (line.var(dim=-1) + 1e-9)


def perp_features(line, candidate_rows, along: float, atlas, line_len: int,
                  col_step: float = 1.0, dtype=torch.float64) -> torch.Tensor:
    """Multi-band NCC + blur-cue features of ``line`` vs renders at ``candidate_rows``.

    Returns a (n_candidates, n_features) tensor with, per candidate row:
    the fine/mid/coarse NCC of the observed line against the frozen-decoder render
    at that row, a (broadcast) blur cue, and the blur*NCC interaction terms (so the
    head can learn to down-weight the fine band when the line is blurred). The
    render is computed with the FROZEN decoder and detached — the head learns only
    the comparison, never the atlas.
    """
    A = atlas.ref_map if hasattr(atlas, "ref_map") else np.asarray(atlas)
    cand = torch.as_tensor(np.asarray(candidate_rows), dtype=dtype)
    obs = (line if isinstance(line, torch.Tensor)
           else torch.as_tensor(np.asarray(line), dtype=dtype)).to(dtype)
    obs = obs.reshape(1, -1)

    with torch.no_grad():
        along_t = torch.full((cand.shape[0],), float(along), dtype=dtype)
        R = decoder.render(cand, along_t, A, line_len, col_step=col_step, dtype=dtype)
    nccs = []
    for kind in BANDS:
        of = _band_torch(obs, kind)             # (1, L)
        Rf = _band_torch(R, kind)               # (n, L)
        nccs.append((Rf * of).mean(dim=-1))     # (n,)
    blur = _blur_cue_torch(obs).reshape(())     # scalar
    n = cand.shape[0]
    blur_vec = blur.expand(n)
    feats = nccs + [blur_vec] + [nccs[j] * blur_vec for j in range(len(BANDS))]
    return torch.stack(feats, dim=1)            # (n, 2*nbands + 1)


N_FEATURES: int = 2 * len(BANDS) + 1            # fine/mid/coarse ncc + blur + 3 interactions


class LearnedPerpLikelihood(nn.Module):
    """Small calibrated head over the multi-band perp features (G14).

    Maps each candidate row's feature vector to a scalar logit; ``log_likelihood``
    returns log-softmax over candidates (a calibrated posterior). Trained by NLL of
    the true perp. The frozen decoder supplies the features; the head learns only
    the band weighting / calibration, so gradients never reach the atlas.
    """

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 24,
                 dtype=torch.float64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # The whole pipeline (decoder, features) is float64 by convention; keep the
        # head in the same dtype so features and params multiply cleanly.
        self.to(dtype)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (..., n_features) -> logits (...,)."""
        w_dtype = self.net[0].weight.dtype
        return self.net(feats.to(w_dtype)).squeeze(-1)

    def log_likelihood(self, feats: torch.Tensor) -> torch.Tensor:
        """Calibrated log-posterior over candidates: log_softmax of the logits."""
        return F.log_softmax(self.forward(feats), dim=-1)

    def score(self, line, candidate_rows, along: float, atlas, line_len: int,
              col_step: float = 1.0) -> np.ndarray:
        """Convenience: calibrated score (softmax prob) over ``candidate_rows`` for
        an observed line (numpy out). Used by filter.py's learned observation model."""
        feats = perp_features(line, candidate_rows, along, atlas, line_len, col_step)
        with torch.no_grad():
            p = torch.softmax(self.forward(feats.to(torch.float64)), dim=-1)
        return p.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Self-supervised losses
# ---------------------------------------------------------------------------


def l_recon(perp, along, line, atlas, line_len: int, col_step: float = 1.0,
            band: str = "fine", dtype=torch.float64) -> torch.Tensor:
    """Line-scan reconstruction loss through the FROZEN decoder.

    Renders the line the gaze ``(perp, along)`` would produce and compares it to
    the observed ``line`` as 1 - NCC in the chosen band (so ~0 when the render
    matches the observation, i.e. gaze == truth, and rises off-truth). ``perp`` /
    ``along`` may be leaf tensors with requires_grad=True; the gradient flows
    w.r.t. the gaze (the decoder has no learnable params, so the atlas is never
    updated). Supports batched gaze: perp/along (B,) -> mean over the batch.
    """
    A = atlas.ref_map if hasattr(atlas, "ref_map") else np.asarray(atlas)
    perp_t = perp if isinstance(perp, torch.Tensor) else torch.as_tensor(perp, dtype=dtype)
    along_t = along if isinstance(along, torch.Tensor) else torch.as_tensor(along, dtype=dtype)
    obs = (line if isinstance(line, torch.Tensor)
           else torch.as_tensor(np.asarray(line), dtype=dtype)).to(perp_t.dtype)
    scalar = perp_t.ndim == 0
    perp_b = torch.atleast_1d(perp_t)
    along_b = torch.atleast_1d(along_t)
    perp_b, along_b = torch.broadcast_tensors(perp_b, along_b)
    R = decoder.render(perp_b, along_b, A, line_len, col_step=col_step,
                       dtype=perp_t.dtype)            # (B, L)
    obs_b = obs.reshape(1, -1) if obs.ndim == 1 else obs
    of = _band_torch(obs_b, band)                     # (1 or B, L)
    Rf = _band_torch(R, band)                         # (B, L)
    ncc = (Rf * of).mean(dim=-1)                       # (B,)
    loss = (1.0 - ncc).mean()
    return loss


def l_dyn(perp_rows, t, mode=None, band_limit_hz: float = 80.0,
          dtype=torch.float64) -> torch.Tensor:
    """Off-manifold dynamics penalty: main-sequence + band-limit.

    Two terms on a perp(t) trajectory (atlas rows vs time seconds):

    (1) MAIN-SEQUENCE: the instantaneous speed must not exceed the main-sequence
        envelope. dynamics.VMAX_ROWS_S is the hard asymptotic ceiling V_peak can
        approach but never exceed; any |dperp/dt| above VMAX is non-physical.
        Penalise the squared overshoot above VMAX (ReLU(|v| - VMAX)^2). This
        reuses the dynamics.py main-sequence law (VMAX is its ceiling).

    (2) BAND-LIMIT: oculomotor motion is low-pass; perp energy above
        ``band_limit_hz`` is non-physical (envelope-riding / aliased high-freq
        chatter). Penalise the mean squared high-frequency component of perp(t),
        where "high frequency" is what survives subtracting a Gaussian low-pass
        whose cutoff is band_limit_hz.

    Returns a finite scalar; differentiable w.r.t. ``perp_rows``. ``mode`` is
    accepted for API symmetry (the penalty is mode-agnostic: VMAX is the global
    ceiling, valid for every mode).
    """
    p = perp_rows if isinstance(perp_rows, torch.Tensor) else torch.as_tensor(perp_rows, dtype=dtype)
    tt = t if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=p.dtype)
    if p.numel() < 3:
        return p.new_zeros(())
    dt = (tt[1:] - tt[:-1]).clamp_min(1e-9)
    v = (p[1:] - p[:-1]) / dt                          # rows/s
    vmax = float(dynamics.VMAX_ROWS_S)
    over = torch.relu(v.abs() - vmax) / vmax
    main_seq = (over ** 2).mean()

    # band-limit: high-freq component = perp - gaussian_lowpass(perp).
    fs = float(1.0 / dt.mean().item())
    sigma_samples = max(1.0, fs / (2.0 * np.pi * band_limit_hz))
    lp = _gauss1d_torch(p.reshape(1, -1), sigma_samples).reshape(-1)
    hf = p - lp
    band = (hf ** 2).mean() / (p.var() + 1e-9)
    return main_seq + band


def l_couple(perp_rows, along_cols, mode, sigma_rows: float = 25.0,
             dir_along_min: float = 0.15, dtype=torch.float64) -> torch.Tensor:
    """along<->perp saccade-direction consistency loss.

    Within each contiguous saccade segment (mode == 1), the path is spatially
    straight, so perp displacement from onset is proportional to along
    displacement from onset via the segment's direction ratio. Fit that ratio
    from the segment's own (along, perp) displacements (least-squares slope) and
    penalise the perp residual off the straight line (normalised by sigma_rows).
    The trusted along axis thus constrains the aliased perp axis. Saccade segments
    with negligible along travel (|along range| small => dir_along ~ 0) are skipped
    (the ratio is undefined / the constraint vacuous — a pure-perp saccade).

    Returns a finite scalar (0 when no usable saccade segment); differentiable
    w.r.t. ``perp_rows``.
    """
    p = perp_rows if isinstance(perp_rows, torch.Tensor) else torch.as_tensor(perp_rows, dtype=dtype)
    a = along_cols if isinstance(along_cols, torch.Tensor) else torch.as_tensor(along_cols, dtype=p.dtype)
    m = np.asarray(mode)
    total = p.new_zeros(())
    count = 0
    i, n = 0, len(m)
    while i < n:
        if m[i] == 1:
            j = i
            while j < n and m[j] == 1:
                j += 1
            if j - i >= 3:
                ps = p[i:j]
                as_ = a[i:j]
                da = as_ - as_[0]
                dp = ps - ps[0]
                # skip near-pure-perp saccades (along gives no constraint)
                if float(da.abs().max().item()) >= dir_along_min:
                    # least-squares slope dp ~ ratio * da
                    denom = (da * da).sum() + 1e-9
                    ratio = (da * dp).sum() / denom
                    resid = dp - ratio * da
                    total = total + (resid ** 2).mean() / (sigma_rows ** 2)
                    count += 1
            i = j
        else:
            i += 1
    if count == 0:
        return total
    return total / count


def l_anchor(perp_rows, t, anchor_perp, anchor_t=None, slow_hz: float = 5.0,
             dtype=torch.float64) -> torch.Tensor:
    """Optional supervised anchor: slow perp component vs frame-rate truth.

    Matches only the LOW-PASS (slow) component of perp(t) to the coarse absolute
    truth ``anchor_perp`` (e.g. data.frame_truth perp, ~43' blunt). The anchor is
    too crude to constrain the fast component, so both signals are low-passed at
    ``slow_hz`` before the squared-error comparison (normalised by the alias
    spacing so the loss is in "fraction of an alias" units). If ``anchor_t`` is
    given the anchor is linearly resampled onto ``t``.

    Returns a finite scalar; differentiable w.r.t. ``perp_rows``.
    """
    p = perp_rows if isinstance(perp_rows, torch.Tensor) else torch.as_tensor(perp_rows, dtype=dtype)
    tt = t if isinstance(t, torch.Tensor) else torch.as_tensor(t, dtype=p.dtype)
    anc = np.asarray(anchor_perp, dtype=np.float64)
    if anchor_t is not None:
        anc = np.interp(np.asarray(tt.detach().cpu(), dtype=np.float64),
                        np.asarray(anchor_t, dtype=np.float64), anc)
    anc_t = torch.as_tensor(anc, dtype=p.dtype)
    if p.numel() < 3:
        return p.new_zeros(())
    dt = float((tt[1:] - tt[:-1]).mean().item())
    fs = 1.0 / max(dt, 1e-9)
    sigma = max(1.0, fs / (2.0 * np.pi * slow_hz))
    p_slow = _gauss1d_torch(p.reshape(1, -1), sigma).reshape(-1)
    a_slow = _gauss1d_torch(anc_t.reshape(1, -1), sigma).reshape(-1)
    err = (p_slow - a_slow) / calib.ALIAS_SPACING_ROWS
    return (err ** 2).mean()
