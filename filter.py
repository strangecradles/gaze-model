"""filter.py — multimodal particle-filter CORE + REACQUISITION (GOALS.md G10/G11/G12).

A particle filter (Monte-Carlo localization / DPF) over the shared state
``(perp, along, v_perp, v_along, mode)`` that estimates 2D gaze from the 1D
line-scan stream by ANALYSIS-BY-SYNTHESIS. This is the component PLAN.md flags as
the repeatedly-failed piece (four worse-than-raw Kalmans); the prescribed fix is
a genuinely MULTIMODAL particle belief plus the main-sequence IMM prior, never a
unimodal Gaussian collapse.

Per timestep (one observed line, dt = 1/rate):

  1. PREDICT — ``state = dynamics.predict(state, dt, rng)``. The IMM prior
     (pursuit OU + saccade ballistic main sequence) propagates every particle.
     No generic-smoothness fallback (that lives in dynamics.py and is forbidden).

  2. OBSERVATION WEIGHT (multimodal perp) — render every particle's line in ONE
     batched ``decoder.render`` call (N, L). High-pass (fine band, sigma ~6) both
     the observed line and the N renders, z-score each, and take the per-particle
     normalized cross-correlation ``ncc = mean(obs_hp * render_hp)``. Map to a
     likelihood weight ``w_obs = exp(BETA * (ncc - ncc.max()))``. BETA is tuned
     (~20) so the weights discriminate the sharp true peak from alias peaks
     without collapsing onto a single particle. The MULTIMODALITY is carried by
     the particle cloud spreading over the alias peaks — we never fit or collapse
     to a Gaussian.

  3. TRUSTED ALONG measurement (low variance) — the along axis is the trusted,
     precise channel, so the along-shift measurement enters as a tight Gaussian
     on the along POSITION: ``w_along = exp(-0.5*((pos_along - along_meas)/
     sigma_along)^2)`` with a SMALL ``sigma_along`` (~2 cols).

  3b. DIRECTION COUPLING (G12 / L_couple) — for particles in saccade mode, the
     saccade is spatially straight so the along and perp displacements from onset
     are proportional: ``(pos_perp - sac_perp_onset) = (sac_dir_perp/sac_dir_along)
     * (pos_along - sac_along_onset)``.  This is exact for the dynamics.py
     min-jerk model (verified to machine precision; see G12 investigation).  A
     coupling Gaussian ``w_couple = exp(-0.5*((pos_perp - pred_perp)/sigma_couple)^2)``
     is added for saccade-mode particles whose ``|sac_dir_along| >= COUPLE_DIR_ALONG_MIN``
     (otherwise the direction ratio blows up for pure-perp saccades where along
     gives no constraint).  ``COUPLE_SIGMA_ROWS`` accommodates the model mismatch
     between traj_gen (sin² profile) and dynamics (min-jerk): the bow error peaks
     at ~23r for 200r saccades, giving sigma=25r. This coupling activates whenever
     the IMM naturally transitions particles to saccade mode; it correctly
     discriminates right- vs wrong-direction particles via the trusted along channel,
     implementing PLAN.md's ``L_couple`` principle within the particle filter.
     The coupling has no effect on fixation (mode==0) particles.

  4. ``weight *= w_obs * w_along * w_couple``; normalize; track ``ESS = 1/sum(w^2)``.

  5a. LOCK-LOSS DETECTION (G11) — DISTINCT from normal ESS-resampling. The fine-
     NCC perp peak is razor-sharp, so ESS collapses on ~97% of healthy fixation
     steps; using ESS alone to detect lock loss would fire constantly and destroy
     the working lock. Instead we monitor the OBSERVATION QUALITY directly: track
     a rolling count of consecutive steps where ``max_ncc < NCC_LOCK_LOSS_THR``
     (typically 0.35 — well below the healthy minimum of ~0.50 and well above the
     garbage-line maximum of ~0.28). When this count reaches ``NCC_LOSS_WINDOW``
     (default 5 steps = 1.25 ms at 4 kHz), or when it reaches the hard
     ``COAST_CAP`` (default 100 steps = 25 ms), a RESEED fires.

  5b. RESEED (G11) — redraw all particles from
         perp  ~ N(coarse_anchor, RESEED_PERP_SIGMA)
         along ~ N(along_meas,   RESEED_ALONG_SIGMA)
     and reset velocities to zero, mode to pursuit, weights to uniform. The coarse
     anchor is an ABSOLUTE MEASUREMENT (it repositions the cloud to the broad
     ~43' perp estimate) — it does NOT act as a veto on otherwise-valid particles.
     ``coarse_anchor`` is passed per-step as an argument to ``step()`` / ``run()``;
     for the synthetic tests, simulate it as ``true_perp + N(0, COARSE_SIGMA_ROWS)``.
     ``n_reseeds`` and a per-step boolean ``reseeded`` are exposed in the result.

  5c. NORMAL RESAMPLE (G10) — when ESS < ESS_FRAC*N AND no reseed fired, apply
     systematic resampling + position roughening (anti-impoverishment). This path
     fires on essentially every healthy fixation step because the sharp peak drives
     ESS→1; it is the normal tracking mechanism and must NOT trigger a reseed.

  6. POSTERIOR per step — keep the full weighted particle cloud AND a point
     estimate. The point estimate is the weighted mean of the particles within
     half an alias spacing of the MAP particle, so an alias cluster elsewhere in
     the cloud does not bias the estimate (a plain weighted mean would).

State / dynamics are reused verbatim from dynamics.py (the IMM is NOT
reimplemented here). The observation model is likelihood.py's fine band, batched.

Tunable constants ``BETA``, ``SIGMA_ALONG``, ``ESS_FRAC``, ``ROUGHEN_PERP``,
``ROUGHEN_ALONG``, ``NCC_LOCK_LOSS_THR``, ``NCC_LOSS_WINDOW``, ``COAST_CAP``,
``RESEED_PERP_SIGMA``, ``RESEED_ALONG_SIGMA``, ``COARSE_SIGMA_ROWS``,
``COUPLE_SIGMA_ROWS``, ``COUPLE_DIR_ALONG_MIN`` are module-level and overridable
per call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from along_quality import AlongQualityModel
import calib
import decoder
import dynamics
import losses as losses_mod
from dynamics import ParticleState

# ---------------------------------------------------------------------------
# Tunable constants (tuned in the G10 investigation; see module docstring).
# perp RMS through a 4 kHz fixation/drift segment with these defaults is
# ~0.08 rows (~0.04'), far below the 0.1 deg (~12.5 row) DoD threshold.
# ---------------------------------------------------------------------------

BETA: float = 20.0              # observation-likelihood sharpness (w_obs = exp(BETA*ncc))
SIGMA_ALONG: float = 2.0        # along-position measurement std (cols) — trusted channel
ESS_FRAC: float = 0.5           # resample when ESS < ESS_FRAC * N
ROUGHEN_PERP: float = 0.5       # post-resample perp jitter std (rows) — anti-impoverishment
ROUGHEN_ALONG: float = 0.5      # post-resample along jitter std (cols)
HP_SIGMA: float = 6.0           # fine-band high-pass sigma (matches likelihood.py)

# ---------------------------------------------------------------------------
# G11 reacquisition constants (tuned from investigation; see module docstring).
# Healthy fixation max_ncc: median=0.78, min=0.50.
# Garbage/lock-loss max_ncc: median=0.13, max=0.28.  Threshold 0.35 is cleanly
# between the two distributions with a large margin on both sides.
# ---------------------------------------------------------------------------
NCC_LOCK_LOSS_THR: float = 0.35  # max_ncc below this counts as a bad observation
NCC_LOSS_WINDOW: int = 5         # consecutive bad steps before reseed fires (~1.25 ms @ 4 kHz)
COAST_CAP: int = 100             # hard cap: force reseed after this many bad steps (no infinite coast)
COARSE_SIGMA_ROWS: float = 0.717 * calib.ROWS_PER_DEG  # ~89 rows ≈ 43' coarse anchor uncertainty
RESEED_PERP_SIGMA: float = calib.ALIAS_SPACING_ROWS     # reseed spread ≈ 1 alias spacing (~125 rows)
RESEED_ALONG_SIGMA: float = 3.0  # along reseed spread (cols)

# ---------------------------------------------------------------------------
# G12 direction-coupling constants (L_couple from PLAN.md).
# For saccade-mode particles the coupling constraint pred_perp = onset_perp +
# (dir_p/dir_a) * (pos_along - onset_along) is EXACT for the dynamics.py
# min-jerk model (verified to machine precision in the G12 investigation).
# COUPLE_SIGMA_ROWS accommodates the mismatch between traj_gen (sin² profile,
# np.gradient velocities) and dynamics (min-jerk substeps): the bow error peaks
# at ~23r for 200r saccades.  COUPLE_DIR_ALONG_MIN excludes near-pure-perp
# saccades where the ratio dir_p/dir_a blows up and the constraint is vacuous.
# ---------------------------------------------------------------------------
COUPLE_SIGMA_ROWS: float = 25.0  # coupling Gaussian sigma (rows); accommodates ~23r bow error
COUPLE_DIR_ALONG_MIN: float = 0.15  # minimum |sac_dir_along| to apply coupling

# Lock-gated gain (de-jitter): soft likelihood + minimal roughen during steady pursuit
# lock; restore hot values during reseed / low-NCC / saccade-dominant steps.
BETA_HOT: float = 20.0
BETA_COOL: float = 4.0
ROUGHEN_PERP_HOT: float = 0.5
ROUGHEN_PERP_COOL: float = 0.08
ROUGHEN_ALONG_HOT: float = 0.5
ROUGHEN_ALONG_COOL: float = 0.08
LOCK_NCC_THR: float = 0.35
LOCK_P_PURSUIT_THR: float = 0.9

# Belief-level mosaic motion prior (de-flip): during steady locked pursuit, down-weight
# particles displaced from the causal track prediction beyond the per-step physiological
# (main-sequence) budget, at the FINE mosaic scale. Default OFF -> byte-identical.
MOSAIC_PRIOR_SIGMA_ROWS: float = calib.MOSAIC_SPACING_ROWS / 2.0  # ~3 rows softening shoulder
MOSAIC_PRIOR_RESEED_HOLD: int = 8  # steps after a reseed before the prior re-arms (reacquisition)
MOSAIC_PRIOR_TRACK_TAU_S: float = 0.003  # EMA time-const of the anchor (resists ~0.7ms migration,
#                                          follows ~6 rows/s pursuit). tau->0 reduces to per-step follow.

# SDSLO single-line ambiguity controls (default OFF in run/ParticleFilter).
ALONG_QUALITY_FLOOR: float = 0.20
ALONG_SIGMA_MAX: float = 16.0
HYPOTHESIS_TOP_K: int = 5
HYPOTHESIS_CLUSTER_ROWS: float = 6.0
HYPOTHESIS_MIN_MASS: float = 0.02
HYPOTHESIS_AMBIGUITY_MASS: float = 0.70
HYPOTHESIS_NCC_THR: float = 0.70
HYPOTHESIS_MIN_PARTICLES: int = 6
HYPOTHESIS_TRANSITION_SIGMA_ROWS: float = 3.0
HYPOTHESIS_OBS_WEIGHT: float = 4.0
HYPOTHESIS_LAG_MS: float = 1.0
HYPOTHESIS_VEL_COST: float = 0.0
HYPOTHESIS_VEL_SIGMA_DEG_S: float = 80.0
HYPOTHESIS_ACCEL_COST: float = 0.0
HYPOTHESIS_ACCEL_SIGMA_DEG_S2: float = 5000.0
HYPOTHESIS_BLEND_IMMEDIATE: bool = False
HYPOTHESIS_BLEND_DELTA_ROWS: float = 12.0
HYPOTHESIS_BLEND_ALPHA: float = 0.5
HYPOTHESIS_BLEND_SACCADE_P: float = 0.25
SLEW_GATE_MAX_DEG_S: float = 50.0
SLEW_GATE_COST: float = 1e6
SLEW_GATE_SACCADE_P: float = 0.25


@dataclass
class StepPosterior:
    """The per-step posterior: a point estimate AND the full weighted cloud.

    Point estimate (``est_*``) is the alias-robust weighted mean (particles within
    half an alias spacing of the MAP particle). The weighted particle arrays are
    the genuine multimodal belief; ``mode_posterior`` is ``(p_pursuit, p_saccade)``.
    """

    est_perp: float
    est_along: float
    est_v_perp: float
    est_v_along: float
    ess: float
    mode_posterior: tuple[float, float]
    resampled: bool
    reseeded: bool             # True if a G11 reseed fired this step
    max_ncc: float             # best particle NCC this step (observation quality)
    along_sigma_eff: float     # effective along measurement std after quality scaling
    hyp_perp: np.ndarray       # top-K hypothesis perp estimates (rows)
    hyp_along: np.ndarray      # top-K hypothesis along estimates (cols)
    hyp_v_perp: np.ndarray     # top-K hypothesis perp velocities (rows/s)
    hyp_v_along: np.ndarray    # top-K hypothesis along velocities (cols/s)
    hyp_logp: np.ndarray       # log posterior mass per hypothesis
    # the weighted particle cloud (the multimodal belief)
    pos_perp: np.ndarray
    pos_along: np.ndarray
    weight: np.ndarray


@dataclass
class FilterResult:
    """Result of running the filter over a whole stream."""

    est_perp: np.ndarray       # (T,)
    est_along: np.ndarray      # (T,)
    est_v_perp: np.ndarray     # (T,)
    est_v_along: np.ndarray    # (T,)
    ess: np.ndarray            # (T,)
    p_saccade: np.ndarray      # (T,) saccade-mode posterior
    resampled: np.ndarray      # (T,) bool — whether a normal resample fired this step
    reseeded: np.ndarray       # (T,) bool — whether a G11 reseed fired this step
    max_ncc: np.ndarray        # (T,) best-particle NCC (observation quality monitor)
    along_sigma_eff: np.ndarray  # (T,) effective along measurement std
    hyp_count: np.ndarray      # (T,) number of retained top-K hypotheses
    n_reseeds: int             # total reseed events over the run
    posteriors: list[StepPosterior]  # per-step full weighted clouds
    rate: float


@dataclass(frozen=True)
class FixedLagEstimate:
    """One committed fixed-lag top-K hypothesis estimate."""

    index: int
    est_perp: float
    est_along: float
    est_v_perp: float
    est_v_along: float
    hyp_index: int
    hyp_rank: int
    hyp_logp_gap: float
    hyp_logp_margin: float
    local_best_perp: float
    local_best_along: float
    blended_immediate: bool
    path_perp: float
    path_along: float


# ---------------------------------------------------------------------------
# Band filtering (matches likelihood.py's fine band), batched
# ---------------------------------------------------------------------------


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """Per-row z-score of a (..., L) array (last axis)."""
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + 1e-9)


def _highpass(x: np.ndarray, sigma: float) -> np.ndarray:
    """Fine-band high-pass over the last axis (x - gaussian_lowpass(x))."""
    return x - gaussian_filter1d(x, sigma, axis=-1)


def _fine(x: np.ndarray, sigma: float) -> np.ndarray:
    """Fine-band, z-scored — the observation feature both obs and renders use."""
    return _zscore_rows(_highpass(x, sigma))


# ---------------------------------------------------------------------------
# Filter construction
# ---------------------------------------------------------------------------


def init_filter(n_particles: int, init_perp: float, init_along: float,
                perp_spread: float, along_spread: float,
                init_v_perp: float = 0.0, init_v_along: float = 0.0,
                mode: int = 0, seed: Optional[int] = None,
                rng: Optional[np.random.Generator] = None) -> ParticleState:
    """Seed ``n_particles`` around an initial gaze lock.

    Particles start in pursuit (mode 0) at ``(init_perp, init_along)`` with
    Gaussian position scatter ``perp_spread`` / ``along_spread`` (atlas rows /
    cols). ``perp_spread`` may be BROAD (an alias spacing or more) to exercise the
    multimodal acquisition machinery; the filter still locks the true peak.
    Weights are uniform.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    st = dynamics.init_state(n_particles, pos_perp=init_perp, pos_along=init_along,
                             vel_perp=init_v_perp, vel_along=init_v_along, mode=mode)
    st.pos_perp = st.pos_perp + rng.normal(0.0, float(perp_spread), n_particles)
    st.pos_along = st.pos_along + rng.normal(0.0, float(along_spread), n_particles)
    return st


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

_STATE_FIELDS = (
    "pos_perp", "pos_along", "vel_perp", "vel_along", "mode",
    "sac_phase", "sac_dur", "sac_dur_nat", "sac_amp", "sac_vpeak",
    "sac_dir_perp", "sac_dir_along",
)


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic (low-variance) resampling indices for the given weights."""
    n = weights.shape[0]
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against float drift so searchsorted stays in range
    idx = np.searchsorted(cumulative, positions)
    return np.clip(idx, 0, n - 1)


def _hypothesis_clusters(st: ParticleState, weights: np.ndarray, top_k: int,
                         cluster_rows: float,
                         min_mass: float = 0.0) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Cluster weighted particles into top-K single-line appearance hypotheses."""
    n = len(weights)
    unused = np.ones(n, dtype=bool)
    order = np.argsort(weights)[::-1]
    masks: list[np.ndarray] = []
    vals = {k: [] for k in ("mass", "perp", "along", "v_perp", "v_along")}
    for idx in order:
        if not unused[idx]:
            continue
        center = st.pos_perp[idx]
        mask = unused & (np.abs(st.pos_perp - center) <= cluster_rows)
        mass = float(weights[mask].sum())
        unused[mask] = False
        if mass <= 0.0:
            continue
        if mass < min_mass and masks:
            continue
        wn = weights[mask] / mass
        masks.append(mask)
        vals["mass"].append(mass)
        vals["perp"].append(float(np.sum(wn * st.pos_perp[mask])))
        vals["along"].append(float(np.sum(wn * st.pos_along[mask])))
        vals["v_perp"].append(float(np.sum(wn * st.vel_perp[mask])))
        vals["v_along"].append(float(np.sum(wn * st.vel_along[mask])))
        if len(masks) >= max(1, int(top_k)):
            break
    if not masks:
        imap = int(np.argmax(weights))
        mask = np.zeros(n, dtype=bool)
        mask[imap] = True
        masks = [mask]
        vals = dict(
            mass=[float(weights[imap])],
            perp=[float(st.pos_perp[imap])],
            along=[float(st.pos_along[imap])],
            v_perp=[float(st.vel_perp[imap])],
            v_along=[float(st.vel_along[imap])],
        )
    arrs = {k: np.asarray(v, dtype=np.float64) for k, v in vals.items()}
    mass = arrs["mass"]
    arrs["logp"] = np.log(mass / max(float(mass.sum()), 1e-300) + 1e-300)
    return masks, arrs


def _hypothesis_resample(weights: np.ndarray, masks: list[np.ndarray],
                         rng: np.random.Generator, min_particles: int) -> np.ndarray:
    """Resample while preserving non-negligible top-K hypothesis clusters."""
    n = len(weights)
    if len(masks) <= 1:
        return _systematic_resample(weights, rng)
    masses = np.asarray([weights[m].sum() for m in masks], dtype=np.float64)
    keep = masses > 0.0
    masks = [m for m, k in zip(masks, keep) if k]
    masses = masses[keep]
    if len(masks) <= 1:
        return _systematic_resample(weights, rng)
    masses = masses / masses.sum()
    min_each = min(max(1, int(min_particles)), max(1, n // len(masks)))
    alloc = np.maximum(min_each, np.floor(masses * n).astype(int))
    while int(alloc.sum()) > n:
        j = int(np.argmax(alloc - min_each))
        if alloc[j] <= min_each:
            j = int(np.argmax(alloc))
        alloc[j] -= 1
    while int(alloc.sum()) < n:
        alloc[int(np.argmax(masses))] += 1
    out = []
    for mask, count in zip(masks, alloc):
        ii = np.where(mask)[0]
        ww = weights[ii]
        ww = ww / ww.sum()
        out.append(rng.choice(ii, size=int(count), replace=True, p=ww))
    idx = np.concatenate(out) if out else _systematic_resample(weights, rng)
    if len(idx) < n:
        extra = _systematic_resample(weights, rng)[:n - len(idx)]
        idx = np.concatenate([idx, extra])
    rng.shuffle(idx)
    return idx[:n]


class FixedLagHypothesisResolver:
    """Causal streaming fixed-lag resolver over top-K PF hypotheses.

    ``push()`` ingests one :class:`StepPosterior` and may emit the estimate from
    ``lag_ms`` earlier. ``flush()`` commits the final tail at end-of-run. The
    transition model matches :func:`_fixed_lag_resolve`, but also accepts a
    per-step ``dt_s`` for real people-data runs that skip invalid lines.
    """

    def __init__(self, rate: float, lag_ms: float = HYPOTHESIS_LAG_MS,
                 transition_sigma_rows: float = HYPOTHESIS_TRANSITION_SIGMA_ROWS,
                 obs_weight: float = HYPOTHESIS_OBS_WEIGHT,
                 velocity_cost: float = HYPOTHESIS_VEL_COST,
                 velocity_sigma_deg_s: float = HYPOTHESIS_VEL_SIGMA_DEG_S,
                 acceleration_cost: float = HYPOTHESIS_ACCEL_COST,
                 acceleration_sigma_deg_s2: float = HYPOTHESIS_ACCEL_SIGMA_DEG_S2,
                 slew_gate: bool = False,
                 slew_max_deg_s: float = SLEW_GATE_MAX_DEG_S,
                 slew_gate_cost: float = SLEW_GATE_COST,
                 slew_gate_saccade_p: float = SLEW_GATE_SACCADE_P,
                 blend_immediate: bool = HYPOTHESIS_BLEND_IMMEDIATE,
                 blend_delta_rows: float = HYPOTHESIS_BLEND_DELTA_ROWS,
                 blend_alpha: float = HYPOTHESIS_BLEND_ALPHA,
                 blend_saccade_p: float = HYPOTHESIS_BLEND_SACCADE_P):
        self.rate = float(rate)
        self.lag = max(1, int(round(float(lag_ms) * 1e-3 * self.rate)))
        self.dt_default = 1.0 / self.rate
        self.transition_sigma_rows = float(transition_sigma_rows)
        self.obs_weight = float(obs_weight)
        self.velocity_cost = float(velocity_cost)
        self.velocity_sigma_rows_s = float(velocity_sigma_deg_s) * calib.ROWS_PER_DEG
        self.acceleration_cost = float(acceleration_cost)
        self.acceleration_sigma_rows_s2 = float(acceleration_sigma_deg_s2) * calib.ROWS_PER_DEG
        self.slew_gate = bool(slew_gate)
        self.slew_max_rows_s = float(slew_max_deg_s) * calib.ROWS_PER_DEG
        self.slew_gate_cost = float(slew_gate_cost)
        self.slew_gate_saccade_p = float(slew_gate_saccade_p)
        self.blend_immediate = bool(blend_immediate)
        self.blend_delta_rows = float(blend_delta_rows)
        self.blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        self.blend_saccade_p = float(blend_saccade_p)
        self.hp: list[np.ndarray] = []
        self.ha: list[np.ndarray] = []
        self.hvp: list[np.ndarray] = []
        self.hva: list[np.ndarray] = []
        self.hl: list[np.ndarray] = []
        self.ep: list[float] = []
        self.ea: list[float] = []
        self.evp: list[float] = []
        self.eva: list[float] = []
        self.p_saccade: list[float] = []
        self.scores: list[np.ndarray] = []
        self.back: list[np.ndarray] = []
        self.pair_scores: list[np.ndarray | None] = []
        self.pair_back: list[np.ndarray | None] = []
        self.dt_to_prev: list[float] = []
        self._emitted_until = -1

    def _trace_state(self, t_end: int, state: int, target: int) -> int:
        j = int(state)
        for tt in range(t_end, target, -1):
            j = int(self.back[tt][j])
        return j

    def _estimate(self, target: int, state: int) -> FixedLagEstimate:
        j = int(state)
        local_logp = np.asarray(self.hl[target], dtype=np.float64)
        order = np.argsort(local_logp)[::-1] if local_logp.size else np.asarray([j])
        rank = int(np.where(order == j)[0][0]) if np.any(order == j) else -1
        best = int(order[0]) if order.size else j
        gap = float(local_logp[best] - local_logp[j]) if local_logp.size else 0.0
        if local_logp.size > 1:
            margin = float(local_logp[order[0]] - local_logp[order[1]])
        else:
            margin = float("inf")
        path_perp = float(self.hp[target][j])
        path_along = float(self.ha[target][j])
        est_perp = path_perp
        est_along = path_along
        est_v_perp = float(self.hvp[target][j])
        est_v_along = float(self.hva[target][j])
        blended = False
        if (self.blend_immediate
                and self.p_saccade[target] < self.blend_saccade_p
                and abs(path_perp - self.ep[target]) > self.blend_delta_rows):
            a = self.blend_alpha
            est_perp = (1.0 - a) * path_perp + a * self.ep[target]
            est_along = (1.0 - a) * path_along + a * self.ea[target]
            est_v_perp = (1.0 - a) * est_v_perp + a * self.evp[target]
            est_v_along = (1.0 - a) * est_v_along + a * self.eva[target]
            blended = True
        return FixedLagEstimate(
            index=int(target),
            est_perp=float(est_perp),
            est_along=float(est_along),
            est_v_perp=float(est_v_perp),
            est_v_along=float(est_v_along),
            hyp_index=j,
            hyp_rank=rank,
            hyp_logp_gap=gap,
            hyp_logp_margin=margin,
            local_best_perp=float(self.hp[target][best]),
            local_best_along=float(self.ha[target][best]),
            blended_immediate=blended,
            path_perp=path_perp,
            path_along=path_along,
        )

    def _pair_transition_cost(self, t: int, dt: float) -> tuple[np.ndarray, float]:
        p_sac = max(self.p_saccade[t - 1], self.p_saccade[t])
        sigma = self.transition_sigma_rows
        if p_sac >= 0.25:
            sigma = max(sigma, 0.25 * dynamics.VMAX_ROWS_S * dt)
        dp = self.hp[t][None, :] - self.hp[t - 1][:, None]
        da = self.ha[t][None, :] - self.ha[t - 1][:, None]
        cost = 0.5 * (dp / max(sigma, 1e-6)) ** 2
        cost += 0.05 * 0.5 * (da / max(2.0 * sigma, 1e-6)) ** 2
        if self.velocity_cost > 0.0 and p_sac < self.slew_gate_saccade_p:
            step_v = dp / dt
            pred_v = self.hvp[t - 1][:, None]
            cost += self.velocity_cost * 0.5 * (
                (step_v - pred_v) / max(self.velocity_sigma_rows_s, 1e-6)
            ) ** 2
        if self.slew_gate and p_sac < self.slew_gate_saccade_p:
            cost = np.where(np.abs(dp) > self.slew_max_rows_s * dt,
                            cost + self.slew_gate_cost, cost)
        cost = np.where(np.abs(dp) / dt > 1.05 * dynamics.VMAX_ROWS_S,
                        cost + 1e6, cost)
        return cost, p_sac

    def _trace_pair_state(self, t_end: int, prev: int, cur: int, target: int) -> int:
        tt = int(t_end)
        i = int(prev)
        j = int(cur)
        while target < tt - 1:
            back = self.pair_back[tt]
            if back is None:
                break
            k = int(back[i, j])
            j = i
            i = k
            tt -= 1
        if target == tt:
            return j
        return i

    def _push_second_order(self, t: int, obs: np.ndarray, dt: float) -> FixedLagEstimate | None:
        if t == 0:
            self.scores.append(obs.copy())
            self.back.append(np.zeros(len(obs), dtype=np.int64))
            self.pair_scores.append(None)
            self.pair_back.append(None)
            return None

        trans_cost, p_sac = self._pair_transition_cost(t, dt)
        if t == 1:
            cur_pair = self.scores[0][:, None] - trans_cost + obs[None, :]
            back = np.zeros(cur_pair.shape, dtype=np.int64)
        else:
            prev_pair = self.pair_scores[t - 1]
            if prev_pair is None:
                raise RuntimeError("second-order resolver missing previous pair scores")
            prev_dt = max(self.dt_to_prev[t - 1], 1e-9)
            v_prev = (self.hp[t - 1][None, :] - self.hp[t - 2][:, None]) / prev_dt
            v_cur = (self.hp[t][None, :] - self.hp[t - 1][:, None]) / dt
            cand = prev_pair[:, :, None] - trans_cost[None, :, :]
            if self.acceleration_cost > 0.0 and p_sac < self.slew_gate_saccade_p:
                accel = (v_cur[None, :, :] - v_prev[:, :, None]) / dt
                cand = cand - self.acceleration_cost * 0.5 * (
                    accel / max(self.acceleration_sigma_rows_s2, 1e-6)
                ) ** 2
            back = np.argmax(cand, axis=0).astype(np.int64)
            cur_pair = obs[None, :] + cand[back, np.arange(cand.shape[1])[:, None],
                                           np.arange(cand.shape[2])[None, :]]
        self.pair_scores.append(cur_pair)
        self.pair_back.append(back)
        cur_scores = np.max(cur_pair, axis=0)
        cur_back = np.argmax(cur_pair, axis=0).astype(np.int64)
        self.scores.append(cur_scores)
        self.back.append(cur_back)

        if t >= self.lag:
            target = t - self.lag
            prev, cur = np.unravel_index(int(np.argmax(cur_pair)), cur_pair.shape)
            j = self._trace_pair_state(t, int(prev), int(cur), target)
            self._emitted_until = target
            return self._estimate(target, j)
        return None

    def push(self, post: StepPosterior, dt_s: float | None = None) -> FixedLagEstimate | None:
        hp = post.hyp_perp if len(post.hyp_perp) else np.asarray([post.est_perp])
        ha = post.hyp_along if len(post.hyp_along) else np.asarray([post.est_along])
        hvp = post.hyp_v_perp if len(post.hyp_v_perp) else np.asarray([post.est_v_perp])
        hva = post.hyp_v_along if len(post.hyp_v_along) else np.asarray([post.est_v_along])
        hl = post.hyp_logp if len(post.hyp_logp) else np.asarray([0.0])
        t = len(self.hp)
        self.hp.append(np.asarray(hp, dtype=np.float64))
        self.ha.append(np.asarray(ha, dtype=np.float64))
        self.hvp.append(np.asarray(hvp, dtype=np.float64))
        self.hva.append(np.asarray(hva, dtype=np.float64))
        self.hl.append(np.asarray(hl, dtype=np.float64))
        self.ep.append(float(post.est_perp))
        self.ea.append(float(post.est_along))
        self.evp.append(float(post.est_v_perp))
        self.eva.append(float(post.est_v_along))
        self.p_saccade.append(float(post.mode_posterior[1]))

        obs = self.obs_weight * (self.hl[t] - float(np.max(self.hl[t])))
        dt = self.dt_default if dt_s is None else max(float(dt_s), 1e-9)
        self.dt_to_prev.append(dt if t > 0 else self.dt_default)
        if self.acceleration_cost > 0.0:
            return self._push_second_order(t, obs, dt)

        if t == 0:
            cur = obs.copy()
            bp = np.zeros(len(obs), dtype=np.int64)
        else:
            cost, _ = self._pair_transition_cost(t, dt)
            cand = self.scores[-1][:, None] - cost
            bp = np.argmax(cand, axis=0).astype(np.int64)
            cur = obs + cand[bp, np.arange(len(obs))]
        self.scores.append(cur)
        self.back.append(bp)

        if t >= self.lag:
            target = t - self.lag
            j = self._trace_state(t, int(np.argmax(cur)), target)
            self._emitted_until = target
            return self._estimate(target, j)
        return None

    def flush(self) -> list[FixedLagEstimate]:
        T = len(self.hp)
        if T == 0 or self._emitted_until >= T - 1:
            return []
        final_state = int(np.argmax(self.scores[-1]))
        if self.acceleration_cost > 0.0 and len(self.hp) > 1:
            pair = self.pair_scores[-1]
            if pair is not None:
                prev, cur = np.unravel_index(int(np.argmax(pair)), pair.shape)
                out: list[FixedLagEstimate] = []
                for target in range(self._emitted_until + 1, T):
                    j = self._trace_pair_state(T - 1, int(prev), int(cur), target)
                    out.append(self._estimate(target, j))
                self._emitted_until = T - 1
                return out
        out: list[FixedLagEstimate] = []
        for target in range(self._emitted_until + 1, T):
            j = self._trace_state(T - 1, final_state, target)
            out.append(self._estimate(target, j))
        self._emitted_until = T - 1
        return out


def _fixed_lag_resolve(posteriors: list[StepPosterior], rate: float,
                       lag_ms: float, transition_sigma_rows: float,
                       obs_weight: float = HYPOTHESIS_OBS_WEIGHT,
                       velocity_cost: float = HYPOTHESIS_VEL_COST,
                       velocity_sigma_deg_s: float = HYPOTHESIS_VEL_SIGMA_DEG_S,
                       acceleration_cost: float = HYPOTHESIS_ACCEL_COST,
                       acceleration_sigma_deg_s2: float = HYPOTHESIS_ACCEL_SIGMA_DEG_S2,
                       slew_gate: bool = False,
                       slew_max_deg_s: float = SLEW_GATE_MAX_DEG_S,
                       slew_gate_cost: float = SLEW_GATE_COST,
                       slew_gate_saccade_p: float = SLEW_GATE_SACCADE_P,
                       blend_immediate: bool = HYPOTHESIS_BLEND_IMMEDIATE,
                       blend_delta_rows: float = HYPOTHESIS_BLEND_DELTA_ROWS,
                       blend_alpha: float = HYPOTHESIS_BLEND_ALPHA,
                       blend_saccade_p: float = HYPOTHESIS_BLEND_SACCADE_P
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal fixed-lag resolver over top-K hypotheses from each PF step."""
    T = len(posteriors)
    if T == 0:
        z = np.array([], dtype=np.float64)
        return z, z, z, z
    resolver = FixedLagHypothesisResolver(
        rate, lag_ms=lag_ms, transition_sigma_rows=transition_sigma_rows,
        obs_weight=obs_weight, velocity_cost=velocity_cost,
        velocity_sigma_deg_s=velocity_sigma_deg_s,
        acceleration_cost=acceleration_cost,
        acceleration_sigma_deg_s2=acceleration_sigma_deg_s2,
        slew_gate=slew_gate,
        slew_max_deg_s=slew_max_deg_s, slew_gate_cost=slew_gate_cost,
        slew_gate_saccade_p=slew_gate_saccade_p,
        blend_immediate=blend_immediate,
        blend_delta_rows=blend_delta_rows,
        blend_alpha=blend_alpha,
        blend_saccade_p=blend_saccade_p)
    out_p = np.full(T, np.nan)
    out_a = np.full(T, np.nan)
    out_vp = np.full(T, np.nan)
    out_va = np.full(T, np.nan)
    for post in posteriors:
        est = resolver.push(post)
        if est is not None:
            out_p[est.index] = est.est_perp
            out_a[est.index] = est.est_along
            out_vp[est.index] = est.est_v_perp
            out_va[est.index] = est.est_v_along
    for est in resolver.flush():
        out_p[est.index] = est.est_perp
        out_a[est.index] = est.est_along
        out_vp[est.index] = est.est_v_perp
        out_va[est.index] = est.est_v_along
    return out_p, out_a, out_vp, out_va


# ---------------------------------------------------------------------------
# The filter
# ---------------------------------------------------------------------------


class ParticleFilter:
    """Multimodal particle filter; call :meth:`step` once per observed line.

    Parameters are the frozen physics atlas plus the line geometry the decoder
    needs (``line_len`` / ``col_step``, matching synth_stream). The tunables
    default to the module-level constants and can be overridden per filter.

    G11 reacquisition: the filter monitors the per-step best-particle NCC
    (``max_ncc``). When ``max_ncc`` stays below ``ncc_loss_thr`` for
    ``ncc_loss_window`` consecutive steps (or reaches ``coast_cap``), it performs
    a full RESEED from the coarse perp anchor — distinct from the normal per-step
    ESS-driven resample+roughen which fires on almost every healthy fixation step.
    """

    def __init__(self, state: ParticleState, atlas, line_len: int,
                 col_step: float = 1.0, beta: float = BETA,
                 sigma_along: float = SIGMA_ALONG, ess_frac: float = ESS_FRAC,
                 roughen_perp: float = ROUGHEN_PERP,
                 roughen_along: float = ROUGHEN_ALONG, hp_sigma: float = HP_SIGMA,
                 ncc_loss_thr: float = NCC_LOCK_LOSS_THR,
                 ncc_loss_window: int = NCC_LOSS_WINDOW,
                 coast_cap: int = COAST_CAP,
                 reseed_perp_sigma: float = RESEED_PERP_SIGMA,
                 reseed_along_sigma: float = RESEED_ALONG_SIGMA,
                 couple_sigma: float = COUPLE_SIGMA_ROWS,
                 couple_dir_along_min: float = COUPLE_DIR_ALONG_MIN,
                 likelihood: str = "physics",
                 learned_head: "losses_mod.LearnedPerpLikelihood | None" = None,
                 resample_enabled: bool = True,
                 lock_gated_gain: bool = False,
                 beta_hot: float = BETA_HOT,
                 beta_cool: float = BETA_COOL,
                 roughen_perp_hot: float | None = None,
                 roughen_perp_cool: float = ROUGHEN_PERP_COOL,
                 roughen_along_hot: float | None = None,
                 roughen_along_cool: float = ROUGHEN_ALONG_COOL,
                 lock_ncc_thr: float = LOCK_NCC_THR,
                 lock_p_pursuit_thr: float = LOCK_P_PURSUIT_THR,
                 mosaic_reject: bool = False,
                 mosaic_window_rows: float = 3.0,
                 mosaic_prior: bool = False,
                 mosaic_prior_sigma_rows: float = MOSAIC_PRIOR_SIGMA_ROWS,
                 mosaic_prior_reseed_hold: int = MOSAIC_PRIOR_RESEED_HOLD,
                 mosaic_prior_ncc_thr: float = LOCK_NCC_THR,
                 mosaic_prior_track_tau_s: float = MOSAIC_PRIOR_TRACK_TAU_S,
                 quality_scaled_along: bool = False,
                 along_quality_floor: float = ALONG_QUALITY_FLOOR,
                 along_sigma_max: float = ALONG_SIGMA_MAX,
                 along_quality_model: AlongQualityModel | None = None,
                 multi_hypothesis: bool = False,
                 hypothesis_top_k: int = HYPOTHESIS_TOP_K,
                 hypothesis_cluster_rows: float = HYPOTHESIS_CLUSTER_ROWS,
                 hypothesis_min_mass: float = HYPOTHESIS_MIN_MASS,
                 hypothesis_ambiguity_mass: float = HYPOTHESIS_AMBIGUITY_MASS,
                 hypothesis_ncc_thr: float = HYPOTHESIS_NCC_THR,
                 hypothesis_min_particles: int = HYPOTHESIS_MIN_PARTICLES):
        if likelihood not in ("physics", "learned"):
            raise ValueError("likelihood must be 'physics' or 'learned'")
        if likelihood == "learned" and learned_head is None:
            raise ValueError("likelihood='learned' requires a trained learned_head "
                             "(train.load_head() or train.train_head())")
        self.likelihood = likelihood
        self.learned_head = learned_head
        self.state = state
        self.atlas = atlas.ref_map if hasattr(atlas, "ref_map") else np.asarray(atlas)
        self.line_len = int(line_len)
        self.col_step = float(col_step)
        self.beta = float(beta)
        self.sigma_along = float(sigma_along)
        self.ess_frac = float(ess_frac)
        self.roughen_perp = float(roughen_perp)
        self.roughen_along = float(roughen_along)
        self.hp_sigma = float(hp_sigma)
        # G11 reacquisition params
        self.ncc_loss_thr = float(ncc_loss_thr)
        self.ncc_loss_window = int(ncc_loss_window)
        self.coast_cap = int(coast_cap)
        self.reseed_perp_sigma = float(reseed_perp_sigma)
        self.reseed_along_sigma = float(reseed_along_sigma)
        # G11 state: consecutive-bad-NCC counter
        self._low_ncc_count: int = 0
        self.n_reseeds: int = 0
        # G12 direction-coupling state: saccade onset positions per particle.
        # Initialised to the current particle positions; updated whenever a
        # particle transitions from pursuit to saccade mode (mode 0->1) or after
        # a reseed.  Used by the coupling weight w_couple (step 3b).
        n = state.n
        self.couple_sigma = float(couple_sigma)
        self.couple_dir_along_min = float(couple_dir_along_min)
        self._sac_perp_onset: np.ndarray = state.pos_perp.copy()
        self._sac_along_onset: np.ndarray = state.pos_along.copy()
        self._prev_mode: np.ndarray = state.mode.copy()
        self.resample_enabled = bool(resample_enabled)
        self.lock_gated_gain = bool(lock_gated_gain)
        self.beta_hot = float(beta_hot)
        self.beta_cool = float(beta_cool)
        self.roughen_perp_hot = float(roughen_perp if roughen_perp_hot is None
                                       else roughen_perp_hot)
        self.roughen_perp_cool = float(roughen_perp_cool)
        self.roughen_along_hot = float(roughen_along if roughen_along_hot is None
                                       else roughen_along_hot)
        self.roughen_along_cool = float(roughen_along_cool)
        self.lock_ncc_thr = float(lock_ncc_thr)
        self.lock_p_pursuit_thr = float(lock_p_pursuit_thr)
        # mode-gated mosaic-alias rejection (default OFF -> byte-identical)
        self.mosaic_reject = bool(mosaic_reject)
        self.mosaic_window_rows = float(mosaic_window_rows)
        # belief-level mosaic motion prior (default OFF -> byte-identical)
        self.mosaic_prior = bool(mosaic_prior)
        self.mosaic_prior_sigma_rows = float(mosaic_prior_sigma_rows)
        self.mosaic_prior_reseed_hold = int(mosaic_prior_reseed_hold)
        # dedicated (typically lower) NCC arm threshold so the prior fires AT flip-prone lines
        # (NCC~0.32, below the 0.35 lock threshold); the p_pursuit>=0.9 gate still disarms it on
        # real saccades. Default == LOCK_NCC_THR -> unchanged from the _lock_steady behaviour.
        self.mosaic_prior_ncc_thr = float(mosaic_prior_ncc_thr)
        self.mosaic_prior_track_tau_s = float(mosaic_prior_track_tau_s)
        self._track_perp: float = float(state.pos_perp.mean())   # EMA anchor (armed) / free-follow (disarmed)
        self._steps_since_reseed: int = 10 ** 9   # large -> prior armed once locked
        self.quality_scaled_along = bool(quality_scaled_along)
        self.along_quality_floor = float(along_quality_floor)
        self.along_sigma_max = float(along_sigma_max)
        self.along_quality_model = along_quality_model
        self.multi_hypothesis = bool(multi_hypothesis)
        self.hypothesis_top_k = int(hypothesis_top_k)
        self.hypothesis_cluster_rows = float(hypothesis_cluster_rows)
        self.hypothesis_min_mass = float(hypothesis_min_mass)
        self.hypothesis_ambiguity_mass = float(hypothesis_ambiguity_mass)
        self.hypothesis_ncc_thr = float(hypothesis_ncc_thr)
        self.hypothesis_min_particles = int(hypothesis_min_particles)

    @property
    def n(self) -> int:
        return self.state.n

    def _lock_steady(self, max_ncc: float, p_pursuit: float) -> bool:
        return (max_ncc >= self.lock_ncc_thr
                and p_pursuit >= self.lock_p_pursuit_thr)

    def step(self, line, along_meas: float, dt: float,
             rng: np.random.Generator,
             coarse_anchor: Optional[float] = None,
             along_quality: Optional[float] = None) -> StepPosterior:
        """Advance one timestep on one observed ``line`` and return the posterior.

        Parameters
        ----------
        line          : (L,) observed line.
        along_meas    : trusted along-position measurement (atlas cols).
        dt            : step time (1/rate).
        rng           : seeded generator (determinism).
        coarse_anchor : absolute perp estimate from the coarse channel (atlas rows),
                        ~43' uncertainty. Used for G11 reseeding. Pass None to
                        disable reacquisition (falls back to along_meas as a
                        perp-uninformed reseed centre — still better than coasting).
                        For synthetic tests: ``true_perp + N(0, COARSE_SIGMA_ROWS)``.

        Mutates ``self.state``.
        """
        st = self.state

        if self.mosaic_prior:                 # guarded -> OFF path untouched / byte-identical
            self._steps_since_reseed += 1

        # Track mode before predict so we can detect 0->1 transitions below.
        self._prev_mode[:] = st.mode

        # 1. PREDICT — IMM prior (dynamics.py; NOT reimplemented here)
        st = dynamics.predict(st, dt, rng)
        p_pursuit_pre, _ = dynamics.mode_posterior(st)
        if self.lock_gated_gain:
            beta_eff = (self.beta_cool if p_pursuit_pre >= self.lock_p_pursuit_thr
                        else self.beta_hot)
        else:
            beta_eff = self.beta

        # Record saccade onset positions for particles that just entered saccade
        # mode (pursuit->saccade transition fired inside dynamics.predict).
        new_sac = (st.mode == 1) & (self._prev_mode == 0)
        if new_sac.any():
            self._sac_perp_onset[new_sac] = st.pos_perp[new_sac]
            self._sac_along_onset[new_sac] = st.pos_along[new_sac]

        n = st.n
        obs = np.asarray(line, dtype=np.float64)
        if obs.ndim != 1:
            obs = obs.ravel()

        # 2. OBSERVATION WEIGHT (multimodal perp): batched render + fine NCC
        R = decoder.render(torch.as_tensor(st.pos_perp, dtype=torch.float64),
                           torch.as_tensor(st.pos_along, dtype=torch.float64),
                           self.atlas, self.line_len,
                           col_step=self.col_step).detach().cpu().numpy()
        obs_feat = _fine(obs[None, :], self.hp_sigma)[0]          # (L,)
        ren_feat = _fine(R, self.hp_sigma)                        # (N, L)
        ncc = (ren_feat * obs_feat[None, :]).mean(axis=1)         # (N,)
        # max_ncc (physics fine-NCC) is the G11 lock-loss / observation-quality
        # monitor in BOTH likelihood modes — it is the calibrated lock indicator.
        max_ncc = float(ncc.max())

        if self.likelihood == "physics":
            # subtract max for numerical stability; multimodality is preserved
            # because particles on other alias peaks keep their (smaller, non-zero)
            # weight.
            w_obs = np.exp(beta_eff * (ncc - max_ncc))
        else:
            # LEARNED, blur-aware calibrated likelihood (G14). The head scores the
            # particle perp positions directly (candidate rows = the cloud) from the
            # frozen-decoder multi-band features + a blur cue; during a blurred
            # saccade it leans on the blur-robust band the raw fine-NCC loses. Used
            # as an UNNORMALISED relative weight so the multimodal cloud is preserved
            # (we exp the calibrated logits, max-subtracted for stability).
            feats = losses_mod.perp_features(
                obs, st.pos_perp, float(along_meas), self.atlas, self.line_len,
                col_step=self.col_step)
            with torch.no_grad():
                logit = self.learned_head(feats.to(torch.float64)).detach().cpu().numpy()
            w_obs = np.exp(logit - logit.max())

        # 3. ALONG measurement on the along POSITION. In SDSLO stress/real noisy
        # lines this channel can be uncertain, so an explicit quality can widen
        # the likelihood instead of forcing the PF onto a bad along match.
        sigma_along_eff = self.sigma_along
        if self.along_quality_model is not None:
            q = max_ncc if along_quality is None else float(along_quality)
            sigma_along_eff = self.along_quality_model.sigma_scalar(q, self.sigma_along)
        elif self.quality_scaled_along:
            q = max_ncc if along_quality is None else float(along_quality)
            if not np.isfinite(q):
                q = self.along_quality_floor
            q = float(np.clip(q, self.along_quality_floor, 1.0))
            sigma_along_eff = min(self.along_sigma_max, self.sigma_along / q)
        w_along = np.exp(-0.5 * ((st.pos_along - float(along_meas)) /
                                 sigma_along_eff) ** 2)

        # 3b. DIRECTION COUPLING (G12 / L_couple): for saccade-mode particles
        #     whose along direction component is significant, the straight-saccade
        #     constraint pred_perp = onset_perp + (dir_p/dir_a)*(pos_along -
        #     onset_along) is exact for dynamics.py min-jerk particles and acts as
        #     a tight prior that discriminates right- vs wrong-direction hypotheses
        #     via the trusted along channel.  COUPLE_SIGMA_ROWS=25 accommodates the
        #     ~23r bow error from the traj_gen (sin²) vs dynamics (min-jerk) model
        #     mismatch observed in the G12 investigation.
        w_couple = np.ones(n, dtype=np.float64)
        if self.couple_sigma > 0.0:
            sac_mask = st.mode == 1
            use_coupling = sac_mask & (np.abs(st.sac_dir_along) >= self.couple_dir_along_min)
            if use_coupling.any():
                dir_p = st.sac_dir_perp[use_coupling]
                dir_a = st.sac_dir_along[use_coupling]
                ratio = dir_p / dir_a  # = sac_dir_perp / sac_dir_along (exact for min-jerk)
                da_onset = st.pos_along[use_coupling] - self._sac_along_onset[use_coupling]
                pred_perp = self._sac_perp_onset[use_coupling] + ratio * da_onset
                perp_err = st.pos_perp[use_coupling] - pred_perp
                w_couple[use_coupling] = np.exp(
                    -0.5 * (perp_err / self.couple_sigma) ** 2)

        # 3c. MOSAIC MOTION PRIOR (belief-level, flag-gated, default OFF -> byte-identical).
        #     During steadily-locked pursuit, down-weight particles displaced from the causal
        #     track prediction beyond the per-step physiological (main-sequence) budget, at the
        #     FINE mosaic scale. This closes the alias flip at the SOURCE (the cloud position):
        #     the flip reaches a wrong ~6-row mosaic peak only via roughening / wide reseed,
        #     bypassing the dynamics velocity model — the prior penalises that displacement
        #     because no physiological one-step motion supports it. A real microsaccade rides
        #     along with pred_track (smooth ramp) so excess ~ 0 and it is NOT clipped. Fully
        #     relaxed in saccade mode (via _lock_steady p_pursuit>=0.9) and for the first
        #     mosaic_prior_reseed_hold steps after a reseed (coarse reacquisition).
        w_mosaic = 1.0
        mosaic_armed = (self.mosaic_prior
                        and self._steps_since_reseed >= self.mosaic_prior_reseed_hold
                        and max_ncc >= self.mosaic_prior_ncc_thr
                        and p_pursuit_pre >= self.lock_p_pursuit_thr)
        if mosaic_armed:
            # Anchor to the SLOW EMA track (self._track_perp), NOT a per-step prediction: the
            # flip arrives via a gradual sub-per-step roughening drift (~6 rows over ~0.7 ms) that
            # a one-step prediction would simply follow. The EMA anchor (tau ~ ms) follows slow
            # pursuit (~6 rows/s) but resists the fast migration, so migrating particles accrue
            # excess vs the held anchor and are suppressed before the cloud commits to the wrong
            # peak. Budget is the residual one-step PURSUIT motion (a real saccade drops max_ncc /
            # raises p_saccade and disarms the prior); legitimate roughening is absorbed by sigma.
            disp_max = dynamics.SIGMA_V_PURSUIT_ROWS_S * dt
            excess = np.maximum(0.0, np.abs(st.pos_perp - self._track_perp) - disp_max)
            w_mosaic = np.exp(-0.5 * (excess / self.mosaic_prior_sigma_rows) ** 2)

        # 4. reweight + normalize + ESS
        w = st.weight * w_obs * w_along * w_couple * w_mosaic
        s = w.sum()
        if not np.isfinite(s) or s <= 0.0:
            w = np.full(n, 1.0 / n)
        else:
            w = w / s
        st.weight = w
        ess = float(1.0 / np.sum(w ** 2))
        hyp_masks, hyp = _hypothesis_clusters(
            st, w, self.hypothesis_top_k if self.multi_hypothesis else 1,
            self.hypothesis_cluster_rows,
            self.hypothesis_min_mass if self.multi_hypothesis else 0.0)

        # 6. POSTERIOR point estimate: alias-robust weighted mean (within half an
        #    alias spacing of the MAP particle) so a far alias cluster cannot bias
        #    the estimate; the FULL weighted cloud is returned alongside.
        imap = int(np.argmax(w))
        mode_post = dynamics.mode_posterior(st)
        # The coarse 30' alias window cannot reject the fine ~mosaic-spacing alias: a
        # mosaic flip sits ~6 rows from the MAP, far inside the window, so the robust
        # mean averages the flipped mode in. With --mosaic-reject, narrow the window
        # toward the mosaic spacing ONLY when steadily locked (pursuit + high NCC), where
        # the temporal prior holds the MAP on the correct peak so the flipped mode is
        # rejected. In saccade mode / lock-loss the wide alias window is kept, so real
        # microsaccades (legitimate ~mosaic-spacing jumps) are NOT clipped. Mode-gated.
        alias_window = 0.5 * calib.ALIAS_SPACING_ROWS
        window = alias_window
        if self.mosaic_reject and self._lock_steady(max_ncc, float(mode_post[0])):
            window = float(self.mosaic_window_rows)
        near = np.abs(st.pos_perp - st.pos_perp[imap]) < window
        wn = w * near
        if wn.sum() <= 0.0:
            wn = w
        wn_sum = wn.sum()
        est_perp = float(np.sum(wn * st.pos_perp) / wn_sum)
        est_v_perp = float(np.sum(wn * st.vel_perp) / wn_sum)
        # along is unimodal & trusted -> plain weighted mean
        est_along = float(np.sum(w * st.pos_along))
        est_v_along = float(np.sum(w * st.vel_along))

        hyp_is_ambiguous = (self.multi_hypothesis
                            and max_ncc < self.hypothesis_ncc_thr
                            and len(hyp["mass"]) > 1
                            and float(hyp["mass"][0]) < self.hypothesis_ambiguity_mass)
        if not hyp_is_ambiguous:
            hyp = dict(
                perp=np.asarray([est_perp], dtype=np.float64),
                along=np.asarray([est_along], dtype=np.float64),
                v_perp=np.asarray([est_v_perp], dtype=np.float64),
                v_along=np.asarray([est_v_along], dtype=np.float64),
                logp=np.asarray([0.0], dtype=np.float64),
                mass=np.asarray([1.0], dtype=np.float64),
            )

        if self.mosaic_prior:                 # causal EMA anchor for next step's mosaic prior
            if mosaic_armed:                  # slow EMA -> resists fast mosaic migration
                alpha = min(dt / self.mosaic_prior_track_tau_s, 1.0)
                self._track_perp += alpha * (est_perp - self._track_perp)
            else:                             # disarmed -> free-follow (re-baseline post saccade/reseed)
                self._track_perp = est_perp

        # snapshot the cloud BEFORE resampling (this is the belief at this step)
        post = StepPosterior(
            est_perp=est_perp, est_along=est_along,
            est_v_perp=est_v_perp, est_v_along=est_v_along,
            ess=ess, mode_posterior=mode_post,
            resampled=False, reseeded=False, max_ncc=max_ncc,
            along_sigma_eff=float(sigma_along_eff),
            hyp_perp=hyp["perp"], hyp_along=hyp["along"],
            hyp_v_perp=hyp["v_perp"], hyp_v_along=hyp["v_along"],
            hyp_logp=hyp["logp"],
            pos_perp=st.pos_perp.copy(), pos_along=st.pos_along.copy(),
            weight=w.copy(),
        )

        # 5a. LOCK-LOSS DETECTION (G11): track consecutive bad-NCC steps.
        #     DISTINCT from normal ESS resampling — ESS collapses on ~97% of
        #     healthy fixation steps; using it alone would fire reseeds constantly.
        if max_ncc < self.ncc_loss_thr:
            self._low_ncc_count += 1
        else:
            self._low_ncc_count = 0

        do_reseed = (self._low_ncc_count >= self.ncc_loss_window or
                     self._low_ncc_count >= self.coast_cap)

        if do_reseed:
            # 5b. RESEED from coarse anchor (absolute perp measurement, ~43').
            #     Redraws ALL particles; resets velocities to zero and mode to
            #     pursuit.  The coarse anchor is an ABSOLUTE measurement — it
            #     repositions the cloud, it is not a veto.
            anchor = float(coarse_anchor) if coarse_anchor is not None else est_perp
            st.pos_perp[:] = rng.normal(anchor, self.reseed_perp_sigma, n)
            st.pos_along[:] = rng.normal(float(along_meas), self.reseed_along_sigma, n)
            st.vel_perp[:] = 0.0
            st.vel_along[:] = 0.0
            st.mode[:] = 0
            st.sac_phase[:] = 0.0
            st.sac_dur[:] = np.inf
            st.sac_dur_nat[:] = np.inf
            st.sac_amp[:] = 0.0
            st.sac_vpeak[:] = 0.0
            st.sac_dir_perp[:] = 0.0
            st.sac_dir_along[:] = 0.0
            st.weight[:] = 1.0 / n
            self._low_ncc_count = 0
            self.n_reseeds += 1
            self._steps_since_reseed = 0       # hold the mosaic prior off during reacquisition
            post.reseeded = True
            # Reset coupling onset arrays: after a reseed the saccade-onset
            # context is invalid; the next predict step will set them correctly
            # when particles transition 0->1.
            self._sac_perp_onset[:] = st.pos_perp
            self._sac_along_onset[:] = st.pos_along
            self._prev_mode[:] = st.mode  # all-pursuit after reseed

        elif self.resample_enabled and ess < self.ess_frac * n:
            # 5c. NORMAL RESAMPLE + ROUGHEN when ESS collapses (every healthy step).
            if hyp_is_ambiguous:
                idx = _hypothesis_resample(
                    w, hyp_masks, rng, self.hypothesis_min_particles)
            else:
                idx = _systematic_resample(w, rng)
            for f in _STATE_FIELDS:
                setattr(st, f, getattr(st, f)[idx])
            self._sac_perp_onset = self._sac_perp_onset[idx]
            self._sac_along_onset = self._sac_along_onset[idx]
            self._prev_mode = self._prev_mode[idx]
            if self.lock_gated_gain and self._lock_steady(max_ncc, mode_post[0]):
                rp, ra = self.roughen_perp_cool, self.roughen_along_cool
            elif self.lock_gated_gain:
                rp, ra = self.roughen_perp_hot, self.roughen_along_hot
            else:
                rp, ra = self.roughen_perp, self.roughen_along
            st.pos_perp = st.pos_perp + rng.normal(0.0, rp, n)
            st.pos_along = st.pos_along + rng.normal(0.0, ra, n)
            st.weight = np.full(n, 1.0 / n)
            post.resampled = True

        self.state = st
        return post


# ---------------------------------------------------------------------------
# Convenience: run over a whole stream
# ---------------------------------------------------------------------------


def run(lines, along_meas, rate: float, atlas, init_perp: float, init_along: float,
        n_particles: int = 500, perp_spread: float = None,
        along_spread: float = 4.0, line_len: Optional[int] = None,
        col_step: float = 1.0, seed: int = 0, beta: float = BETA,
        sigma_along: float = SIGMA_ALONG, ess_frac: float = ESS_FRAC,
        roughen_perp: float = ROUGHEN_PERP, roughen_along: float = ROUGHEN_ALONG,
        hp_sigma: float = HP_SIGMA,
        ncc_loss_thr: float = NCC_LOCK_LOSS_THR,
        ncc_loss_window: int = NCC_LOSS_WINDOW,
        coast_cap: int = COAST_CAP,
        reseed_perp_sigma: float = RESEED_PERP_SIGMA,
        reseed_along_sigma: float = RESEED_ALONG_SIGMA,
        couple_sigma: float = COUPLE_SIGMA_ROWS,
        couple_dir_along_min: float = COUPLE_DIR_ALONG_MIN,
        coarse_anchor: Optional[np.ndarray] = None,
        likelihood: str = "physics",
        learned_head: "losses_mod.LearnedPerpLikelihood | None" = None,
        resample_enabled: bool = True,
        lock_gated_gain: bool = False,
        along_quality: Optional[np.ndarray] = None,
        quality_scaled_along: bool = False,
        along_quality_floor: float = ALONG_QUALITY_FLOOR,
        along_sigma_max: float = ALONG_SIGMA_MAX,
        along_quality_model: AlongQualityModel | None = None,
        multi_hypothesis: bool = False,
        lag_ms: float = HYPOTHESIS_LAG_MS,
        hypothesis_top_k: int = HYPOTHESIS_TOP_K,
        hypothesis_cluster_rows: float = HYPOTHESIS_CLUSTER_ROWS,
        hypothesis_min_mass: float = HYPOTHESIS_MIN_MASS,
        hypothesis_ambiguity_mass: float = HYPOTHESIS_AMBIGUITY_MASS,
        hypothesis_ncc_thr: float = HYPOTHESIS_NCC_THR,
        hypothesis_min_particles: int = HYPOTHESIS_MIN_PARTICLES,
        hypothesis_transition_sigma_rows: float = HYPOTHESIS_TRANSITION_SIGMA_ROWS,
        hypothesis_obs_weight: float = HYPOTHESIS_OBS_WEIGHT,
        hypothesis_velocity_cost: float = HYPOTHESIS_VEL_COST,
        hypothesis_velocity_sigma_deg_s: float = HYPOTHESIS_VEL_SIGMA_DEG_S,
        hypothesis_acceleration_cost: float = HYPOTHESIS_ACCEL_COST,
        hypothesis_acceleration_sigma_deg_s2: float = HYPOTHESIS_ACCEL_SIGMA_DEG_S2,
        slew_gate: bool = False,
        slew_max_deg_s: float = SLEW_GATE_MAX_DEG_S,
        slew_gate_cost: float = SLEW_GATE_COST,
        slew_gate_saccade_p: float = SLEW_GATE_SACCADE_P,
        hypothesis_blend_immediate: bool = HYPOTHESIS_BLEND_IMMEDIATE,
        hypothesis_blend_delta_rows: float = HYPOTHESIS_BLEND_DELTA_ROWS,
        hypothesis_blend_alpha: float = HYPOTHESIS_BLEND_ALPHA,
        hypothesis_blend_saccade_p: float = HYPOTHESIS_BLEND_SACCADE_P
        ) -> FilterResult:
    """Run the particle filter over a stream of observed lines.

    Parameters
    ----------
    lines      : (T, L) array of observed lines, OR a synth_stream.SyntheticStream
                 (its ``.lines`` / ``.rate`` / ``.line_len`` / ``.col_step`` are used).
    along_meas : (T,) trusted along-position measurement (atlas cols). For a
                 SyntheticStream this may be None to use the true along_cols.
    rate       : effective stream rate (Hz); dt = 1/rate. Ignored if ``lines`` is a
                 SyntheticStream (its rate is used).
    atlas      : the frozen physics atlas (data.load_atlas() or its ref_map).
    init_perp, init_along : initial gaze lock (atlas rows / cols).
    perp_spread: initial perp scatter (rows); defaults to one alias spacing so the
                 multimodal machinery is exercised. ``along_spread`` likewise (cols).
    coarse_anchor : (T,) per-step coarse perp estimate (atlas rows) for G11 reseeding.
                 For SyntheticStream pass None to auto-generate as
                 true_perp + N(0, COARSE_SIGMA_ROWS) (simulating the real coarse channel).
                 For real data use data.coarse_perp(...).
    likelihood : 'physics' (default, the G10-G13 fine-NCC observation model) or
                 'learned' (the G14 blur-aware calibrated head). 'learned' requires
                 ``learned_head`` (train.load_head()); it changes ONLY the perp
                 observation weight, leaving the trusted-along / coupling / reseed
                 machinery untouched. The default keeps every existing test intact.
    learned_head : a trained losses.LearnedPerpLikelihood (required iff likelihood=='learned').

    Returns a :class:`FilterResult` with per-step estimates, ESS, reseed events,
    and the full per-step posteriors (enough to inspect multimodality).
    """
    # SyntheticStream overload
    if hasattr(lines, "lines") and hasattr(lines, "rate"):
        stream = lines
        line_arr = np.asarray(stream.lines, dtype=np.float64)
        rate = float(stream.rate)
        if line_len is None:
            line_len = int(stream.line_len)
        col_step = float(stream.col_step)
        if along_meas is None:
            along_meas = np.asarray(stream.trajectory.along_cols, dtype=np.float64)
        # Auto-generate synthetic coarse anchor if not provided
        if coarse_anchor is None:
            _rng_ca = np.random.default_rng(seed + 999)
            true_perp = np.asarray(stream.trajectory.perp_rows, dtype=np.float64)
            coarse_anchor = true_perp + _rng_ca.normal(0.0, COARSE_SIGMA_ROWS,
                                                        len(true_perp))
    else:
        line_arr = np.asarray(lines, dtype=np.float64)
        if line_len is None:
            line_len = line_arr.shape[1]

    along_meas = np.asarray(along_meas, dtype=np.float64)
    T = line_arr.shape[0]
    if perp_spread is None:
        perp_spread = calib.ALIAS_SPACING_ROWS
    dt = 1.0 / float(rate)

    # coarse_anchor must be (T,) or None (reseed disabled)
    ca_arr: Optional[np.ndarray] = (
        np.asarray(coarse_anchor, dtype=np.float64) if coarse_anchor is not None else None
    )
    aq_arr: Optional[np.ndarray] = (
        np.asarray(along_quality, dtype=np.float64) if along_quality is not None else None
    )

    rng = np.random.default_rng(seed)
    state = init_filter(n_particles, init_perp, init_along, perp_spread,
                        along_spread, rng=rng)
    pf = ParticleFilter(state, atlas, line_len, col_step=col_step, beta=beta,
                        sigma_along=sigma_along, ess_frac=ess_frac,
                        roughen_perp=roughen_perp, roughen_along=roughen_along,
                        hp_sigma=hp_sigma, ncc_loss_thr=ncc_loss_thr,
                        ncc_loss_window=ncc_loss_window, coast_cap=coast_cap,
                        reseed_perp_sigma=reseed_perp_sigma,
                        reseed_along_sigma=reseed_along_sigma,
                        couple_sigma=couple_sigma,
                        couple_dir_along_min=couple_dir_along_min,
                        likelihood=likelihood, learned_head=learned_head,
                        resample_enabled=resample_enabled,
                        lock_gated_gain=lock_gated_gain,
                        quality_scaled_along=quality_scaled_along,
                        along_quality_floor=along_quality_floor,
                        along_sigma_max=along_sigma_max,
                        along_quality_model=along_quality_model,
                        multi_hypothesis=multi_hypothesis,
                        hypothesis_top_k=hypothesis_top_k,
                        hypothesis_cluster_rows=hypothesis_cluster_rows,
                        hypothesis_min_mass=hypothesis_min_mass,
                        hypothesis_ambiguity_mass=hypothesis_ambiguity_mass,
                        hypothesis_ncc_thr=hypothesis_ncc_thr,
                        hypothesis_min_particles=hypothesis_min_particles)

    est_perp = np.empty(T)
    est_along = np.empty(T)
    est_v_perp = np.empty(T)
    est_v_along = np.empty(T)
    ess = np.empty(T)
    p_sacc = np.empty(T)
    resampled = np.zeros(T, dtype=bool)
    reseeded = np.zeros(T, dtype=bool)
    max_ncc_arr = np.empty(T)
    along_sigma_eff = np.empty(T)
    hyp_count = np.empty(T, dtype=np.int32)
    posteriors: list[StepPosterior] = []

    for t in range(T):
        ca_t = float(ca_arr[t]) if ca_arr is not None else None
        aq_t = float(aq_arr[t]) if aq_arr is not None else None
        post = pf.step(line_arr[t], float(along_meas[t]), dt, rng,
                       coarse_anchor=ca_t, along_quality=aq_t)
        est_perp[t] = post.est_perp
        est_along[t] = post.est_along
        est_v_perp[t] = post.est_v_perp
        est_v_along[t] = post.est_v_along
        ess[t] = post.ess
        p_sacc[t] = post.mode_posterior[1]
        resampled[t] = post.resampled
        reseeded[t] = post.reseeded
        max_ncc_arr[t] = post.max_ncc
        along_sigma_eff[t] = post.along_sigma_eff
        hyp_count[t] = len(post.hyp_perp)
        posteriors.append(post)

    if multi_hypothesis:
        est_perp, est_along, est_v_perp, est_v_along = _fixed_lag_resolve(
            posteriors, float(rate), lag_ms, hypothesis_transition_sigma_rows,
            hypothesis_obs_weight,
            velocity_cost=hypothesis_velocity_cost,
            velocity_sigma_deg_s=hypothesis_velocity_sigma_deg_s,
            acceleration_cost=hypothesis_acceleration_cost,
            acceleration_sigma_deg_s2=hypothesis_acceleration_sigma_deg_s2,
            slew_gate=slew_gate,
            slew_max_deg_s=slew_max_deg_s, slew_gate_cost=slew_gate_cost,
            slew_gate_saccade_p=slew_gate_saccade_p,
            blend_immediate=hypothesis_blend_immediate,
            blend_delta_rows=hypothesis_blend_delta_rows,
            blend_alpha=hypothesis_blend_alpha,
            blend_saccade_p=hypothesis_blend_saccade_p)

    return FilterResult(
        est_perp=est_perp, est_along=est_along, est_v_perp=est_v_perp,
        est_v_along=est_v_along, ess=ess, p_saccade=p_sacc,
        resampled=resampled, reseeded=reseeded, max_ncc=max_ncc_arr,
        along_sigma_eff=along_sigma_eff, hyp_count=hyp_count,
        n_reseeds=pf.n_reseeds, posteriors=posteriors, rate=float(rate),
    )
