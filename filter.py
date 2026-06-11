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
    n_reseeds: int             # total reseed events over the run
    posteriors: list[StepPosterior]  # per-step full weighted clouds
    rate: float


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
                 learned_head: "losses_mod.LearnedPerpLikelihood | None" = None):
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

    @property
    def n(self) -> int:
        return self.state.n

    def step(self, line, along_meas: float, dt: float,
             rng: np.random.Generator,
             coarse_anchor: Optional[float] = None) -> StepPosterior:
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

        # Track mode before predict so we can detect 0->1 transitions below.
        self._prev_mode[:] = st.mode

        # 1. PREDICT — IMM prior (dynamics.py; NOT reimplemented here)
        st = dynamics.predict(st, dt, rng)

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
            w_obs = np.exp(self.beta * (ncc - max_ncc))
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

        # 3. TRUSTED ALONG measurement (low variance) on the along POSITION
        w_along = np.exp(-0.5 * ((st.pos_along - float(along_meas)) /
                                 self.sigma_along) ** 2)

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

        # 4. reweight + normalize + ESS
        w = st.weight * w_obs * w_along * w_couple
        s = w.sum()
        if not np.isfinite(s) or s <= 0.0:
            w = np.full(n, 1.0 / n)
        else:
            w = w / s
        st.weight = w
        ess = float(1.0 / np.sum(w ** 2))

        # 6. POSTERIOR point estimate: alias-robust weighted mean (within half an
        #    alias spacing of the MAP particle) so a far alias cluster cannot bias
        #    the estimate; the FULL weighted cloud is returned alongside.
        imap = int(np.argmax(w))
        alias_window = 0.5 * calib.ALIAS_SPACING_ROWS
        near = np.abs(st.pos_perp - st.pos_perp[imap]) < alias_window
        wn = w * near
        if wn.sum() <= 0.0:
            wn = w
        wn_sum = wn.sum()
        est_perp = float(np.sum(wn * st.pos_perp) / wn_sum)
        est_v_perp = float(np.sum(wn * st.vel_perp) / wn_sum)
        # along is unimodal & trusted -> plain weighted mean
        est_along = float(np.sum(w * st.pos_along))
        est_v_along = float(np.sum(w * st.vel_along))
        mode_post = dynamics.mode_posterior(st)

        # snapshot the cloud BEFORE resampling (this is the belief at this step)
        post = StepPosterior(
            est_perp=est_perp, est_along=est_along,
            est_v_perp=est_v_perp, est_v_along=est_v_along,
            ess=ess, mode_posterior=mode_post,
            resampled=False, reseeded=False, max_ncc=max_ncc,
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
            post.reseeded = True
            # Reset coupling onset arrays: after a reseed the saccade-onset
            # context is invalid; the next predict step will set them correctly
            # when particles transition 0->1.
            self._sac_perp_onset[:] = st.pos_perp
            self._sac_along_onset[:] = st.pos_along
            self._prev_mode[:] = st.mode  # all-pursuit after reseed

        elif ess < self.ess_frac * n:
            # 5c. NORMAL RESAMPLE + ROUGHEN when ESS collapses (every healthy step).
            #     This is the anti-impoverishment mechanism for the sharp fine peak;
            #     it must NOT be confused with the G11 reseed.
            idx = _systematic_resample(w, rng)
            for f in _STATE_FIELDS:
                setattr(st, f, getattr(st, f)[idx])
            # Carry coupling onset arrays along with the resampled particles so
            # saccade-mode particles keep their correct onset after resampling.
            self._sac_perp_onset = self._sac_perp_onset[idx]
            self._sac_along_onset = self._sac_along_onset[idx]
            self._prev_mode = self._prev_mode[idx]
            # roughening: small position jitter so the resampled cloud keeps
            # exploring the razor-sharp fine peak (anti-impoverishment).
            st.pos_perp = st.pos_perp + rng.normal(0.0, self.roughen_perp, n)
            st.pos_along = st.pos_along + rng.normal(0.0, self.roughen_along, n)
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
        learned_head: "losses_mod.LearnedPerpLikelihood | None" = None) -> FilterResult:
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
                        likelihood=likelihood, learned_head=learned_head)

    est_perp = np.empty(T)
    est_along = np.empty(T)
    est_v_perp = np.empty(T)
    est_v_along = np.empty(T)
    ess = np.empty(T)
    p_sacc = np.empty(T)
    resampled = np.zeros(T, dtype=bool)
    reseeded = np.zeros(T, dtype=bool)
    max_ncc_arr = np.empty(T)
    posteriors: list[StepPosterior] = []

    for t in range(T):
        ca_t = float(ca_arr[t]) if ca_arr is not None else None
        post = pf.step(line_arr[t], float(along_meas[t]), dt, rng,
                       coarse_anchor=ca_t)
        est_perp[t] = post.est_perp
        est_along[t] = post.est_along
        est_v_perp[t] = post.est_v_perp
        est_v_along[t] = post.est_v_along
        ess[t] = post.ess
        p_sacc[t] = post.mode_posterior[1]
        resampled[t] = post.resampled
        reseeded[t] = post.reseeded
        max_ncc_arr[t] = post.max_ncc
        posteriors.append(post)

    return FilterResult(
        est_perp=est_perp, est_along=est_along, est_v_perp=est_v_perp,
        est_v_along=est_v_along, ess=ess, p_saccade=p_sacc,
        resampled=resampled, reseeded=reseeded, max_ncc=max_ncc_arr,
        n_reseeds=pf.n_reseeds, posteriors=posteriors, rate=float(rate),
    )
