"""dynamics.py — IMM (interacting-multiple-model) motion prior (GOALS.md G9).

The differentiable particle filter (G10) needs a *dynamics prior* that can
propagate particle states one step at a time and, run open-loop, reproduce real
oculomotor statistics. PLAN.md is emphatic about WHICH prior: an IMM with a
PURSUIT mode and a SACCADE mode whose kinematics follow the measured **main
sequence** (peak-velocity-vs-amplitude). A generic large-process-noise
"smoothness" prior is FORBIDDEN — it rewards envelope-riding (the documented
perp r=0.86-vs-stimulus / 0.11-vs-real-motion failure). There is therefore NO
mode in this module that is merely "large Gaussian process noise":

  * PURSUIT/FIXATION (mode 0): near-constant-velocity with an Ornstein-Uhlenbeck
    (band-limited, correlated) velocity at a SMALL, explicitly capped
    acceleration. Drift + smooth pursuit live here. This is deliberately the
    OPPOSITE of a large-sigma random walk — the OU velocity is low-pass with a
    correlation time TAU_PURSUIT_S and a stationary std of a few rows/s, and the
    per-step acceleration is hard-capped at ACCEL_CAP_ROWS_S2.

  * SACCADE (mode 1): a BALLISTIC velocity pulse. On saccade onset a target
    amplitude A (heavy-tailed small, ~2-40') and a direction are drawn; the pulse
    peak velocity is the main sequence V_peak(A) = Vmax * (1 - exp(-A / A0)) and
    the pulse is a min-jerk velocity profile that integrates to exactly A over
    its duration D = (min-jerk peak/mean) * A / V_peak. The peak velocity is tied
    to amplitude BY CONSTRUCTION; there is no free process-noise sigma anywhere in
    the saccade dynamics. See ``_assert_main_sequence_only``.

Shared state schema (MUST match traj_gen.py and the G10 filter — do not change):
a particle state is (perp_rows, along_cols, v_perp_rows_s, v_along_cols_s, mode):
  - perp / along : position in ATLAS units (perp ~125 rows/deg via
    calib.ROWS_PER_DEG; along columns isotropic-assumed, same rows/deg scale).
  - v_perp / v_along : velocity in rows/s and cols/s.
  - mode : int8 {0 = pursuit/fixation, 1 = saccade}.

A particle set is the :class:`ParticleState` dataclass with arrays of shape (N,):
``pos_perp, pos_along, vel_perp, vel_along`` (float64), ``mode`` (int8), and an
optional ``weight`` (N,). Saccade bookkeeping (phase / duration / commanded
amplitude+peak / direction) travels alongside the public state so a mid-saccade
pulse survives across ``predict`` calls.

Public API:
  - :func:`predict(state, dt, rng) -> ParticleState` — advance every particle by
    dt and resample/propagate the mode per the IMM transition matrix.
  - :func:`mode_posterior(state) -> (p_pursuit, p_saccade)` — weight-weighted
    mode fractions (sum to 1).
  - :func:`rollout(...)` — open-loop pure-prior rollout (no observations) that
    accumulates a single-particle trajectory plus the per-saccade ground-truth
    (amplitude, peak velocity) main sequence the tests measure against.
  - :func:`transition_matrix(dt)` — the per-dt IMM mode transition matrix.

Coordinate convention matches data.py / calib.py: PERP = atlas row axis, ALONG =
atlas column axis. Saccades are spatially straight, so a single amplitude A is
split between the two axes by the drawn direction (cos/sin); peak velocity along
each axis shares the same V_peak(A) main sequence (isotropic rows/deg scale).

Run ``python dynamics.py --report`` for the measured rollout statistics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import calib
import data

# ---------------------------------------------------------------------------
# Measured / tuned constants (frozen from the G9 investigation; see module
# docstring). Tolerances in tests/test_dynamics.py are set from the rollout
# statistics these produce, not the other way around.
# ---------------------------------------------------------------------------

# spatial scale (perp axis is the calibrated one; along assumed isotropic)
ROWS_PER_DEG: float = calib.ROWS_PER_DEG          # 124.64 rows/deg
ARCMIN_PER_ROW: float = calib.ARCMIN_PER_ROW      # 0.4814' per row

# saccade main sequence: V_peak(A) = VMAX * (1 - exp(-A / A0))
VMAX_ROWS_S: float = data.UNITS.saccade_vmax_rows_per_s   # 103000 rows/s ~ 826 deg/s asymptote
A0_ARCMIN: float = 6.5                             # main-sequence knee amplitude
A0_ROWS: float = A0_ARCMIN / ARCMIN_PER_ROW        # ~13.5 rows
# A0 is tuned so the rollout's small-amplitude log-log peak-vel-vs-amplitude
# slope lands near the measured along-channel main sequence (~0.37; see
# calib.main_seq_slope). Over the sampled [~2, 40]' band the saturating form
# gives an effective slope ~0.4.

# saccade amplitude draw: heavy-tailed small (lognormal in arcmin), ~2-40'
AMP_MED_ARCMIN: float = 11.0
AMP_LOGSIGMA: float = 0.6
AMP_MIN_ARCMIN: float = 1.5
AMP_MAX_ARCMIN: float = 60.0

# min-jerk velocity profile: position p(s) = A*(10 s^3 - 15 s^4 + 6 s^5),
# velocity dp/ds peaks at s=0.5 with peak/mean ratio 1.875. The pulse's NATURAL
# duration is set so its peak equals the main-sequence V_peak and it integrates
# to A:  D_nat = MINJERK_PEAK_OVER_MEAN * A / V_peak(A).
MINJERK_PEAK_OVER_MEAN: float = 1.875

# For these arcmin-scale microsaccades D_nat is sub-millisecond (a 12'/0.2deg
# saccade at ~700 deg/s lasts ~0.5 ms — confirmed on the 12 kHz along channel,
# durations 0.2-0.3 ms). To keep each saccade an EXTRACTABLE mode==1 run at the
# rollout rates the tests use (a few hundred Hz to a few kHz), the saccade
# *segment* is held in mode 1 for at least SACCADE_SEG_MIN_S; the ballistic
# min-jerk pulse plays out over its natural sub-ms duration at the segment start
# (peak = V_peak(A), integral = A), then the eye is stationary for the remainder.
# This does NOT alter the main sequence: peak velocity and amplitude are exactly
# V_peak(A) and A regardless of the segment length.
SACCADE_SEG_MIN_S: float = 0.006                   # min mode==1 segment (~6 ms)

# IMM mode-transition hazards (per second). pursuit->saccade gives ~2 saccades/s;
# saccade->pursuit is not a free hazard — a saccade ends when its ballistic pulse
# completes (duration D, set by the main sequence). H_SACC_TO_PURSUIT is the
# fallback hazard used only to convert to a transition-matrix entry for
# :func:`transition_matrix` (mean saccade duration ~ a few tens of ms).
HAZARD_PURSUIT_TO_SACCADE: float = 2.0
_MEAN_SACCADE_DUR_S: float = 0.045                 # ~45 ms (matrix entry only)
HAZARD_SACCADE_TO_PURSUIT: float = 1.0 / _MEAN_SACCADE_DUR_S

# pursuit OU velocity (band-limited, correlated — NOT a large random walk)
TAU_PURSUIT_S: float = 0.15                        # velocity correlation time
SIGMA_V_PURSUIT_ROWS_S: float = 6.0                # stationary velocity std (~3'/s)
ACCEL_CAP_ROWS_S2: float = 4000.0                  # hard per-step acceleration cap

# minimum substeps used to integrate a saccade pulse within one predict() dt so
# the ballistic amplitude integrates accurately even when D < dt (sub-ms pulses).
_SACCADE_SUBSTEPS: int = 64


# ---------------------------------------------------------------------------
# Particle state
# ---------------------------------------------------------------------------


@dataclass
class ParticleState:
    """A set of N particles over the shared (perp, along, v_perp, v_along, mode)
    state, plus the saccade bookkeeping a ballistic pulse needs to persist across
    predict() steps.

    Public schema (matches traj_gen.py / G10 filter):
      pos_perp, pos_along : (N,) float64 — position, ATLAS units (perp rows /
        along cols; ~125 rows/deg via calib.ROWS_PER_DEG).
      vel_perp, vel_along : (N,) float64 — velocity, rows/s and cols/s.
      mode                : (N,) int8 — 0 = pursuit/fixation, 1 = saccade.
      weight              : (N,) float64 — particle weights (default uniform;
        kept untouched by the prior, the G10 update reweights them).

    Saccade bookkeeping (internal to the prior; carried alongside the state):
      sac_phase  : (N,) float64 — elapsed segment time (s); valid where mode==1.
      sac_dur    : (N,) float64 — segment duration (s); mode==1 holds this long
        (>= SACCADE_SEG_MIN_S) so the saccade is an extractable run.
      sac_dur_nat: (N,) float64 — natural min-jerk pulse duration D_nat (s); the
        ballistic pulse plays out over [0, D_nat] at the segment start.
      sac_amp    : (N,) float64 — commanded amplitude A (rows, total, > 0).
      sac_vpeak  : (N,) float64 — commanded main-sequence peak velocity (rows/s).
      sac_dir_perp, sac_dir_along : (N,) float64 — unit direction (cos/sin).
    """

    pos_perp: np.ndarray
    pos_along: np.ndarray
    vel_perp: np.ndarray
    vel_along: np.ndarray
    mode: np.ndarray
    weight: np.ndarray
    # saccade bookkeeping
    sac_phase: np.ndarray
    sac_dur: np.ndarray
    sac_dur_nat: np.ndarray
    sac_amp: np.ndarray
    sac_vpeak: np.ndarray
    sac_dir_perp: np.ndarray
    sac_dir_along: np.ndarray

    @property
    def n(self) -> int:
        return self.pos_perp.shape[0]

    def copy(self) -> "ParticleState":
        return ParticleState(
            self.pos_perp.copy(), self.pos_along.copy(),
            self.vel_perp.copy(), self.vel_along.copy(),
            self.mode.copy(), self.weight.copy(),
            self.sac_phase.copy(), self.sac_dur.copy(), self.sac_dur_nat.copy(),
            self.sac_amp.copy(), self.sac_vpeak.copy(),
            self.sac_dir_perp.copy(), self.sac_dir_along.copy(),
        )


def init_state(n: int, pos_perp: float = 0.0, pos_along: float = 0.0,
               vel_perp: float = 0.0, vel_along: float = 0.0,
               mode: int = 0) -> ParticleState:
    """Construct N identical particles in pursuit (default) at a given position.

    All saccade bookkeeping starts inactive (phase=0, dur=inf so no pulse is mid
    flight). Weights are uniform (sum to 1).
    """
    z = np.zeros(n, dtype=np.float64)
    return ParticleState(
        pos_perp=z + float(pos_perp),
        pos_along=z + float(pos_along),
        vel_perp=z + float(vel_perp),
        vel_along=z + float(vel_along),
        mode=np.full(n, int(mode), dtype=np.int8),
        weight=np.full(n, 1.0 / n, dtype=np.float64),
        sac_phase=z.copy(),
        sac_dur=np.full(n, np.inf, dtype=np.float64),
        sac_dur_nat=np.full(n, np.inf, dtype=np.float64),
        sac_amp=z.copy(),
        sac_vpeak=z.copy(),
        sac_dir_perp=z.copy(),
        sac_dir_along=z.copy(),
    )


# ---------------------------------------------------------------------------
# Main-sequence kinematics (NOT generic smoothness)
# ---------------------------------------------------------------------------


def v_peak(amplitude_rows: np.ndarray | float) -> np.ndarray | float:
    """Main-sequence peak velocity V_peak(A) = VMAX * (1 - exp(-A / A0)).

    Peak velocity is a saturating function of amplitude: small saccades are
    slow, large saccades approach VMAX. This is the ONLY thing that sets saccade
    speed — there is no process-noise term. Units: rows/s for A in rows.
    """
    A = np.asarray(amplitude_rows, dtype=np.float64)
    return VMAX_ROWS_S * (1.0 - np.exp(-A / A0_ROWS))


def saccade_duration(amplitude_rows: np.ndarray | float) -> np.ndarray | float:
    """Min-jerk pulse duration D = (peak/mean) * A / V_peak(A) (seconds).

    Ties duration to the main sequence so the pulse peak equals V_peak(A) and the
    pulse integrates to exactly A.
    """
    A = np.asarray(amplitude_rows, dtype=np.float64)
    return MINJERK_PEAK_OVER_MEAN * A / v_peak(A)


def _minjerk_velocity(s: np.ndarray, amplitude: np.ndarray, dur: np.ndarray
                      ) -> np.ndarray:
    """Min-jerk velocity profile, |v| as a function of normalized phase s in [0,1].

    p(s) = A*(10 s^3 - 15 s^4 + 6 s^5); v = dp/dt = (A/D)*(30 s^2 - 60 s^3 + 30 s^4).
    Integrates to A over the pulse and peaks at s=0.5 with V_peak = 1.875*A/D.
    """
    s = np.clip(s, 0.0, 1.0)
    shape = 30.0 * s ** 2 - 60.0 * s ** 3 + 30.0 * s ** 4
    return (amplitude / dur) * shape


def transition_matrix(dt: float) -> np.ndarray:
    """Per-dt IMM mode-transition matrix T, T[i, j] = P(mode j at t+dt | mode i).

    Hazards are converted to per-dt probabilities with p = 1 - exp(-h*dt).
    Row 0 = pursuit, row 1 = saccade. (The actual saccade->pursuit transition in
    predict() is event-driven: a saccade ends when its ballistic pulse completes.
    This matrix exposes the equivalent mean-duration hazard for the G10 filter's
    mode prior.) Rows sum to 1.
    """
    p_ps = 1.0 - math.exp(-HAZARD_PURSUIT_TO_SACCADE * dt)
    p_sp = 1.0 - math.exp(-HAZARD_SACCADE_TO_PURSUIT * dt)
    return np.array([[1.0 - p_ps, p_ps],
                     [p_sp, 1.0 - p_sp]], dtype=np.float64)


def _assert_main_sequence_only() -> None:
    """Guard: saccade speed must come from the main sequence, never process noise.

    There is intentionally no sigma / random-walk parameter governing saccade
    velocity in this module; saccade peak velocity is V_peak(A) by construction
    and the pulse is deterministic given (A, direction). This function documents
    and checks that invariant: V_peak is monotone increasing in amplitude and
    saturates toward VMAX (a generic-smoothness mode would have velocity
    independent of, or linear without saturation in, amplitude).
    """
    A = np.array([1.0, 10.0, 100.0, 1000.0]) / ARCMIN_PER_ROW  # 1'..1000' in rows
    V = np.asarray(v_peak(A))
    assert np.all(np.diff(V) >= 0), "main sequence must be monotone non-decreasing"
    assert V[-1] <= VMAX_ROWS_S and V[-1] > 0.9 * VMAX_ROWS_S, \
        "large saccades must saturate toward VMAX"
    # small-amplitude regime is sub-VMAX (slow) — saccades are NOT a constant
    # large velocity, they obey the amplitude-dependent law
    assert V[0] < 0.5 * VMAX_ROWS_S, "small saccades must be slow (main sequence)"


_assert_main_sequence_only()


# ---------------------------------------------------------------------------
# IMM predict
# ---------------------------------------------------------------------------


def _draw_saccade(rng: np.random.Generator, n: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray]:
    """Draw n new saccades: (amplitude_rows, vpeak, dur_nat, dur_seg, dir_perp,
    dir_along).

    Amplitude is heavy-tailed small (lognormal in arcmin, clipped to [2, 40]'-ish);
    direction is uniform on the circle (saccades are spatially straight, so a
    single amplitude is split across the two axes by cos/sin). ``dur_nat`` is the
    natural min-jerk pulse duration (sub-ms; sets the peak velocity); ``dur_seg``
    is the mode==1 segment duration (>= SACCADE_SEG_MIN_S; keeps the saccade an
    extractable run).
    """
    amp_arc = np.exp(rng.normal(math.log(AMP_MED_ARCMIN), AMP_LOGSIGMA, size=n))
    amp_arc = np.clip(amp_arc, AMP_MIN_ARCMIN, AMP_MAX_ARCMIN)
    amp_rows = amp_arc / ARCMIN_PER_ROW
    vpk = np.asarray(v_peak(amp_rows), dtype=np.float64)
    dur_nat = MINJERK_PEAK_OVER_MEAN * amp_rows / vpk
    dur_seg = np.maximum(dur_nat, SACCADE_SEG_MIN_S)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    return amp_rows, vpk, dur_nat, dur_seg, np.cos(theta), np.sin(theta)


def predict(state: ParticleState, dt: float,
            rng: Optional[np.random.Generator] = None) -> ParticleState:
    """Advance every particle by ``dt`` under the IMM prior, returning a new
    :class:`ParticleState`.

    Pursuit particles (mode 0): OU velocity update (band-limited, capped accel)
    then ``pos += vel*dt``; each may transition to saccade with per-dt
    probability ``1 - exp(-HAZARD_PURSUIT_TO_SACCADE*dt)``, on which a fresh
    main-sequence saccade is drawn.

    Saccade particles (mode 1): the ballistic min-jerk pulse is integrated across
    ``dt`` (sub-stepped so the amplitude integrates accurately even for sub-ms
    pulses); the velocity follows V_peak(A) and the position accumulates ~A. When
    the pulse phase reaches its duration the particle returns to pursuit with zero
    velocity (event-driven, NOT a hazard random walk).

    ``rng`` defaults to a fresh default_rng(); pass a seeded generator for
    determinism. Weights are propagated unchanged (the G10 update reweights).
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = float(dt)
    out = state.copy()
    n = out.n

    pursuit = out.mode == 0
    saccade = out.mode == 1

    # --- pursuit / fixation: OU velocity (correlated, low-accel, capped) ---
    if pursuit.any():
        idx = np.flatnonzero(pursuit)
        # OU: dv = -(v/tau) dt + sigma*sqrt(2/tau)*sqrt(dt)*N(0,1). Stationary
        # std -> SIGMA_V_PURSUIT. This is band-limited, NOT a large random walk.
        drive = SIGMA_V_PURSUIT_ROWS_S * math.sqrt(2.0 / TAU_PURSUIT_S) * math.sqrt(dt)
        for vel in (out.vel_perp, out.vel_along):
            dv = (-vel[idx] / TAU_PURSUIT_S) * dt + drive * rng.standard_normal(idx.size)
            # hard acceleration cap (explicit — the anti-runaway guarantee)
            dv = np.clip(dv, -ACCEL_CAP_ROWS_S2 * dt, ACCEL_CAP_ROWS_S2 * dt)
            vel[idx] += dv
        out.pos_perp[idx] += out.vel_perp[idx] * dt
        out.pos_along[idx] += out.vel_along[idx] * dt

        # pursuit -> saccade transitions
        p_ps = 1.0 - math.exp(-HAZARD_PURSUIT_TO_SACCADE * dt)
        fire = idx[rng.random(idx.size) < p_ps]
        if fire.size:
            amp, vpk, dur_nat, dur_seg, dperp, dalong = _draw_saccade(rng, fire.size)
            out.mode[fire] = 1
            out.sac_phase[fire] = 0.0
            out.sac_dur[fire] = dur_seg
            out.sac_dur_nat[fire] = dur_nat
            out.sac_amp[fire] = amp
            out.sac_vpeak[fire] = vpk
            out.sac_dir_perp[fire] = dperp
            out.sac_dir_along[fire] = dalong
            saccade = out.mode == 1  # include the freshly fired particles

    # --- saccade: ballistic min-jerk pulse over its natural (sub-ms) duration,
    #     held in mode 1 for the segment so it is an extractable run ---
    if saccade.any():
        idx = np.flatnonzero(saccade)
        amp = out.sac_amp[idx]
        dur_seg = out.sac_dur[idx]
        dur_nat = out.sac_dur_nat[idx]
        phase = out.sac_phase[idx]            # segment time at step start
        dperp = out.sac_dir_perp[idx]
        dalong = out.sac_dir_along[idx]

        # integrate the min-jerk pulse across this dt with fine substeps so the
        # sub-ms pulse integrates to ~A accurately and its peak (= V_peak) is
        # captured. The pulse is non-zero only over [0, dur_nat]; afterwards the
        # eye is stationary for the rest of the segment.
        sub = max(_SACCADE_SUBSTEPS,
                  int(math.ceil(dt / (np.min(dur_nat) / _SACCADE_SUBSTEPS)))
                  if np.min(dur_nat) > 0 else _SACCADE_SUBSTEPS)
        h = dt / sub
        t = phase.copy()
        peak_mag = np.zeros(idx.size)         # max |v| reached within this step
        for _ in range(sub):
            in_pulse = t < dur_nat
            if in_pulse.any():
                s = np.where(in_pulse, t / dur_nat, 0.0)
                vmag = np.where(in_pulse, _minjerk_velocity(s, amp, dur_nat), 0.0)
                out.pos_perp[idx] += dperp * vmag * h
                out.pos_along[idx] += dalong * vmag * h
                peak_mag = np.maximum(peak_mag, np.abs(vmag))
            t = t + h
        out.sac_phase[idx] = t
        # report the representative saccade velocity this step: the peak speed
        # reached within the step (signed by direction) so max|v| over a mode==1
        # run equals the main-sequence V_peak(A).
        out.vel_perp[idx] = dperp * peak_mag
        out.vel_along[idx] = dalong * peak_mag

        # segments that completed return to pursuit (event-driven; the segment is
        # held for SACCADE_SEG_MIN_S even though the ballistic pulse finished
        # earlier, so the saccade is a multi-sample extractable run)
        done = idx[t >= dur_seg]
        if done.size:
            out.mode[done] = 0
            out.vel_perp[done] = 0.0
            out.vel_along[done] = 0.0
            out.sac_dur[done] = np.inf
            out.sac_dur_nat[done] = np.inf
            out.sac_phase[done] = 0.0

    return out


def mode_posterior(state: ParticleState) -> tuple[float, float]:
    """Weight-weighted mode posterior ``(p_pursuit, p_saccade)``; sums to 1.

    Uses the particle weights (uniform by default). If all weights are zero a
    uniform fallback is used so the posterior is always a valid distribution.
    """
    w = state.weight.astype(np.float64)
    wsum = w.sum()
    if not np.isfinite(wsum) or wsum <= 0:
        w = np.ones_like(w)
        wsum = w.sum()
    w = w / wsum
    p_sacc = float(w[state.mode == 1].sum())
    p_purs = float(w[state.mode == 0].sum())
    # guard against any out-of-set modes (there are none by construction)
    total = p_purs + p_sacc
    if total <= 0:
        return 1.0, 0.0
    return p_purs / total, p_sacc / total


# ---------------------------------------------------------------------------
# Open-loop pure-prior rollout (no observations)
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    """Result of an open-loop pure-prior rollout (single particle)."""
    t: np.ndarray            # (T,) seconds
    pos_perp: np.ndarray     # (T,) rows
    pos_along: np.ndarray    # (T,) cols
    vel_perp: np.ndarray     # (T,) rows/s
    vel_along: np.ndarray    # (T,) cols/s
    mode: np.ndarray         # (T,) int8
    dt: float
    # per-saccade ground-truth main sequence (commanded, axis-resolved displacement)
    sacc_amplitude: np.ndarray   # (S,) rows — total commanded amplitude A
    sacc_peakvel: np.ndarray     # (S,) rows/s — commanded V_peak(A)


def rollout(duration_s: float = 60.0, fs: float = 1000.0, seed: int = 0,
            pos_perp: float = 0.0, pos_along: float = 0.0) -> Rollout:
    """Run the prior OPEN-LOOP (no observations) for ``duration_s`` at ``fs`` Hz.

    Repeatedly calls :func:`predict` on a single particle from an initial pursuit
    state, accumulating the trajectory and recording each saccade's commanded
    (amplitude, peak velocity) — the ground-truth main sequence the tests fit.
    Deterministic for a given ``seed``.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / float(fs)
    n_steps = int(round(duration_s * fs))
    st = init_state(1, pos_perp=pos_perp, pos_along=pos_along, mode=0)

    pp = np.empty(n_steps); pa = np.empty(n_steps)
    vp = np.empty(n_steps); va = np.empty(n_steps)
    mm = np.empty(n_steps, dtype=np.int8)
    sacc_amp: list[float] = []
    sacc_vpk: list[float] = []

    prev_mode = int(st.mode[0])
    for i in range(n_steps):
        st = predict(st, dt, rng)
        cur_mode = int(st.mode[0])
        # record a saccade's ground truth at the step it *starts* (0 -> 1)
        if prev_mode == 0 and cur_mode == 1:
            sacc_amp.append(float(st.sac_amp[0]))
            sacc_vpk.append(float(st.sac_vpeak[0]))
        prev_mode = cur_mode
        pp[i] = st.pos_perp[0]; pa[i] = st.pos_along[0]
        vp[i] = st.vel_perp[0]; va[i] = st.vel_along[0]
        mm[i] = cur_mode

    t = np.arange(n_steps, dtype=np.float64) * dt
    return Rollout(
        t=t, pos_perp=pp, pos_along=pa, vel_perp=vp, vel_along=va, mode=mm, dt=dt,
        sacc_amplitude=np.asarray(sacc_amp, dtype=np.float64),
        sacc_peakvel=np.asarray(sacc_vpk, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Rollout statistics (used by --report and to set test tolerances)
# ---------------------------------------------------------------------------


def main_sequence_fit(amplitude_rows: np.ndarray, peakvel_rows_s: np.ndarray
                      ) -> tuple[float, float]:
    """Log-log OLS fit of peak-velocity vs amplitude. Returns (slope, intercept).

    The slope is the main-sequence exponent (positive: bigger saccades faster).
    """
    good = (amplitude_rows > 0) & (peakvel_rows_s > 0)
    la = np.log10(amplitude_rows[good])
    lp = np.log10(peakvel_rows_s[good])
    A = np.vstack([la, np.ones_like(la)]).T
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    return float(coef[0]), float(coef[1])


def fixation_segments(ro: Rollout, min_dur_s: float = 0.2) -> list[tuple[int, int]]:
    """Contiguous pursuit/fixation runs (mode==0) at least ``min_dur_s`` long."""
    m = ro.mode == 0
    segs: list[tuple[int, int]] = []
    i, n = 0, len(m)
    min_len = int(min_dur_s / ro.dt)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i >= min_len:
                segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def _report() -> str:
    ro = rollout(duration_s=60.0, fs=1000.0, seed=0)
    slope, inter = main_sequence_fit(ro.sacc_amplitude, ro.sacc_peakvel)
    rate = len(ro.sacc_amplitude) / ro.t[-1]
    segs = fixation_segments(ro)
    drift = np.array([np.ptp(ro.pos_perp[a:b]) for a, b in segs]) * ARCMIN_PER_ROW
    # pursuit acceleration (rows/s^2) over fixation segments
    accs = []
    for a, b in segs:
        accs.append(np.abs(np.diff(ro.vel_perp[a:b])) / ro.dt)
    acc = np.concatenate(accs) if accs else np.zeros(1)
    amp_arc = ro.sacc_amplitude * ARCMIN_PER_ROW

    T = transition_matrix(ro.dt)
    L = []
    L.append("2D Gaze Tracker — G9 IMM dynamics prior (pure-prior rollout)")
    L.append("=" * 60)
    L.append(f"  rollout: 60 s @ 1000 Hz, seed 0")
    L.append(f"  A0 = {A0_ARCMIN:.2f}' ({A0_ROWS:.2f} rows)   "
             f"VMAX = {VMAX_ROWS_S:.0f} rows/s ({VMAX_ROWS_S/ROWS_PER_DEG:.0f} deg/s)")
    L.append("  " + "-" * 56)
    L.append(f"  transition matrix T(dt={ro.dt*1e3:.2f} ms):")
    L.append(f"    pursuit  -> [{T[0,0]:.5f}  {T[0,1]:.5f}]")
    L.append(f"    saccade  -> [{T[1,0]:.5f}  {T[1,1]:.5f}]")
    L.append("  " + "-" * 56)
    L.append(f"  saccades: n = {len(ro.sacc_amplitude)}   rate = {rate:.2f} /s")
    L.append(f"    amplitude arcmin p10/p50/p90 = "
             f"{np.percentile(amp_arc,[10,50,90]).round(1)}")
    L.append(f"    main-sequence log-log slope  = {slope:.4f}")
    L.append(f"    peak-vel max / VMAX          = "
             f"{ro.sacc_peakvel.max()/VMAX_ROWS_S:.3f}")
    L.append(f"    amp<->peakvel corr           = "
             f"{np.corrcoef(ro.sacc_amplitude, ro.sacc_peakvel)[0,1]:.3f}")
    L.append("  " + "-" * 56)
    L.append(f"  fixation/pursuit: n segs = {len(segs)}")
    L.append(f"    drift (ptp/seg) arcmin p50/p90 = "
             f"{np.median(drift):.2f} / {np.percentile(drift,90):.2f}")
    L.append(f"    accel rows/s^2 p50/p99         = "
             f"{np.percentile(acc,50):.0f} / {np.percentile(acc,99):.0f} "
             f"(cap {ACCEL_CAP_ROWS_S2:.0f})")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="G9 IMM dynamics prior")
    p.add_argument("--report", action="store_true",
                   help="print the pure-prior rollout main-sequence + drift stats")
    args = p.parse_args()
    if args.report:
        print(_report())
    else:
        p.print_help()
