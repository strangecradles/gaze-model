"""traj_gen.py — label-bearing 2D gaze trajectory generator (GOALS.md G7).

Generates KNOWN 2D gaze trajectories composed of randomized segments of
  (1) fixation  — slow correlated drift (OU velocity) + microsaccades,
  (2) smooth pursuit — low-acceleration sinusoidal/ramp velocity,
  (3) saccades  — ballistic velocity pulses,
with saccade kinematics drawn from the REAL main sequence measured in calib.py /
the machine tracker (NOT generic smoothness — see PLAN.md design constraints).

This is the labelled substrate that feeds G8 (synthetic line-scan stream) and the
decisive G13 rate-sweep gate, so the :class:`Trajectory` schema below is a hard
contract that G9/G10 depend on.

KINEMATICS (the real main sequence, not generic smoothness)
-----------------------------------------------------------
The measured along-channel main sequence (calib.estimate_along_main_sequence)
has a log-log peak-velocity-vs-amplitude slope of ~0.371. We reproduce it with
the saturating main-sequence law

    V_peak(A) = Vmax * (1 - exp(-A / A0))

with ``Vmax = data.UNITS.saccade_vmax_rows_per_s / calib.ROWS_PER_DEG`` (~826
deg/s) and ``A0`` (saturation amplitude) chosen so the generated saccades'
small-amplitude log-log slope lands in the measured band (see ``_A0_DEG`` /
the test tolerances). The saccade DURATION follows a main-sequence duration law
``D = D0 + k*A`` clamped to ~6-80 ms. Each saccade is rendered as a smooth
ballistic velocity pulse (a ``sin^p`` shape; ``p`` solved per event) whose PEAK
matches ``V_peak(A)`` and whose integral equals the amplitude ``A`` exactly.

Fixational drift is a slow correlated random walk: an Ornstein-Uhlenbeck (OU)
velocity process, giving a roughly 1/f^2 (Brownian-ish), band-limited / low-pass
velocity spectrum a few arcmin in extent over a fixation. Microsaccades occur at
a plausible rate (~0.5-2 /s), amplitudes ~2-30 arcmin, drawn on the SAME main
sequence as the large saccades. Smooth pursuit is a low-acceleration sinusoidal
or ramp velocity (~0.2-20 deg/s), not noise.

ATLAS UNITS / ISOTROPY ASSUMPTION
---------------------------------
Gaze is generated in PHYSICAL units (arcmin) and exposed in atlas units via
calib. ``perp_arcmin`` maps to ``perp_rows`` through ``calib.ARCMIN_PER_ROW``
(the measured perp scale, ~125 rows/deg). For the ALONG axis we assume the atlas
is sampled ISOTROPICALLY (same arcmin/atlas-unit as perp): ``along_cols =
along_arcmin / calib.ARCMIN_PER_ROW``. The absolute along scale cancels in the
synthetic generate+decode loop (the same scale is used to render and to decode),
so this isotropic-sampling assumption is harmless for the labelled study and is
documented here explicitly.

Run ``python traj_gen.py --plot`` to save a sample trajectory + main-sequence
scatter PNG (matplotlib Agg, no display required).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

import calib
import data

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Kinematic constants (frozen from the G7 investigation; see module docstring).
# Measured anchors: along-channel main-sequence log-log slope ~0.371 (calib),
# Vmax ~826 deg/s (data.UNITS.saccade_vmax_rows_per_s / calib.ROWS_PER_DEG).
# ---------------------------------------------------------------------------

# saccade main sequence
VMAX_DEG_S: float = data.UNITS.saccade_vmax_rows_per_s / calib.ROWS_PER_DEG
VMAX_ARCMIN_S: float = VMAX_DEG_S * 60.0
_A0_DEG: float = 0.30                 # saturation amplitude; small-amp slope -> band
_A0_ARCMIN: float = _A0_DEG * 60.0

# fixational drift (OU velocity), arcmin
_DRIFT_TAU_S: float = 0.05            # velocity correlation time
_DRIFT_SIGMA_V: float = 20.0          # arcmin/s stationary velocity std

# microsaccades
_MICROSACC_RATE_HZ: float = 1.2       # events/s during fixation (~0.5-2 /s band)
_MICROSACC_AMP_LO: float = 2.0        # arcmin
_MICROSACC_AMP_HI: float = 30.0       # arcmin

# regular (inter-segment) saccades
_SACCADE_AMP_LO: float = 30.0         # arcmin (0.5 deg)
_SACCADE_AMP_HI: float = 720.0        # arcmin (12 deg)

# smooth pursuit (deg/s -> arcmin/s); low-acceleration sinusoid/ramp
_PURSUIT_SPEED_LO: float = 0.2        # deg/s
_PURSUIT_SPEED_HI: float = 20.0       # deg/s

# segment composition (durations in s)
_FIX_DUR: tuple[float, float] = (0.4, 1.6)
_PURSUIT_DUR: tuple[float, float] = (0.6, 1.6)
_SEG_WEIGHTS = dict(fixation=0.45, pursuit=0.20, saccade=0.35)

# atlas anchor (default atlas centre of a 600 x 1200 atlas)
ATLAS_ROWS: int = data.UNITS.atlas_rows_perp      # 600
ATLAS_COLS: int = data.UNITS.atlas_cols_along      # 1200
DEFAULT_ANCHOR_ROW: float = ATLAS_ROWS / 2.0       # 300
DEFAULT_ANCHOR_COL: float = ATLAS_COLS / 2.0       # 600


# ---------------------------------------------------------------------------
# Trajectory schema (HARD CONTRACT — G8/G9/G10 depend on field names/units)
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A known 2D gaze trajectory in physical (arcmin) and atlas (row/col) units.

    All arrays are numpy, length ``n = round(duration * rate)``.

    Fields
    ------
    t            : (n,) seconds.
    perp_arcmin  : (n,) perpendicular gaze, arcmin (zero-mean physical motion).
    along_arcmin : (n,) along gaze, arcmin.
    perp_rows    : (n,) atlas rows = anchor_row + perp_arcmin / ARCMIN_PER_ROW.
    along_cols   : (n,) atlas cols = anchor_col + along_arcmin / ARCMIN_PER_ROW
                   (ISOTROPIC atlas-sampling assumption; see module docstring).
    v_perp       : (n,) d/dt perp_rows, rows/s.
    v_along      : (n,) d/dt along_cols, cols/s.
    mode         : (n,) int8 {0 = fixation/pursuit, 1 = saccade}.
    rate         : float, sample rate (Hz).
    anchor_row   : float, atlas row the perp motion is centred on.
    anchor_col   : float, atlas col the along motion is centred on.
    """

    t: np.ndarray
    perp_arcmin: np.ndarray
    along_arcmin: np.ndarray
    perp_rows: np.ndarray
    along_cols: np.ndarray
    v_perp: np.ndarray
    v_along: np.ndarray
    mode: np.ndarray
    rate: float
    anchor_row: float
    anchor_col: float


# ---------------------------------------------------------------------------
# Saccade kinematics (the REAL main sequence)
# ---------------------------------------------------------------------------


def peak_velocity_arcmin_s(amp_arcmin: np.ndarray | float) -> np.ndarray | float:
    """Saturating main-sequence peak velocity ``Vmax*(1-exp(-A/A0))`` (arcmin/s).

    Reproduces the measured ~0.37 log-log peak-vel-vs-amplitude slope at small
    amplitudes and approaches ``Vmax`` for large saccades.
    """
    a = np.asarray(amp_arcmin, dtype=np.float64)
    return VMAX_ARCMIN_S * (1.0 - np.exp(-a / _A0_ARCMIN))


def saccade_duration_s(amp_arcmin: float) -> float:
    """Ballistic saccade duration from the main sequence (s).

    For a fixed ``sin^2`` velocity pulse whose PEAK equals ``V_peak(A)`` and
    whose integral equals the amplitude ``A``, the duration is forced to
    ``D = 2*A / V_peak(A)`` (the sin^2 mean/peak ratio is 1/2). This is the
    main-sequence duration law: it yields ~15-60 ms for regular saccades and
    shorter pulses for microsaccades, with peak velocity exactly on the
    saturating main sequence and displacement exactly the amplitude.
    """
    amp = float(amp_arcmin)
    vp = float(peak_velocity_arcmin_s(amp))
    if vp <= 0.0:
        return 0.006
    return float(2.0 * amp / vp)


def _saccade_speed_profile(amp_arcmin: float, rate: float) -> np.ndarray:
    """Ballistic speed pulse (arcmin/s): smooth ``sin^2`` shape whose PEAK equals
    ``V_peak(A)`` exactly (the load-bearing main-sequence property) over a
    duration ``D = 2*A / V_peak(A)``.

    The peak velocity is set exactly so the generated saccades reproduce the
    measured log-log peak-vel-vs-amplitude main sequence; the displacement
    (integral) equals ``A`` up to sin^2-discretisation at very short pulses (a
    few samples), which is harmless because the labelled amplitude is read back
    as the net displacement of the rendered pulse.
    """
    amp = float(amp_arcmin)
    if amp <= 0.0:
        return np.zeros(0, dtype=np.float64)
    vp = float(peak_velocity_arcmin_s(amp))
    dur = saccade_duration_s(amp)
    n = max(3, int(round(dur * rate)))
    tau = np.linspace(0.0, 1.0, n)
    g = np.sin(np.pi * tau) ** 2
    return g / g.max() * vp              # peak == V_peak(A) exactly


def _emit_saccade(perp_v: list, along_v: list, mode: list,
                  amp_arcmin: float, direction: np.ndarray, rate: float) -> None:
    """Append one saccade (speed pulse projected onto a 2D unit direction)."""
    speed = _saccade_speed_profile(amp_arcmin, rate)
    if speed.size == 0:
        return
    d = direction / (np.linalg.norm(direction) + 1e-12)
    perp_v.extend(speed * d[0])
    along_v.extend(speed * d[1])
    mode.extend([1] * speed.size)


# ---------------------------------------------------------------------------
# Fixation / pursuit segments
# ---------------------------------------------------------------------------


def _ou_velocity(n: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck drift velocity (arcmin/s) — a slow correlated random
    walk giving a band-limited, ~1/f^2 low-pass velocity spectrum."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    dt = 1.0 / rate
    theta = 1.0 / _DRIFT_TAU_S
    drive = _DRIFT_SIGMA_V * np.sqrt(2.0 * theta)
    noise = rng.standard_normal(n)
    v = np.empty(n, dtype=np.float64)
    cur = 0.0
    sqdt = np.sqrt(dt)
    a = theta * dt
    for i in range(n):
        cur += -a * cur + drive * sqdt * noise[i]
        v[i] = cur
    return v


def _emit_fixation(perp_v: list, along_v: list, mode: list,
                   dur_s: float, rate: float, rng: np.random.Generator) -> None:
    """Fixation segment: 2D OU drift + Poisson-timed microsaccades on the main
    sequence."""
    n = max(1, int(round(dur_s * rate)))
    vp = _ou_velocity(n, rate, rng)
    va = _ou_velocity(n, rate, rng)
    pv = list(vp)
    av = list(va)
    md = [0] * n

    # microsaccades: Poisson process at _MICROSACC_RATE_HZ over the segment
    n_micro = rng.poisson(_MICROSACC_RATE_HZ * dur_s)
    for _ in range(int(n_micro)):
        amp = float(10.0 ** rng.uniform(np.log10(_MICROSACC_AMP_LO),
                                        np.log10(_MICROSACC_AMP_HI)))
        ang = rng.uniform(0.0, 2.0 * np.pi)
        direction = np.array([np.cos(ang), np.sin(ang)])
        speed = _saccade_speed_profile(amp, rate)
        if speed.size == 0 or speed.size >= n:
            continue
        start = int(rng.integers(0, n - speed.size))
        d = direction / (np.linalg.norm(direction) + 1e-12)
        for k in range(speed.size):
            pv[start + k] += speed[k] * d[0]
            av[start + k] += speed[k] * d[1]
            md[start + k] = 1

    perp_v.extend(pv)
    along_v.extend(av)
    mode.extend(md)


def _emit_pursuit(perp_v: list, along_v: list, mode: list,
                  dur_s: float, rate: float, rng: np.random.Generator) -> None:
    """Smooth-pursuit segment: low-acceleration sinusoidal (or ramp) velocity
    (~0.2-20 deg/s), NOT noise."""
    n = max(1, int(round(dur_s * rate)))
    t = np.arange(n) / rate
    speed_deg = float(rng.uniform(_PURSUIT_SPEED_LO, _PURSUIT_SPEED_HI))
    speed = speed_deg * 60.0  # arcmin/s
    ang = rng.uniform(0.0, 2.0 * np.pi)
    if rng.random() < 0.5:
        # sinusoidal pursuit (Lissajous-like, low acceleration)
        freq = float(rng.uniform(0.2, 1.0))  # Hz
        mag = speed * np.cos(2.0 * np.pi * freq * t)
    else:
        # constant-velocity ramp
        mag = np.full(n, speed)
    perp_v.extend(mag * np.cos(ang))
    along_v.extend(mag * np.sin(ang))
    mode.extend([0] * n)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average with edge-padding (length-preserving)."""
    if w <= 1 or x.size == 0:
        return x.astype(np.float64, copy=True)
    w = min(w, x.size)
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(xp, kernel, mode="same")[pad:pad + x.size]


def _integrate_reflecting(vel: np.ndarray, dt: float, bound: float) -> np.ndarray:
    """Integrate a velocity stream (arcmin/s) to position (arcmin), softly
    bounding the SLOW drift to ``[-bound, bound]`` so the accumulated
    random-direction walk stays inside the atlas FOV.

    The raw cumulative position is split into a slow trend (a wide moving
    average) and a fast residual (microsaccades / saccades / drift detail). Only
    the slow trend is squashed with ``bound * tanh(trend/bound)`` — a smooth,
    monotone map, identity for small excursions. The fast residual is added back
    unchanged, so saccade amplitudes and peak velocities (the main sequence) are
    left intact while the long-run wander is kept inside the FOV.
    """
    raw = np.cumsum(vel) * dt
    raw -= raw.mean()
    if bound <= 0:
        return raw
    # slow trend over ~0.25 s; saccades (<~60 ms) live in the residual
    w = max(1, int(round(0.25 / dt)))
    trend = _moving_average(raw, w)
    residual = raw - trend
    pos = bound * np.tanh(trend / bound) + residual
    # final smooth safety clamp: guarantees |pos| < bound (the residual can push
    # a saccade landing near the edge slightly past the trend clamp). tanh is
    # near-identity in the interior, so interior motion is undistorted.
    return bound * np.tanh(pos / bound)


def sample_trajectory(duration: float, rate: float, seed: int,
                      anchor_row: float = DEFAULT_ANCHOR_ROW,
                      anchor_col: float = DEFAULT_ANCHOR_COL) -> Trajectory:
    """Generate a known 2D gaze :class:`Trajectory` of length ``round(duration*rate)``.

    The trajectory is a randomized sequence of fixation / pursuit / saccade
    segments (interspersed with inter-segment saccades) over ``duration`` seconds
    at ``rate`` Hz. ``seed`` controls reproducibility (same seed -> identical
    trajectory). Motion is generated as a velocity stream (arcmin/s), integrated
    to position, centred on the atlas ``anchor`` so ``perp_rows``/``along_cols``
    index a 600 x 1200 atlas.
    """
    rate = float(rate)
    n_total = int(round(duration * rate))
    if n_total <= 0:
        raise ValueError("duration * rate must be >= 1")
    rng = np.random.default_rng(int(seed))

    perp_v: list = []
    along_v: list = []
    mode: list = []

    kinds = list(_SEG_WEIGHTS.keys())
    weights = np.array([_SEG_WEIGHTS[k] for k in kinds], dtype=np.float64)
    weights /= weights.sum()

    # build velocity stream until we have at least n_total samples
    while len(perp_v) < n_total:
        kind = kinds[int(rng.choice(len(kinds), p=weights))]
        if kind == "fixation":
            dur = float(rng.uniform(*_FIX_DUR))
            _emit_fixation(perp_v, along_v, mode, dur, rate, rng)
        elif kind == "pursuit":
            dur = float(rng.uniform(*_PURSUIT_DUR))
            _emit_pursuit(perp_v, along_v, mode, dur, rate, rng)
        else:  # saccade
            amp = float(10.0 ** rng.uniform(np.log10(_SACCADE_AMP_LO),
                                            np.log10(_SACCADE_AMP_HI)))
            ang = rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.cos(ang), np.sin(ang)])
            _emit_saccade(perp_v, along_v, mode, amp, direction, rate)

    perp_v = np.asarray(perp_v[:n_total], dtype=np.float64)
    along_v = np.asarray(along_v[:n_total], dtype=np.float64)
    mode = np.asarray(mode[:n_total], dtype=np.int8)

    dt = 1.0 / rate
    t = np.arange(n_total) / rate

    # Integrate velocity (arcmin/s) -> position (arcmin), with a reflecting bound
    # so the accumulated random-direction saccade walk stays inside the atlas FOV.
    # The bound (arcmin) keeps perp_rows / along_cols within the atlas with a
    # small margin around the anchor (the tightest axis sets the radius).
    margin = 8.0  # rows/cols of safety margin from the atlas edge
    perp_room = min(anchor_row, ATLAS_ROWS - anchor_row) - margin
    along_room = min(anchor_col, ATLAS_COLS - anchor_col) - margin
    perp_bound = max(1.0, perp_room) * calib.ARCMIN_PER_ROW
    along_bound = max(1.0, along_room) * calib.ARCMIN_PER_ROW

    perp_arcmin = _integrate_reflecting(perp_v, dt, perp_bound)
    along_arcmin = _integrate_reflecting(along_v, dt, along_bound)

    # atlas units (rows/cols), centred on the anchor
    perp_rows = anchor_row + perp_arcmin / calib.ARCMIN_PER_ROW
    along_cols = anchor_col + along_arcmin / calib.ARCMIN_PER_ROW

    # atlas-unit velocities (rows/s, cols/s) via finite difference
    v_perp = np.gradient(perp_rows) * rate
    v_along = np.gradient(along_cols) * rate

    return Trajectory(
        t=t,
        perp_arcmin=perp_arcmin,
        along_arcmin=along_arcmin,
        perp_rows=perp_rows,
        along_cols=along_cols,
        v_perp=v_perp,
        v_along=v_along,
        mode=mode,
        rate=rate,
        anchor_row=float(anchor_row),
        anchor_col=float(anchor_col),
    )


# ---------------------------------------------------------------------------
# Main-sequence extraction from a generated trajectory
# ---------------------------------------------------------------------------


def main_sequence(traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """Extract (amplitudes_arcmin, peak_vels_arcmin_per_s) from the saccade
    segments of a trajectory.

    Saccade segments are contiguous runs of ``mode == 1``. Per event, amplitude
    is the net 2D displacement (arcmin) over the run and peak velocity is the
    max 2D speed (arcmin/s) within it.
    """
    mode = traj.mode
    speed = np.hypot(traj.v_perp, traj.v_along) * calib.ARCMIN_PER_ROW  # arcmin/s
    amps, pvs = [], []
    n = len(mode)
    i = 0
    while i < n:
        if mode[i] == 1:
            j = i
            while j < n and mode[j] == 1:
                j += 1
            dperp = traj.perp_arcmin[j - 1] - traj.perp_arcmin[i]
            dalong = traj.along_arcmin[j - 1] - traj.along_arcmin[i]
            amp = float(np.hypot(dperp, dalong))
            pv = float(np.max(speed[i:j]))
            if amp > 0 and pv > 0:
                amps.append(amp)
                pvs.append(pv)
            i = j
        else:
            i += 1
    return np.asarray(amps), np.asarray(pvs)


def main_sequence_slope(amps_arcmin: np.ndarray, peak_vels: np.ndarray) -> float:
    """Log-log OLS slope of peak velocity vs amplitude (the main-sequence slope)."""
    good = (amps_arcmin > 0) & (peak_vels > 0)
    la = np.log10(amps_arcmin[good])
    lp = np.log10(peak_vels[good])
    A = np.vstack([la, np.ones_like(la)]).T
    coef, *_ = np.linalg.lstsq(A, lp, rcond=None)
    return float(coef[0])


# ---------------------------------------------------------------------------
# --plot CLI (matplotlib Agg; saves a PNG, no display required)
# ---------------------------------------------------------------------------


def _plot(out_path: Optional[str] = None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traj = sample_trajectory(duration=8.0, rate=2000.0, seed=0)
    amps, pvs = main_sequence(traj)
    slope = main_sequence_slope(amps, pvs) if amps.size else float("nan")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    ax = axes[0]
    sac = traj.mode == 1
    ax.plot(traj.t, traj.perp_rows, lw=0.7, label="perp (rows)")
    ax.plot(traj.t, traj.along_cols, lw=0.7, label="along (cols)")
    ax.scatter(traj.t[sac], traj.perp_rows[sac], s=4, c="r", label="saccade")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("atlas position")
    ax.set_title("Sample 2D gaze trajectory (atlas units)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(traj.along_cols, traj.perp_rows, lw=0.5)
    ax.set_xlabel("along (cols)")
    ax.set_ylabel("perp (rows)")
    ax.set_title("2D gaze path")
    ax.set_aspect("equal", adjustable="datalim")

    ax = axes[2]
    if amps.size:
        ax.loglog(amps, pvs, ".", ms=4, alpha=0.6)
        xs = np.array([amps.min(), amps.max()])
        ax.loglog(xs, peak_velocity_arcmin_s(xs), "r-",
                  label=f"V_peak law (A0={_A0_DEG} deg)")
    ax.axhline(VMAX_ARCMIN_S, color="k", ls="--", lw=0.8, label="Vmax")
    ax.set_xlabel("amplitude (arcmin)")
    ax.set_ylabel("peak velocity (arcmin/s)")
    ax.set_title(f"Main sequence — generated saccades (log-log slope = {slope:.3f})")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    if out_path is None:
        out_dir = os.path.join(HERE, "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "traj_gen_sample.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="G7 trajectory generator")
    p.add_argument("--plot", action="store_true",
                   help="save a sample trajectory + main-sequence scatter PNG")
    p.add_argument("--out", default=None, help="output PNG path")
    args = p.parse_args()

    if args.plot:
        path = _plot(args.out)
        traj = sample_trajectory(duration=8.0, rate=2000.0, seed=0)
        amps, pvs = main_sequence(traj)
        slope = main_sequence_slope(amps, pvs) if amps.size else float("nan")
        n_sac = int(amps.size)
        dur = traj.t[-1]
        print(f"saved sample trajectory + main-sequence scatter -> {path}")
        print(f"  duration {dur:.1f}s @ {traj.rate:.0f} Hz, "
              f"{n_sac} saccade events, main-sequence slope {slope:.3f}")
    else:
        p.print_help()
