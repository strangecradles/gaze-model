"""attention.py — scalar visual-attention metric a(t) for teleop IL features.

The goal's fallback deliverable: a single [0,1] scalar per timestep, derived from
the best 2D gaze reconstruction (the M4 differential particle filter @ 11.8 kHz,
`khz2d` method ``m4_dpf_11823``), for appending to robot-teleop trajectories as
an imitation-learning feature beyond (video, trajectory).

a(t) is an OBSERVABILITY / intake proxy — necessary, not sufficient:

    a(t) = clip( smooth( g_intake * g_lock , TAU_ATT ), 0, 1 )

  - g_intake = 0.5*(1 - tanh((speed - V_HALF)/(0.5*V_HALF)))  — saccadic-
    suppression gate: visual intake is high during fixation/slow pursuit and
    suppressed during fast saccades. V_HALF is set adaptively to a robust high
    percentile of the valid gaze speed (the reconstruction attenuates true
    saccade peak velocity, so this sits at the real pursuit<->saccade boundary).
  - g_lock = clip((max_ncc - Q_LO)/(Q_HI - Q_LO), 0, 1) — confidence/lock gate:
    drops on blink / out-of-FOV lines where gaze is unobservable.
  - smoothed over an attentional-dwell window TAU_ATT.

Validation (necessary, not sufficient): the intake gate falls monotonically with
gaze speed; a collapses on blink / out-of-FOV; a is lower in-flight during
saccades than fixation; a is lower for right/temporal gaze (right-eye SLO FOV
loss). See `attention.report()`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

import khz2d
import khz2d_methods as M

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")

BEST_METHOD = "m4_dpf_11823"     # best 2D gaze reconstruction (M4 PF @ 11.8 kHz)
SPEED_SMOOTH_MS = 3.0            # gaze-speed smoothing
V_HALF_FLOOR = 15.0             # deg/s floor for the adaptive intake half-speed
V_HALF_PCTL = 95.0              # percentile of valid speed used for V_HALF
Q_LO = 0.12                     # max_ncc below this -> g_lock = 0 (unobservable)
Q_HI = 0.6                      # max_ncc above this -> g_lock = 1 (locked)
TAU_ATT_MS = 50.0              # attentional-dwell smoothing window
SACCADE_SPEED_DEG_S = 30.0     # speed above which a line is "in-flight saccade"


def _speed_threshold(speed, valid):
    """Adaptive intake half-speed: a robust high percentile of the valid gaze
    speed, floored at V_HALF_FLOOR (deg/s)."""
    s = np.asarray(speed)[np.asarray(valid) & np.isfinite(speed)]
    if s.size == 0:
        return float(V_HALF_FLOOR)
    return float(max(V_HALF_FLOOR, np.percentile(s, V_HALF_PCTL)))


@dataclass
class Attention:
    t: np.ndarray          # (N,) s, SLO clock
    a: np.ndarray          # (N,) [0,1] attention/intake scalar
    rate: float            # Hz
    speed: np.ndarray      # (N,) deg/s gaze speed
    g_intake: np.ndarray   # (N,) [0,1] saccadic-suppression gate
    g_lock: np.ndarray     # (N,) [0,1] lock/FOV gate
    valid: np.ndarray      # (N,) bool observable
    v_half: float          # deg/s adaptive intake half-speed

    def resample(self, rate_hz: float):
        """Mean-pool a(t) onto a uniform grid at rate_hz; returns (t_grid, a_grid)."""
        dt = 1.0 / rate_hz
        t_grid = np.arange(self.t[0], self.t[-1], dt)
        idx = np.clip(((self.t - t_grid[0]) / dt + 0.5).astype(int), 0, len(t_grid) - 1)
        s = np.zeros(len(t_grid)); c = np.zeros(len(t_grid))
        np.add.at(s, idx, self.a); np.add.at(c, idx, 1.0)
        a_grid = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        m = np.isfinite(a_grid)
        if not m.all():
            a_grid = np.interp(np.arange(len(t_grid)), np.flatnonzero(m), a_grid[m])
        return t_grid, np.clip(a_grid, 0.0, 1.0)

    def for_timestamps(self, ts, pool: str = "nearest", half_window_s: float | None = None):
        """Sample a(t) at arbitrary control timestamps. pool='nearest' returns the
        nearest native sample; pool='mean' averages within +-half_window_s."""
        ts = np.asarray(ts, float)
        if pool == "nearest":
            j = np.clip(np.searchsorted(self.t, ts), 0, len(self.t) - 1)
            jl = np.clip(j - 1, 0, len(self.t) - 1)
            pick = np.where(np.abs(self.t[j] - ts) <= np.abs(self.t[jl] - ts), j, jl)
            return self.a[pick]
        hw = 0.01 if (half_window_s is None or half_window_s <= 0.0) else half_window_s
        lo = np.searchsorted(self.t, ts - hw, "left")
        hi = np.searchsorted(self.t, ts + hw, "right")
        out = np.empty(len(ts))
        for i in range(len(ts)):
            out[i] = self.a[lo[i]:hi[i]].mean() if hi[i] > lo[i] else \
                self.a[min(lo[i], len(self.a) - 1)]
        return np.clip(out, 0.0, 1.0)


def _smooth_ms(a, fs, ms):
    sigma = max(1.0, fs * ms / 1000.0)
    return gaussian_filter1d(khz2d.fill_nan(a), sigma)


def attention_signal(method: str = BEST_METHOD) -> Attention:
    rec = khz2d.load_method(method)
    if rec is None:
        raise FileNotFoundError(
            f"reconstruction cache for '{method}' not found — build it first "
            f"(e.g. `python khz2d_methods.py --method m4 --rate 11823`).")
    t = np.asarray(rec["t"], float)
    rate = float(rec["rate"])
    valid = rec["valid"].astype(bool)
    pxdeg = M.px_per_deg()
    x = _smooth_ms(np.asarray(rec["x_px"], float) / pxdeg, rate, SPEED_SMOOTH_MS)  # deg
    y = _smooth_ms(np.asarray(rec["y_px"], float) / pxdeg, rate, SPEED_SMOOTH_MS)
    speed = np.hypot(np.gradient(x) * rate, np.gradient(y) * rate)
    v_half = _speed_threshold(speed, valid)
    g_intake = 0.5 * (1.0 - np.tanh((speed - v_half) / (0.5 * v_half)))
    q = np.asarray(rec["max_ncc"], float)
    g_lock = np.clip((q - Q_LO) / (Q_HI - Q_LO), 0.0, 1.0)
    a = np.clip(_smooth_ms(g_intake * g_lock, rate, TAU_ATT_MS), 0.0, 1.0)
    a = np.where(valid & np.isfinite(a), a, 0.0)
    return Attention(t, a, rate, speed, g_intake, g_lock, valid, v_half)


def _validate(att: Attention) -> dict:
    """Honest checks that a(t) behaves as an attention/intake proxy."""
    refs = khz2d.refs()
    a, speed, valid = att.a, att.speed, att.valid
    bands = ((0, 5), (5, 15), (15, 30), (30, 60), (60, 1e9))
    band_a = []
    for lo, hi in bands:
        m = valid & (speed >= lo) & (speed < hi)
        band_a.append((lo, hi, float(att.g_intake[m].mean()) if m.any() else np.nan,
                       int(m.sum())))
    fix = valid & (speed < 5.0)
    sac = valid & (speed > SACCADE_SPEED_DEG_S)
    dotx = np.interp(att.t, refs["dot_t"] + refs["off"], refs["dot_x"])
    right = valid & (dotx > 0.0)
    leftc = valid & (dotx <= 0.0)
    return dict(
        a_mean=float(a.mean()),
        a_frac_high=float((a > 0.5).mean()),
        a_fix=float(a[fix].mean()) if fix.any() else np.nan,
        a_sac=float(a[sac].mean()) if sac.any() else np.nan,
        a_invalid=float(a[~valid].mean()) if (~valid).any() else 0.0,
        a_valid=float(a[valid].mean()) if valid.any() else np.nan,
        a_right=float(a[right].mean()) if right.any() else np.nan,
        a_leftc=float(a[leftc].mean()) if leftc.any() else np.nan,
        n_sac=int(sac.sum()), n_fix=int(fix.sum()),
        band_a=band_a, v_half=att.v_half)


def _figure(att: Attention, vd: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    ax[0].plot(att.t, att.speed, lw=.3, color="0.4")
    ax[0].axhline(att.v_half, color="C1", ls="--", lw=1, label=f"V_HALF={att.v_half:.0f} deg/s")
    ax[0].set_yscale("symlog"); ax[0].set_ylabel("gaze speed\n(deg/s)"); ax[0].legend(fontsize=8)
    ax[0].set_title("Visual-attention metric a(t) — M4 PF @ 11.8 kHz")
    ax[1].plot(att.t, att.g_intake, lw=.3, color="C0", label="g_intake")
    ax[1].plot(att.t, att.g_lock, lw=.3, color="C3", alpha=.6, label="g_lock")
    ax[1].set_ylabel("gates"); ax[1].legend(fontsize=8); ax[1].set_ylim(-.05, 1.05)
    ax[2].plot(att.t, att.a, lw=.4, color="k", label="a(t)")
    ax[2].set_ylabel("a(t)"); ax[2].set_ylim(0, 1); ax[2].set_xlabel("time (s)"); ax[2].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def report(method: str = BEST_METHOD):
    os.makedirs(RESULTS, exist_ok=True)
    att = attention_signal(method)
    vd = _validate(att)
    np.savez(os.path.join(CACHE, "attention.npz"), t=att.t, a=att.a, speed=att.speed,
             g_intake=att.g_intake, g_lock=att.g_lock, valid=att.valid, rate=att.rate)
    _figure(att, vd, os.path.join(RESULTS, "attention.png"))
    t100, a100 = att.resample(100.0)

    L = ["# Scalar Visual-Attention Metric a(t) — Teleop Imitation-Learning Feature\n"]
    L.append("## Definition\n")
    L.append("`a(t) = clip(smooth(g_intake * g_lock, TAU_ATT), 0, 1)` where\n")
    L.append(f"- **g_intake** = `0.5*(1 - tanh((speed - V_HALF)/(0.5*V_HALF)))`, V_HALF = "
             f"{att.v_half:.1f} deg/s,\n")
    L.append(f"- **g_lock** = `clip((max_ncc - {Q_LO})/( {Q_HI} - {Q_LO} ), 0, 1)`,\n")
    L.append(f"- smoothed over an attentional-dwell window TAU_ATT = {TAU_ATT_MS:.0f} ms.\n")
    L.append(f"Intake half-speed V_HALF was set adaptively to {att.v_half:.1f} deg/s (the "
             f"{V_HALF_PCTL:.0f}th pct of valid gaze speed, floored at {V_HALF_FLOOR:.0f}); "
             f"the reconstruction attenuates true saccade peak velocity so this sits at the "
             f"actual pursuit<->saccade boundary.\n")
    L.append("## Intake gate falls monotonically with gaze speed\n")
    L.append("| gaze speed band (deg/s) | mean g_intake | n |")
    L.append("|---|---|---|")
    for lo, hi, m, n in vd["band_a"]:
        hs = "inf" if hi >= 1e8 else f"{hi:.0f}"
        L.append(f"| {lo:.0f}-{hs} | {m:.3f} | {n} |")
    L.append("")
    L.append("## Behaviour (validation — necessary, not sufficient)\n")
    L.append("| quantity | a(t) |")
    L.append("|---|---|")
    L.append(f"| fixation / slow pursuit (speed < 5 deg/s, n={vd['n_fix']}) | {vd['a_fix']:.3f} |")
    L.append(f"| in-flight saccade (speed > {SACCADE_SPEED_DEG_S:.0f} deg/s, n={vd['n_sac']}) | {vd['a_sac']:.3f} |")
    L.append(f"| blink / out-of-FOV (invalid) | {vd['a_invalid']:.3f} |")
    L.append(f"| valid | {vd['a_valid']:.3f} |")
    L.append(f"| right/temporal gaze | {vd['a_right']:.3f} |")
    L.append(f"| left/centre gaze | {vd['a_leftc']:.3f} |")
    L.append("")
    L.append(f"a(t): N={len(att.a)} @ {att.rate:.0f} Hz, mean={att.a.mean():.3f}, "
             f"frac>0.5={(att.a > 0.5).mean() * 100:.0f}%; resampled to 100 Hz -> {len(a100)} steps.\n")
    md = os.path.join(RESULTS, "attention.md")
    with open(md, "w") as f:
        f.write("\n".join(L))
    print(f"a(t): N={len(att.a)} @ {att.rate:.0f} Hz, mean={att.a.mean():.3f}, "
          f"frac>0.5={(att.a > 0.5).mean() * 100:.0f}%")
    return att, vd


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--method", default=BEST_METHOD)
    a = ap.parse_args()
    if a.report:
        report(a.method)
    else:
        att = attention_signal(a.method)
        print(f"a(t): N={len(att.a)} @ {att.rate:.0f} Hz, mean={att.a.mean():.3f}, "
              f"frac>0.5={(att.a > 0.5).mean() * 100:.0f}%")
