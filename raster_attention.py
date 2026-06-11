"""raster_attention.py — scalar attention/engagement metric a(t) for teleop labels.

Built on the RASTER strip-tracking gaze (raster_track/raster_study) from this
study. (A complementary, earlier a(t) built on the M4 line-scan reconstruction
lives in attention.py.)

When >1 kHz absolute gaze is available (it is — see raster_study/raster_synth),
we can also distill a single scalar a(t) in [0,1] = "how deeply is the operator
focusing/attending right now", to append to robot-teleop trajectories as an
auxiliary imitation-learning signal beyond (video, trajectory).

a(t) fuses four literature-grounded, sensor-diverse oculomotor/pupillary
correlates of attention, each computed from THIS capture and z-scored over the
session:

  1. fixation stability  (-gaze dispersion / BCEA in a sliding window):
     focused attention -> tight, stable gaze.                       [Di Russo;
  2. (micro)saccade inhibition (-saccade rate): sustained concentration
     suppresses saccadic sampling, rebounding when attention lapses. [Engbert,
     Pastukhov & Braun]
  3. pursuit gain (eye-vs-target velocity match, when a target exists):
     engaged tracking -> gain ~1; lapses drop it.                    [Kowler]
  4. pupil-linked arousal (tonic + phasic pupil area): LC-NE arousal /
     cognitive effort, an INDEPENDENT sensor from the eye-position channel.
                                                                     [Kahneman;
                                                                      Mathot]

Components are combined (weighted, logistic-squashed) into a(t) with a
confidence c(t) from lock/FOV/blink validity. Ground-truth attention is not
available, so validation is by (a) sensible response to task structure
(engaged pursuit vs saccade/FOV-loss), (b) inter-component agreement, and
(c) agreement with the independent pupil sensor — self-consistency is
necessary, not sufficient (cf. PLAN.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d

import data
import raster_track as rt
import raster_study as rs

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def _z(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    mu = np.nanmean(x[m]) if m.any() else 0.0
    sd = np.nanstd(x[m]) if m.any() else 1.0
    return (x - mu) / (sd if sd > 0 else 1.0)


def _win(strip_hz, seconds):
    return max(3, int(round(strip_hz * seconds)))


@dataclass
class Attention:
    t: np.ndarray
    a: np.ndarray            # [0,1] attention/engagement
    conf: np.ndarray         # [0,1] confidence (lock/FOV/blink)
    comp: dict               # component z-series (for inspection/validation)
    strip_hz: float


def _bin(t_src, x_src, t_grid, fn=np.median):
    """Aggregate scattered (t_src, x_src) into t_grid bins (robust denoise)."""
    dt = t_grid[1] - t_grid[0]
    idx = np.clip(((t_src - t_grid[0]) / dt + 0.5).astype(int), 0, len(t_grid) - 1)
    out = np.full(len(t_grid), np.nan)
    order = np.argsort(idx, kind="stable")
    idx_s = idx[order]; x_s = np.asarray(x_src)[order]
    starts = np.searchsorted(idx_s, np.arange(len(t_grid)), "left")
    ends = np.searchsorted(idx_s, np.arange(len(t_grid)), "right")
    for k in range(len(t_grid)):
        if ends[k] > starts[k]:
            out[k] = fn(x_s[starts[k]:ends[k]])
    return out


def detect_saccades(perp, along, rk, lam=6.0, min_dur=3, refr_s=0.02):
    """Engbert-Kliegl 2D velocity-threshold saccade detection on a band-limited
    gaze. Returns (onset_idx list, in_saccade bool array)."""
    vp = np.gradient(perp) * rk; va = np.gradient(along) * rk
    sp = np.sqrt(max(np.median(vp ** 2) - np.median(vp) ** 2, 1e-9))
    sa = np.sqrt(max(np.median(va ** 2) - np.median(va) ** 2, 1e-9))
    test = (vp / (lam * sp)) ** 2 + (va / (lam * sa)) ** 2 > 1.0
    onsets = []; insac = np.zeros(len(perp), bool)
    i = 0; refr = int(refr_s * rk); last = -10 ** 9
    while i < len(test):
        if test[i]:
            j = i
            while j < len(test) and test[j]:
                j += 1
            if j - i >= min_dur and i - last > refr:
                onsets.append(i); insac[i:j] = True; last = i
            i = j
        else:
            i += 1
    return onsets, insac


def compute(which="test1", S=8, rk=240.0, win_s=1.0) -> Attention:
    """Attention a(t) at a kinematics rate rk (Hz). The kHz gaze is binned to rk
    (medians) to remove the per-strip registration noise floor; eye motion is
    band-limited so attention kinematics live well below rk."""
    tk = rt.track(which, S=S, ref_mode="incremental")
    refs = rs.load_refs(which)
    Hd = rs._detrend(median_filter(tk.along_px, 5), tk.strip_hz)
    Vd = rs._detrend(median_filter(tk.perp_px, 5), tk.strip_hz)
    off = rs.joint_offset(tk.t, Hd, Vd, tk.infov, refs)
    scale = rs.measure_scale(tk.t, Hd, Vd, tk.infov, off, refs)
    sa = scale.along if np.isfinite(scale.along) else 0.58
    sp = scale.perp if np.isfinite(scale.perp) else 0.40

    # bin kHz gaze -> rk grid (median; robust to per-strip noise)
    t = np.arange(tk.t[0], tk.t[-1], 1.0 / rk)
    g = tk.infov
    perp = _bin(tk.t[g], (tk.perp_px[g]) * sp, t)      # arcmin
    along = _bin(tk.t[g], (tk.along_px[g]) * sa, t)
    fov = _bin(tk.t, tk.infov.astype(float), t, np.mean)
    fov = np.nan_to_num(fov, nan=0.0)
    perp = rs._fill(perp); along = rs._fill(along)
    # light smooth at rk
    perp = gaussian_filter1d(perp, 2); along = gaussian_filter1d(along, 2)
    speed = np.hypot(np.gradient(perp), np.gradient(along)) * rk        # arcmin/s

    w = _win(rk, win_s)
    def roll_std(x):
        m = uniform_filter1d(x, w, mode="nearest")
        v = uniform_filter1d(x * x, w, mode="nearest") - m * m
        return np.sqrt(np.clip(v, 0, None))
    # (1) fixation/tracking STABILITY: residual jitter = gaze minus its own smooth
    # (<~3 Hz) component, so following a moving target is NOT penalized; only the
    # tremor/jitter/lag residual remains. Low residual = steady, focused gaze.
    smooth_lp = _win(rk, 1.0 / (2 * np.pi * 3.0))
    res_p = perp - gaussian_filter1d(perp, smooth_lp)
    res_a = along - gaussian_filter1d(along, smooth_lp)
    disp = np.hypot(roll_std(res_p), roll_std(res_a))                  # arcmin
    c_stab = _z(-np.log(disp + 1e-2))
    # (2) (micro)saccade inhibition: -windowed saccade-event rate
    onsets, insac = detect_saccades(perp, along, rk)
    imp = np.zeros(len(t)); imp[onsets] = 1.0
    sac_rate = uniform_filter1d(imp, w, mode="nearest") * rk           # events/s
    c_inhib = _z(-sac_rate)
    # (3) pursuit gain (target present): windowed eye-vs-dot velocity regression
    dotx = np.interp(t, refs.t_stim + off, refs.dotx)
    doty = np.interp(t, refs.t_stim + off, refs.doty)
    vdx = np.gradient(dotx) * rk; vdy = np.gradient(doty) * rk
    vex = np.gradient(along) * rk; vey = np.gradient(perp) * rk
    ve = vex * np.sign(vdx) + vey * np.sign(vdy)
    vd = np.abs(vdx) + np.abs(vdy)
    mve = uniform_filter1d(ve, w); mvd = uniform_filter1d(vd, w)
    cov = uniform_filter1d(ve * vd, w) - mve * mvd
    vv = uniform_filter1d(vd * vd, w) - mvd * mvd
    gain = np.clip(cov / (vv + 1e-6), 0, 2)
    target_present = np.interp(t, refs.t_stim + off, (refs.phase != "sync").astype(float)) > 0.5
    c_pursuit = np.nan_to_num(_z(np.where(target_present, gain, np.nan)), nan=0.0)
    # (4) pupil arousal: tonic + phasic pupil area (independent sensor)
    tr = data.load_tracker(which)
    ttr = tr.t_s - tr.t_s[0]
    area_t = np.interp(t, ttr + off, rs._fill(tr.right_area_mm2))
    c_pupil = 0.7 * _z(area_t) + 0.3 * _z(area_t - gaussian_filter1d(area_t, _win(rk, 2.0)))

    conf = uniform_filter1d(fov, w, mode="nearest")

    # fuse (weights: stability, inhibition, pursuit, pupil)
    W_ = dict(stab=0.30, inhib=0.20, pursuit=0.25, pupil=0.25)
    raw = (W_["stab"] * c_stab + W_["inhib"] * c_inhib +
           W_["pursuit"] * c_pursuit + W_["pupil"] * c_pupil)
    a = 1.0 / (1.0 + np.exp(-raw))
    a = np.where(conf > 0.2, a, np.nan)
    comp = dict(stability=c_stab, inhibition=c_inhib, pursuit=c_pursuit,
                pupil=c_pupil, disp=disp, sac_rate=sac_rate, gain=gain,
                speed=speed, off=off, scale=(sa, sp), n_saccades=len(onsets))
    return Attention(t, a, conf, comp, rk)


def validate(att: Attention, which="test1") -> dict:
    """Self-consistency checks (necessary, not sufficient)."""
    refs = rs.load_refs(which)
    off = att.comp["off"]
    t = att.t
    fin = np.isfinite(att.a)
    out = {}
    # (a) inter-component agreement
    def corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 50:
            return np.nan
        x = x[m] - x[m].mean(); y = y[m] - y[m].mean()
        d = np.linalg.norm(x) * np.linalg.norm(y)
        return float((x * y).sum() / d) if d else np.nan
    out["r_stab_inhib"] = corr(att.comp["stability"], att.comp["inhibition"])
    out["r_stab_pupil"] = corr(att.comp["stability"], att.comp["pupil"])
    out["r_pursuit_pupil"] = corr(att.comp["pursuit"], att.comp["pupil"])
    # (b) task structure: engaged smooth-pursuit (low dot speed, mid-trace) vs
    #     near big saccades. Compare a during smooth-pursuit vs during fast/lapse.
    speed = att.comp["speed"]
    sac_rate = att.comp["sac_rate"]
    engaged = fin & (sac_rate < np.nanpercentile(sac_rate, 33))
    lapsing = fin & (sac_rate > np.nanpercentile(sac_rate, 67))
    out["a_engaged_med"] = float(np.nanmedian(att.a[engaged]))
    out["a_lapsing_med"] = float(np.nanmedian(att.a[lapsing]))
    # (c) pupil (independent) vs full a
    out["r_a_pupil"] = corr(att.a, att.comp["pupil"])
    out["frac_valid"] = float(fin.mean())
    return out


def export_for_teleop(att: Attention, rate_hz=100.0,
                      path=os.path.join(RESULTS, "raster_attention_track.csv")):
    """Resample a(t)+confidence to a teleop control rate and write a CSV with a
    documented schema for appending to robot-teleop trajectories."""
    import pandas as pd
    t0, t1 = att.t[0], att.t[-1]
    tg = np.arange(t0, t1, 1.0 / rate_hz)
    a = np.interp(tg, att.t, np.nan_to_num(att.a, nan=np.nanmedian(att.a)))
    c = np.interp(tg, att.t, np.nan_to_num(att.conf, nan=0.0))
    a = np.clip(np.nan_to_num(a, nan=0.0), 0.0, 1.0)
    c = np.clip(np.nan_to_num(c, nan=0.0), 0.0, 1.0)
    df = pd.DataFrame({"t_s": tg, "attention": a, "confidence": c})
    df.to_csv(path, index=False)
    return path, df


def figure(att: Attention, which="test1", path=os.path.join(RESULTS, "raster_attention.png")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    refs = rs.load_refs(which)
    off = att.comp["off"]
    fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    ax[0].plot(att.t, att.comp["speed"], lw=.4, color="0.4")
    ax[0].set_ylabel("gaze speed\n('/s)"); ax[0].set_yscale("symlog")
    ax[0].set_title("Attention metric a(t) and its oculomotor/pupil components")
    ax[1].plot(att.t, att.comp["sac_rate"], lw=.6, color="C1", label="saccade rate (/s)")
    ax[1].plot(att.t, att.comp["disp"], lw=.6, color="C4", label="dispersion (')")
    ax[1].legend(fontsize=8); ax[1].set_ylabel("sampling")
    ax[2].plot(att.t, att.comp["pupil"], lw=.6, color="C2", label="pupil arousal (z)")
    ax[2].plot(att.t, att.comp["pursuit"], lw=.6, color="C0", alpha=.7, label="pursuit gain (z)")
    ax[2].legend(fontsize=8); ax[2].set_ylabel("arousal / gain")
    ax[3].plot(att.t, att.a, lw=.8, color="k", label="a(t)")
    ax[3].plot(att.t, att.conf, lw=.6, color="C3", alpha=.6, label="confidence")
    ax[3].set_ylim(0, 1); ax[3].legend(fontsize=8); ax[3].set_ylabel("a(t)")
    ax[3].set_xlabel("time (s)")
    # phase bands
    for nm, c in [("H_sine", "0.92"), ("V_sine", "0.86"), ("circle", "0.92"), ("lissajous", "0.86")]:
        sel = refs.phase == nm
        if sel.any():
            t0 = refs.t_stim[sel][0] + off; t1 = refs.t_stim[sel][-1] + off
            for a_ in ax:
                a_.axvspan(t0, t1, color=c, alpha=.4, lw=0)
    plt.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def main():
    att = compute()
    v = validate(att)
    print("[attention] validation (self-consistency):")
    for k, val in v.items():
        print(f"   {k:18s} {val:+.3f}")
    fp = figure(att)
    cp, df = export_for_teleop(att)
    print(f"wrote {fp}\nwrote {cp} ({len(df)} rows @ teleop rate)")


if __name__ == "__main__":
    main()
