"""raster_synth.py — labeled certification of strip-tracking accuracy vs rate.

Real test1 has no high-rate ground truth, so absolute arcmin accuracy cannot be
read off it (only correlation + a truth-free noise floor). Here we render a
SYNTHETIC raster from a clean test1-derived retina and a KNOWN 2D gaze
trajectory at the real per-column raster timing, inject the measured per-rate
noise, then recover gaze by strip registration against the (perfect) reference.
Error vs the known label gives certified accuracy at any strip rate — including
the high-rate regime no real capture supplies. This is PLAN.md's decisive
"prove it on labeled synthetic" step, adapted to the raster.

Trajectory contains fixation (drift+microsaccades), smooth pursuit, and
saccades, so accuracy is reported broken out by regime (fixation precision,
pursuit tracking, through-saccade) — the physically distinct limits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter1d

import data
import noise as noisemod
import raster_track as rt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

LINE_RATE = 11823.0       # measured test1 line rate (cols/s)
FPS = 14.633
W = 808                   # slow-axis cols per frame
H = 1000                  # fast-axis rows per frame


def make_truth(which="test1", crop=(1160, 1000)):
    """Clean retina image at test1 pixel scale = high-coverage center of the
    iteratively-built test1 mosaic, z-scored. Returns (truth f32, valid bool)."""
    frames, fps, mean_int = rt._load_frames(which, None)
    mosaic, cov, offs, (py, px), good = rt._get_mosaic(which, None, frames, mean_int)
    Hc, Wc = mosaic.shape
    ch, cw = min(crop[0], Hc), min(crop[1], Wc)
    r0 = (Hc - ch) // 2; c0 = (Wc - cw) // 2
    truth = mosaic[r0:r0 + ch, c0:c0 + cw].astype(np.float32)
    valid = cov[r0:r0 + ch, c0:c0 + cw] >= rt.MIN_COVER
    # fill any thin uncovered gaps with local mean so the synthetic render is clean
    if not valid.all():
        m = truth[valid].mean() if valid.any() else 0.0
        truth = np.where(valid, truth, m)
    truth = (truth - truth.mean()) / (truth.std() + 1e-9)
    return truth.astype(np.float32), valid


@dataclass
class SynTraj:
    t: np.ndarray          # (Ncol,) s — one entry per slow-axis column sample
    perp: np.ndarray       # (Ncol,) px (vertical gaze offset)
    along: np.ndarray      # (Ncol,) px (horizontal gaze offset)
    mode: np.ndarray       # (Ncol,) 0=fixation/pursuit, 1=saccade
    n_frames: int


def gen_trajectory(n_frames=180, amp_px=70.0, seed=0) -> SynTraj:
    """Known 2D gaze in retina px at per-column timing: smooth pursuit + OU
    fixational drift + occasional saccades (ballistic). amp_px bounds the
    excursion so the rendered window stays inside the truth crop."""
    rng = np.random.default_rng(seed)
    ncol = n_frames * W
    t = np.arange(ncol) / LINE_RATE
    dur = t[-1]
    # smooth pursuit: two slow sinusoids (sub-Hz), bounded
    fp = 0.2
    perp = 0.45 * amp_px * np.sin(2 * np.pi * fp * t + 0.3)
    along = 0.55 * amp_px * np.sin(2 * np.pi * fp * t * 0.8)
    # OU fixational drift (correlated, small)
    dt = 1.0 / LINE_RATE
    tau = 0.04
    a = np.exp(-dt / tau)
    sig = 6.0  # px/s drift velocity scale
    dvp = rng.standard_normal(ncol) * sig * np.sqrt(1 - a * a)
    dva = rng.standard_normal(ncol) * sig * np.sqrt(1 - a * a)
    vp = np.zeros(ncol); va = np.zeros(ncol)
    for i in range(1, ncol):
        vp[i] = a * vp[i - 1] + dvp[i]; va[i] = a * va[i - 1] + dva[i]
    perp = perp + np.cumsum(vp) * dt
    along = along + np.cumsum(va) * dt
    # saccades: Poisson ~1.5/s, ballistic min-jerk-ish pulses, main-sequence
    mode = np.zeros(ncol, np.int8)
    n_sac = rng.poisson(1.5 * dur)
    for _ in range(n_sac):
        c0 = rng.integers(0, ncol)
        A = float(np.clip(rng.lognormal(np.log(18), 0.6), 4, 120))  # px amplitude
        D = max(3, int((0.0025 + 0.0007 * A / 60.0) * LINE_RATE))   # ~few ms
        ang = rng.uniform(0, 2 * np.pi)
        i1 = min(ncol, c0 + D)
        k = np.linspace(0, 1, i1 - c0)
        prof = (k - np.sin(2 * np.pi * k) / (2 * np.pi))            # min-jerk-ish ramp 0->1
        perp[c0:i1] += A * np.cos(ang) * prof
        along[c0:i1] += A * np.sin(ang) * prof
        perp[i1:] += A * np.cos(ang); along[i1:] += A * np.sin(ang)
        mode[c0:i1] = 1
    # keep bounded
    perp = amp_px * np.tanh(perp / amp_px)
    along = amp_px * np.tanh(along / amp_px)
    return SynTraj(t, perp.astype(np.float32), along.astype(np.float32), mode, n_frames)


def render_frames(truth, traj: SynTraj, add_noise=True, rate=LINE_RATE):
    """Render synthetic raster frames by sampling truth at the per-column gaze.
    frame f, column c (time t_{f,c}) samples truth over a vertical line offset by
    (perp, along) at that column's time. Measured per-rate noise added per column.
    """
    Ht, Wt = truth.shape
    oy = (Ht - H) // 2; ox = (Wt - W) // 2
    n = traj.n_frames
    frames = np.empty((n, H, W), np.float32)
    base_rows = np.arange(H)[:, None]                  # (H,1)
    rng = np.random.default_rng(12345)
    for f in range(n):
        sl = slice(f * W, (f + 1) * W)
        gp = traj.perp[sl]; ga = traj.along[sl]        # (W,)
        rowc = oy + base_rows + gp[None, :]            # (H,W)
        colc = ox + np.broadcast_to(np.arange(W)[None, :] + ga[None, :], (H, W))  # (H,W)
        fr = map_coordinates(truth, [rowc.ravel(), colc.ravel()], order=1,
                             mode="nearest").reshape(H, W).astype(np.float32)
        if add_noise:
            sigma = noisemod.noise_sigma(fr.ravel(), rate, line_rate=LINE_RATE)
            fr = fr + rng.standard_normal((H, W)).astype(np.float32) * sigma
        frames[f] = data._deband(fr)
    return frames, oy, ox


def track_to_truth(frames, truth, oy, ox, S, pad=48):
    """Register each strip to the (perfect) truth reference -> absolute gaze.
    Returns (t, perp_px, along_px, q) per strip."""
    import cv2
    n = len(frames)
    Tn = (truth - truth.mean()) / (truth.std() + 1e-9)
    nstrip = W // S
    T, PE, AL, Q = [], [], [], []
    for f in range(n):
        cur = frames[f]
        for s in range(nstrip):
            c = s * S
            strip = rt._nz(cur[:, c:c + S])
            top = oy - pad; left = ox + c - pad
            t0 = max(0, top); l0 = max(0, left)
            t1 = min(truth.shape[0], top + H + 2 * pad); l1 = min(truth.shape[1], left + S + 2 * pad)
            if t1 - t0 < H or l1 - l0 < S:
                continue
            roi = Tn[t0:t1, l0:l1]
            r = cv2.matchTemplate(roi, strip, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            sx = rt._parabolic(r[loc[1], :], loc[0]); sy = rt._parabolic(r[:, loc[0]], loc[1])
            perp = (t0 + sy) - oy
            along = (l0 + sx) - (ox + c)
            T.append(f / FPS + (c + S / 2) / W / FPS)
            PE.append(perp); AL.append(along); Q.append(mx)
    return (np.asarray(T), np.asarray(PE, float), np.asarray(AL, float), np.asarray(Q, float))


def _true_at(traj: SynTraj, t):
    idx = np.clip((t * LINE_RATE).astype(int), 0, len(traj.perp) - 1)
    return traj.perp[idx], traj.along[idx], traj.mode[idx]


@dataclass
class CertRow:
    S: int
    rate: float
    rms_perp_fix: float
    rms_along_fix: float
    rms_perp_sac: float
    rms_along_sac: float
    lock: float


def certify(S_list=(32, 16, 8, 4, 2, 1), ampx_perp=0.403, ampx_along=0.584,
            n_frames=180, seed=0):
    truth, _ = make_truth()
    Ht, Wt = truth.shape
    amp_px = float(min(70, (Ht - H) // 2 - 8, (Wt - W) // 2 - 8))
    traj = gen_trajectory(n_frames=n_frames, amp_px=amp_px, seed=seed)
    print(f"[synth] truth {truth.shape} amp_px={amp_px:.0f} n_frames={n_frames}", flush=True)
    rows = []
    # re-render noise per rate so the per-rate SNR is faithful
    for S in S_list:
        rate = (W // S) * FPS
        frames, oy, ox = render_frames(truth, traj, add_noise=True, rate=rate)
        t, pe, al, q = track_to_truth(frames, truth, oy, ox, S)
        tp, ta, md = _true_at(traj, t)
        lock = float((q > 0.3).mean())
        good = q > 0.3
        ep = (pe - tp); ea = (al - ta)
        fix = good & (md == 0); sac = good & (md == 1)

        def rms(x, m, scl):
            return float(np.sqrt(np.mean(x[m] ** 2)) * scl) if m.sum() > 10 else np.nan
        rows.append(CertRow(S, rate, rms(ep, fix, ampx_perp), rms(ea, fix, ampx_along),
                            rms(ep, sac, ampx_perp), rms(ea, sac, ampx_along), lock))
        print(f"S={S:>2} {rate:6.0f}Hz lock={lock*100:3.0f}%  "
              f"FIX perp={rows[-1].rms_perp_fix:.2f}' along={rows[-1].rms_along_fix:.2f}'  "
              f"SAC perp={rows[-1].rms_perp_sac:.2f}' along={rows[-1].rms_along_sac:.2f}'", flush=True)
    return rows


def write_md(rows, ampx_perp, ampx_along, path=os.path.join(RESULTS, "raster_synth_certified.md")):
    L = ["# Certified strip-tracking accuracy vs rate (labeled synthetic)", ""]
    L.append(f"Synthetic raster rendered from a clean test1 retina + known 2D gaze at the real "
             f"per-column timing with measured per-rate noise; recovered by strip registration "
             f"against the perfect reference. px->arcmin = {ampx_perp:.3f}'/px perp, "
             f"{ampx_along:.3f}'/px along (measured vs the 32.5 Hz tracker).")
    L.append("")
    L.append("| S | rate (Hz) | lock % | FIX perp (') | FIX along (') | SAC perp (') | SAC along (') |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r.S} | {r.rate:.0f} | {r.lock*100:.0f} | {r.rms_perp_fix:.2f} | {r.rms_along_fix:.2f} | "
                 f"{r.rms_perp_sac:.2f} | {r.rms_along_sac:.2f} |")
    L.append("")
    L.append("FIX = fixation/pursuit, SAC = through-saccade. perp = vertical (fast axis), "
             "along = horizontal (slow axis). RMS error vs the known label.")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


def figure(rows, path=os.path.join(RESULTS, "raster_synth_certified.png")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rates = [r.rate for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(rates, [r.rms_perp_fix for r in rows], "o-", color="C3", label="perp (vert)")
    ax[0].plot(rates, [r.rms_along_fix for r in rows], "o-", color="C0", label="along (horiz)")
    ax[0].axhline(6, color="0.5", ls="--", lw=1, label="0.1 deg")
    ax[0].axvline(1000, color="k", ls=":", lw=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].grid(alpha=.3, which="both")
    ax[0].set_title("Fixation/pursuit accuracy (certified)"); ax[0].set_xlabel("rate (Hz)")
    ax[0].set_ylabel("RMS error (' arcmin)"); ax[0].legend()
    ax[1].plot(rates, [r.rms_perp_sac for r in rows], "o-", color="C3", label="perp (vert)")
    ax[1].plot(rates, [r.rms_along_sac for r in rows], "o-", color="C0", label="along (horiz)")
    ax[1].axhline(6, color="0.5", ls="--", lw=1, label="0.1 deg")
    ax[1].axvline(1000, color="k", ls=":", lw=1)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].grid(alpha=.3, which="both")
    ax[1].set_title("Through-saccade accuracy (certified)"); ax[1].set_xlabel("rate (Hz)")
    ax[1].set_ylabel("RMS error (' arcmin)"); ax[1].legend()
    fig.suptitle("Certified 2D gaze accuracy vs strip rate — labeled synthetic raster", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--S", type=int, nargs="+", default=[32, 16, 8, 4, 2, 1])
    p.add_argument("--ampx-perp", type=float, default=0.403)
    p.add_argument("--ampx-along", type=float, default=0.584)
    p.add_argument("--n-frames", type=int, default=180)
    a = p.parse_args()
    rows = certify(tuple(a.S), ampx_perp=a.ampx_perp, ampx_along=a.ampx_along, n_frames=a.n_frames)
    fp = figure(rows); mp = write_md(rows, a.ampx_perp, a.ampx_along)
    print(f"wrote {fp}\nwrote {mp}")


if __name__ == "__main__":
    main()
