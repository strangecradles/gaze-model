"""test1_image_recon.py — three-way SLO image reconstruction on OUR OWN test1
pursuit raster, isolating eye-tracking quality.

GOAL: reconstruct a high-quality SLO image from the test1 raster
(`calibration/video_playback_test1_20260605_100051_SLO_0.mp4`, the SAME video
khz2d/data use for test1) and compare THREE reconstructions that differ ONLY in
the motion trace that drives the intra-frame dewarp:

  1. naive  — chain-only rigid registration (per-frame anchor, NO intra-frame
              residual). The "do nothing about within-frame eye motion" baseline.
  2. strip  — Azimipour-2018 register-and-average driven by the SOTA composite-
              reference strip-registration trace (sota_strip.sota_roorda, S=8,
              validity-matched threshold).
  3. dpf    — same reconstruction driven by OUR particle-filter trace
              (khz2d_methods.m4_dpf, best config learned_n1000_ess0.7_nw3).

Everything except the trace (dewarp interpolation, per-frame integer placement,
weighted averaging, crop, metrics) is byte-identical across the three arms, so
the comparison isolates tracking quality. This reuses the Azimipour dewarp +
register-and-average + reference-free metric kernel from aoslo_image_recon.py,
ADAPTED to the test1 raster geometry.

GEOMETRY (test1):
  frame shape (1000 rows, 808 cols). axis 0 (rows) = FAST axis = VERTICAL gaze;
  axis 1 (cols) = SLOW axis = HORIZONTAL gaze. A "line" (one fast sweep) is a
  COLUMN. So the per-line eye offset is per-COLUMN, the OPPOSITE axis mapping
  from the AOSLO frames in aoslo_image_recon.py (where a line is a row). The
  dewarp therefore applies an offset that depends on the column index c:
      corrected[r, c] = frame( row = r + s*ry[c], col = c + s*rx[c] )
  with (rx[c], ry[c]) the per-column intra-frame residual and s a single global
  sign convention chosen (once) by composite sharpness.

EYE TRACE -> PER-COLUMN OFFSET:
  Each method outputs (t, x_px, y_px, valid) in khz2d chain pixel coordinates
  (x = horizontal/slow, y = vertical/fast). For every frame we interpolate the
  method's valid samples onto the 808 per-column line times. The big inter-frame
  motion is carried by the per-frame integer chain anchor (shared by all arms);
  the intra-frame residual = pos - round(chain) is the small per-column warp that
  the trace supplies (zero for naive). This separation makes the inter-frame
  handling identical and isolates the intra-frame correction under test.

VALIDITY / FOV: test1 has the right-eye temporal FOV dropout. We weight every
  contributed column by a SHARED, method-independent content-validity mask
  (khz2d.fov_mask: line ok & qh>Q_FOV & contrast floor), so masked/invalid lines
  are downweighted identically in all three arms (the per-pixel coverage cnt is
  therefore identical across arms; only the pixel VALUES differ by the trace).

Run:  python test1_image_recon.py
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
from scipy.ndimage import map_coordinates

import data
import khz2d
import sota_strip

HERE = khz2d.HERE
CACHE = khz2d.CACHE
RESULTS = khz2d.RESULTS

# reconstruction window: the validated line-rate 20 s window used by the
# published SOTA comparison (results/sota_comparison.md). Both arms use their
# PER-COLUMN (S=1 / block=1) native traces here, so there is no interpolation-
# density mismatch — the only difference between arms is the trace itself. The
# full 70 s drifts >1 frame (x spans 815 px), so an all-frames mosaic overlaps
# only in a tiny patch; the 20 s window is the representative high-coverage span.
T0, T1 = 0.0, 20.05

# trace caches (the exact per-column caches behind results/sota_comparison.md)
DPF_TAG = "m4_dpf_11823_learned_n1000_ess0.7_nw3_d20"   # OUR best config, per-column
SOTA_TAG = "sota_s1_d20"                                 # SOTA per-column strip trace
SOTA_THR = 0.35      # validity-matched to DPF (~61%); see sota_comparison.md

CROP_COV = 0.5     # keep mosaic pixels covered by >= this * max coverage
EDGE = 16          # extra border crop (dewarp edge artifacts), identical all arms


# ---------------------------------------------------------------------------
# Shared content-validity mask (method-independent)
# ---------------------------------------------------------------------------

def content_mask(nframes):
    """(nframes, 808) bool: per-(frame,column) in-FOV / credible-line mask from
    the khz2d line cache. Identical for every arm (it is about image content, not
    the trace), so it cannot bias the SOTA-vs-DPF comparison."""
    z = np.load(khz2d.LINE_CACHE)
    fr = z["frame"]; col = z["col"]; ok = z["ok"]; qh = z["qh"]; con = z["con"]
    m = (ok.astype(bool) & (qh > khz2d.Q_FOV)
         & (con > khz2d.CONTRAST_FRAC * np.median(con)))
    grid = np.zeros((nframes, 808), bool)
    grid[fr.astype(int), col.astype(int)] = m
    return grid


# ---------------------------------------------------------------------------
# Trace -> per-frame per-column absolute position
# ---------------------------------------------------------------------------

def method_samples(tag, thr=None):
    """Return temporally-sorted valid (t, x, y) of a khz2d method cache. If `thr`
    is given, re-threshold validity from the stored max_ncc (used to validity-
    match the SOTA strip trace to the DPF valid fraction)."""
    c = khz2d.load_method(tag)
    if c is None:
        raise FileNotFoundError(f"missing trace cache khz2d_{tag}.npz")
    t = np.asarray(c["t"], float)
    x = np.asarray(c["x_px"], float); y = np.asarray(c["y_px"], float)
    if thr is not None and "max_ncc" in c:
        v = (c["max_ncc"] > thr)
    else:
        v = c["valid"].astype(bool)
    v = v & np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    o = np.argsort(t[v])
    return t[v][o], x[v][o], y[v][o], float(np.mean(v))


def per_column_offsets(samp_t, samp_x, samp_y, frame_idx, fps):
    """Interpolate a method's (t->x,y) onto the 808 per-column line times of each
    frame in `frame_idx`. Returns ox, oy each (len(frame_idx), 808) absolute
    positions (chain coords). Identical interpolation for every arm."""
    cols = (np.arange(808) + 0.5) / 808.0
    ts = (frame_idx[:, None] + cols[None, :]) / fps     # (F, 808) line times
    ox = np.interp(ts.ravel(), samp_t, samp_x).reshape(ts.shape)
    oy = np.interp(ts.ravel(), samp_t, samp_y).reshape(ts.shape)
    return ox, oy


# ---------------------------------------------------------------------------
# Dewarp kernel (Azimipour eqs 9-10) — COLUMN-indexed for the test1 raster
# ---------------------------------------------------------------------------

def dewarp_cols(frame, rx, ry, sgn, order=3):
    """corrected[r,c] = frame( row = r + sgn*ry[c], col = c + sgn*rx[c] ).

    rx, ry are PER-COLUMN residual shifts (length = #cols). This is the same
    cubic-spline resample as aoslo_image_recon.dewarp_frame, but the per-line
    offset varies along the COLUMN axis (axis 1) because a test1 line is a
    column, the opposite of the AOSLO row-line convention."""
    R, C = frame.shape
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    rows = rr + (sgn * np.asarray(ry))[None, :]
    cols = cc + (sgn * np.asarray(rx))[None, :]
    out = map_coordinates(frame, [rows.ravel(), cols.ravel()], order=order,
                          mode="nearest")
    return out.reshape(R, C)


# ---------------------------------------------------------------------------
# Three-arm register-and-average composite (single streaming pass over frames)
# ---------------------------------------------------------------------------

def synthetic_check():
    """Sanity: warp a clean texture by a known per-column trace, then dewarp with
    the negative trace; the recovered image must match the original interior.
    Confirms the column-axis dewarp + sign convention before touching real data."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((200, 160))
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    C = img.shape[1]
    # known per-column eye motion (smooth)
    rx = 3.0 * np.sin(np.arange(C) / 12.0)
    ry = 2.0 * np.cos(np.arange(C) / 18.0)
    warped = dewarp_cols(img, rx, ry, sgn=+1)          # apply motion
    recov = dewarp_cols(warped, rx, ry, sgn=-1)         # remove it
    a = img[20:-20, 20:-20]; b = recov[20:-20, 20:-20]
    err = float(np.sqrt(np.mean((a - b) ** 2)) / (a.std() + 1e-9))
    return err


def load_window_frames(frame_idx):
    """Decode the windowed frames once into a {f: raw float32} store so the
    sign test + effect-size sub-composites don't re-decode the mp4 each call."""
    fset = set(int(f) for f in frame_idx)
    store = {}
    for f, raw in khz2d._read_frames():
        if f in fset:
            store[f] = raw
        if f > max(fset):
            break
    return store


def build_composites(frame_idx, traces, mask, fps, sgn, store=None):
    """Single pass over the windowed frames building all arms at once.

    traces: dict arm -> (ox, oy) absolute per-column positions (F, 808). The
            naive arm is supplied chain-only positions so its inter-frame
            handling matches the others. `store` (optional) is a preloaded
            {f: raw} frame cache to avoid re-decoding the mp4.
    Returns dict arm -> composite (cropped identically) and the shared cnt/crop.
    """
    ch = khz2d.chain()
    cx = ch["x"]; cy = ch["y"]
    fa = frame_idx
    ax = np.round(cx[fa]).astype(int)      # per-frame integer anchor (shared)
    ay = np.round(cy[fa]).astype(int)
    Ax0, Ay0 = ax.min(), ay.min()
    H, W = 1000, 808
    CW = int(ax.max() - Ax0 + W + 2 * EDGE + 4)
    CH = int(ay.max() - Ay0 + H + 2 * EDGE + 4)
    arms = list(traces.keys())
    acc = {a: np.zeros((CH, CW), np.float64) for a in arms}
    cnt = np.zeros((CH, CW), np.float64)    # identical across arms by construction

    fset = set(int(f) for f in fa)
    pos = {int(f): i for i, f in enumerate(fa)}
    t0 = time.time()
    frame_iter = (((int(f), store[int(f)]) for f in fa) if store is not None
                  else khz2d._read_frames())
    for f, raw in frame_iter:
        if f not in fset:
            continue
        i = pos[f]
        w = mask[f].astype(np.float64)                  # (808,) column weights
        oy_anchor = ay[i] - Ay0 + EDGE
        ox_anchor = ax[i] - Ax0 + EDGE
        wmap = np.broadcast_to(w[None, :], (H, W))
        cnt[oy_anchor:oy_anchor + H, ox_anchor:ox_anchor + W] += wmap
        for a in arms:
            ox, oy = traces[a]
            rx = ox[i] - ax[i]                           # subpixel + intra residual
            ry = oy[i] - ay[i]
            corr = dewarp_cols(raw, rx, ry, sgn=sgn)
            acc[a][oy_anchor:oy_anchor + H, ox_anchor:ox_anchor + W] += corr * wmap
        if i % 50 == 0:
            print(f"    [compose] {i}/{len(fa)} ({time.time()-t0:.0f}s)")

    comp = {a: np.where(cnt > 0, acc[a] / np.maximum(cnt, 1e-9), 0.0) for a in arms}
    keep = cnt >= CROP_COV * cnt.max()
    ys, xs = np.where(keep)
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    r0 += EDGE; r1 -= EDGE; c0 += EDGE; c1 -= EDGE
    out = {a: comp[a][r0:r1, c0:c1] for a in arms}
    cov = cnt[r0:r1, c0:c1]
    return out, cov, (CH, CW)


# ---------------------------------------------------------------------------
# Reference-free image-quality metrics (same kernel as aoslo_image_recon.py)
# ---------------------------------------------------------------------------

def sharpness(img):
    gy, gx = np.gradient(img.astype(np.float64))
    return float(np.mean(gx ** 2 + gy ** 2))


def sharp_axes(img):
    """Gradient energy split by axis. gx = cross-column (axis 1) gradient: a
    noisy PER-COLUMN trace injects column-to-column jitter -> vertical streaks
    that inflate gx WITHOUT real detail. gy = along-column (axis 0). If an arm's
    'sharpness' edge lives entirely in gx, it is a streak artifact, not retinal
    detail. Reported as a guardrail."""
    gy, gx = np.gradient(img.astype(np.float64))
    return float(np.mean(gx ** 2)), float(np.mean(gy ** 2))


def rms_contrast(img):
    m = img.mean()
    return float(img.std() / (abs(m) + 1e-12))


def radial_psd(img):
    R, C = img.shape
    w = np.outer(np.hanning(R), np.hanning(C))
    x = (img - img.mean()) * w
    P = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    cy, cx = R // 2, C // 2
    yy, xx = np.indices((R, C))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rint = r.astype(int)
    tbin = np.bincount(rint.ravel(), P.ravel())
    nbin = np.bincount(rint.ravel())
    radial = tbin / np.maximum(nbin, 1)
    freq = np.arange(radial.size) / float(min(R, C))
    return freq, radial


def struct_power(img, flo=0.05, fhi=0.25):
    """Retinal-structure modulation: mean radial power in the vessel/structure
    band, normalized by total in-band power floor (reference-free)."""
    freq, radial = radial_psd(img)
    band = (freq >= flo) & (freq <= fhi)
    return float(np.mean(radial[band]))


def hf_frac(img, fcut=0.15):
    """Fraction of spectral power above fcut (high-frequency detail proxy)."""
    freq, radial = radial_psd(img)
    tot = np.sum(radial[freq > 0.01]) + 1e-12
    return float(np.sum(radial[freq > fcut]) / tot)


def metrics(img):
    a = img[EDGE:img.shape[0] - EDGE, EDGE:img.shape[1] - EDGE]
    sx, sy = sharp_axes(a)
    return dict(sharp=sharpness(a), contrast=rms_contrast(a),
                struct=struct_power(a), hf=hf_frac(a),
                sharp_x=sx, sharp_y=sy)


# ---------------------------------------------------------------------------
# Effect sizes via disjoint sub-composites
# ---------------------------------------------------------------------------

def _paired_stats(d):
    d = np.asarray(d, float); n = d.size
    mean = float(d.mean()); sd = float(d.std(ddof=1)) if n > 1 else 0.0
    dz = mean / sd if sd > 0 else np.inf
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(5000)])
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else np.inf
    try:
        from scipy.stats import t as tdist
        p = float(2 * tdist.sf(abs(t), n - 1))
    except Exception:
        p = float("nan")
    return dict(mean=mean, ci=ci, dz=float(dz), p=p, n=n)


def subcomposite_effects(frame_idx, traces, mask, fps, sgn, n_groups=8, store=None):
    """Split the window into n_groups disjoint frame groups; build a paired
    composite per arm per group and measure each metric. Returns per-arm metric
    arrays + paired effect sizes (dpf-strip, strip-naive, dpf-naive)."""
    keys = ("sharp", "contrast", "struct", "hf", "sharp_x", "sharp_y")
    groups = np.array_split(frame_idx, n_groups)
    arms = list(traces.keys())
    vals = {a: {k: [] for k in keys} for a in arms}
    for g in groups:
        comps, _, _ = build_composites(g, {a: (traces[a][0][_grp_idx(frame_idx, g)],
                                               traces[a][1][_grp_idx(frame_idx, g)])
                                            for a in arms}, mask, fps, sgn, store=store)
        for a in arms:
            mt = metrics(comps[a])
            for k in mt:
                vals[a][k].append(mt[k])
    for a in arms:
        for k in vals[a]:
            vals[a][k] = np.array(vals[a][k])
    eff = {}
    for k in keys:
        eff[k] = dict(
            dpf_strip=_paired_stats(vals["dpf"][k] - vals["strip"][k]),
            strip_naive=_paired_stats(vals["strip"][k] - vals["naive"][k]),
            dpf_naive=_paired_stats(vals["dpf"][k] - vals["naive"][k]),
        )
    return vals, eff


def _grp_idx(frame_idx, g):
    """Indices into the windowed arrays for the frames in group g."""
    pos = {int(f): i for i, f in enumerate(frame_idx)}
    return np.array([pos[int(f)] for f in g])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _imshow(ax, img, title):
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def _inset(ax, img, box, color):
    r0, r1, c0, c1 = box
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, ec=color, fc="none", lw=1.5))


def make_figure(comps, mets, eff, cov, path):
    arms = ["naive", "strip", "dpf"]
    titles = {"naive": "naive\n(chain-only, no intra-frame)",
              "strip": "Azimipour + SOTA strip trace",
              "dpf": "Azimipour + OUR DPF trace"}
    H, W = comps["naive"].shape
    # zoom box: central structure-rich region
    zr0, zr1 = int(H * 0.40), int(H * 0.62)
    zc0, zc1 = int(W * 0.40), int(W * 0.62)
    box = (zr0, zr1, zc0, zc1)

    fig = plt.figure(figsize=(15, 9)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.55, 0.7],
                          hspace=0.22, wspace=0.08)

    for i, a in enumerate(arms):
        ax = fig.add_subplot(gs[0, i])
        m = mets[a]
        _imshow(ax, comps[a], f"{titles[a]}\nsharp {m['sharp']:.2e} | "
                              f"struct {m['struct']:.2e} | HF {m['hf']:.3f}")
        _inset(ax, comps[a], box, "#ffd000")

    for i, a in enumerate(arms):
        ax = fig.add_subplot(gs[1, i])
        z = comps[a][zr0:zr1, zc0:zc1]
        vmin, vmax = np.percentile(z, [1, 99])
        ax.imshow(z, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{a} (zoom)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # metric bars
    barspec = [("sharp", "gradient sharpness"), ("struct", "structure-band power")]
    colors = {"naive": "#888888", "strip": "#d1495b", "dpf": "#1b8a5a"}
    for j, (k, lab) in enumerate(barspec):
        ax = fig.add_subplot(gs[2, j])
        vals = [mets[a][k] for a in arms]
        ax.bar([titles_short(a) for a in arms], vals,
               color=[colors[a] for a in arms])
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:.2e}" if v < 1 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(lab, fontsize=10); ax.grid(alpha=0.25, axis="y")
        ax.tick_params(axis="x", labelsize=9)

    # streak guard: cross-column (gx) vs along-column (gy) gradient energy.
    # a noisy per-column trace inflates gx (vertical streaks) without real detail.
    ax = fig.add_subplot(gs[2, 2])
    xpos = np.arange(len(arms)); ww = 0.38
    sx = [mets[a]["sharp_x"] for a in arms]; sy = [mets[a]["sharp_y"] for a in arms]
    ax.bar(xpos - ww / 2, sx, ww, label="gx (cross-column)", color="#c44")
    ax.bar(xpos + ww / 2, sy, ww, label="gy (along-column)", color="#48c")
    ax.set_xticks(xpos); ax.set_xticklabels([titles_short(a) for a in arms], fontsize=9)
    ax.set_title("streak guard: cross- vs along-column", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.25, axis="y")

    e = eff["sharp"]
    sub = (f"DPF-strip sharpness d={e['dpf_strip']['mean']:+.2e} "
           f"(dz={e['dpf_strip']['dz']:+.2f}, p={e['dpf_strip']['p']:.2g}); "
           f"strip-naive d={e['strip_naive']['mean']:+.2e} "
           f"(dz={e['strip_naive']['dz']:+.2f})")
    fig.suptitle("test1 SLO reconstruction (Azimipour register-and-average, "
                 f"{T0:.0f}-{T1:.0f} s window) — naive vs SOTA-strip vs OUR-DPF trace\n"
                 + sub, fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def titles_short(a):
    return {"naive": "naive", "strip": "SOTA", "dpf": "OURS"}[a]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngroups", type=int, default=8)
    args = ap.parse_args()

    err = synthetic_check()
    print(f"synthetic column-dewarp round-trip rms/std = {err:.2e} (should be <0.1)")

    ch = khz2d.chain()
    fps = float(ch["fps"])
    nframes = len(ch["x"])
    f_lo = int(np.ceil(T0 * fps)); f_hi = int(np.floor(T1 * fps))
    frame_idx = np.array([f for f in range(f_lo, f_hi + 1)
                          if ch["ok"][f] and f >= 1])
    print(f"window t=[{T0},{T1}] s -> frames {frame_idx[0]}..{frame_idx[-1]} "
          f"({len(frame_idx)} ok frames)")

    mask = content_mask(nframes)
    print(f"content-mask in-FOV fraction (window) = "
          f"{mask[frame_idx].mean()*100:.0f}%")

    # traces -> per-column absolute positions
    dt, dx, dy, dvalid = method_samples(DPF_TAG)
    st, sx, sy, svalid = method_samples(SOTA_TAG, thr=SOTA_THR)
    print(f"DPF valid={dvalid*100:.0f}%  SOTA(thr{SOTA_THR}) valid={svalid*100:.0f}%")
    dpf_ox, dpf_oy = per_column_offsets(dt, dx, dy, frame_idx, fps)
    strip_ox, strip_oy = per_column_offsets(st, sx, sy, frame_idx, fps)
    # naive: chain-only (per-frame anchor, no intra-frame residual)
    naive_ox = np.repeat(ch["x"][frame_idx][:, None], 808, axis=1)
    naive_oy = np.repeat(ch["y"][frame_idx][:, None], 808, axis=1)

    traces = {"naive": (naive_ox, naive_oy),
              "strip": (strip_ox, strip_oy),
              "dpf": (dpf_ox, dpf_oy)}

    print("  preloading window frames ...")
    store = load_window_frames(frame_idx)

    # global sign convention: pick the sign maximizing DPF composite sharpness,
    # apply identically to all arms (physical eye-pos -> image-shift sign).
    best_sgn, best_sh = +1, -np.inf
    for s in (+1.0, -1.0):
        c, _, _ = build_composites(frame_idx, {"dpf": traces["dpf"]}, mask, fps, s,
                                   store=store)
        sh = sharpness(c["dpf"][EDGE:-EDGE, EDGE:-EDGE])
        print(f"  sign {s:+.0f}: DPF sharpness {sh:.3e}")
        if sh > best_sh:
            best_sh, best_sgn = sh, s
    print(f"  -> global sign = {best_sgn:+.0f}")

    comps, cov, csz = build_composites(frame_idx, traces, mask, fps, best_sgn,
                                       store=store)
    mets = {a: metrics(comps[a]) for a in comps}
    print("\n  arm   | sharp     | sharp_x(cross-col) | sharp_y | contrast | struct    | HF-frac")
    for a in ("naive", "strip", "dpf"):
        m = mets[a]
        print(f"  {a:5s} | {m['sharp']:.3e} | {m['sharp_x']:.3e}          | "
              f"{m['sharp_y']:.3e} | {m['contrast']:.4f}  | "
              f"{m['struct']:.3e} | {m['hf']:.4f}")

    print(f"\n  effect sizes ({args.ngroups} disjoint sub-composites):")
    vals, eff = subcomposite_effects(frame_idx, traces, mask, fps, best_sgn,
                                     n_groups=args.ngroups, store=store)
    for k in ("sharp", "sharp_x", "sharp_y", "struct", "hf", "contrast"):
        for pair in ("dpf_strip", "strip_naive", "dpf_naive"):
            e = eff[k][pair]
            print(f"    {k:8s} {pair:11s} d={e['mean']:+.3e} "
                  f"dz={e['dz']:+.2f} CI[{e['ci'][0]:+.2e},{e['ci'][1]:+.2e}] "
                  f"p={e['p']:.2g}")

    make_figure(comps, mets, eff, cov, os.path.join(RESULTS, "test1_image_recon.png"))

    np.savez(os.path.join(CACHE, "test1_image_recon_summary.npz"),
             window=np.array([T0, T1]), sign=np.float64(best_sgn),
             nframes=np.int64(len(frame_idx)),
             dpf_valid=np.float64(dvalid), sota_valid=np.float64(svalid),
             fov_frac=np.float64(mask[frame_idx].mean()),
             comp_naive=comps["naive"], comp_strip=comps["strip"],
             comp_dpf=comps["dpf"], cov=cov,
             **{f"met_{a}_{k}": mets[a][k] for a in mets for k in mets[a]},
             **{f"eff_{k}_{pair}_{s}": eff[k][pair][s]
                for k in eff for pair in eff[k]
                for s in ("mean", "dz", "p")})
    print("  wrote cache/test1_image_recon_summary.npz")
    return comps, mets, eff


if __name__ == "__main__":
    main()
