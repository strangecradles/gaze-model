"""slo_recon_hires.py — high-resolution motion-corrected SLO reconstruction of
the test1 retina, rebuilt with the preprocessing + registration steps that the
register-and-average SOTA (Roorda/Azimipour) relies on but the first
DPF-trace pass (`slo_recon_dpf.py`) was missing.

WHY the first pass ghosted / blurred (and AO-SLO doesn't):
  * placement came from the INCREMENTAL chain (cumulative registration) -> drift
    -> frames mis-placed by ~1 px -> doubled vessels (ghosting);
  * frames were averaged RAW -> per-frame scan banding + illumination falloff
    accumulate as haze;
  * no frame-quality gating -> blinks / low-contrast / distorted frames blur it;
  * a 258 px gaze drift over N=100 -> uneven, shallow averaging at the edges.

This module fixes all four, the way AO-SLO register-and-average does:
  1. PER-FRAME PREPROCESS  : de-band (flat-field) + CLAHE local contrast, so
     every frame is flat and structure-dominated before it is combined.
  2. IMAGE-BASED REGISTRATION to an ITERATIVELY-REFINED reference (sub-pixel
     phase correlation; optional per-strip refinement) — drift-free, so vessels
     stack exactly (no ghosting). NOT the gaze trace (the gaze trace is tuned
     for gaze accuracy, not pixel-exact image registration).
  3. QUALITY GATING        : reject blinks (low mean energy) and frames whose
     registration NCC to the reference is poor; weight survivors by NCC.
  4. DEEP UNIFORM STACK     : keep frames that overlap the reference patch
     (|shift| small) so every output pixel is a deep average -> max SNR, like
     the AO-SLO 12-frame stack.
  5. SUPER-RESOLUTION       : shift-and-add onto a U x finer grid. Eye motion
     gives sub-pixel sample diversity, so the average resolves finer than one
     native pixel (drizzle / SR).
  6. DECONVOLUTION          : a few Richardson-Lucy iterations (or unsharp) on
     the de-noised average recovers resolution lost to the SLO PSF.

We have SLO, not AO-SLO: no adaptive optics, so there are no cones to resolve
and the FOV is wide; "SOTA" here means a clean, ghost-free, deconvolved
register-and-average of conventional SLO, not cellular resolution.

Run:  python slo_recon_hires.py
      python slo_recon_hires.py --U 2 --deconv rl --rl-iters 8 --max-shift 110
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from skimage.registration import phase_cross_correlation

import khz2d
import test1_image_recon as t1

RESULTS = t1.RESULTS
CACHE = t1.CACHE

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ---------------------------------------------------------------------------
# Per-frame preprocessing
# ---------------------------------------------------------------------------

def deband(g):
    """Remove per-row then per-col mean (used only to score structure + for the
    blink gate — NOT for registration: de-banding removes the large-scale
    structure that drives frame-to-frame correlation, dropping NCC ~0.5->0.3)."""
    g = g - g.mean(1, keepdims=True)
    g = g - g.mean(0, keepdims=True)
    return g


def zs(x):
    m = np.isfinite(x)
    if m.sum() < 4:
        return np.zeros_like(x)
    return (x - x[m].mean()) / (x[m].std() + 1e-9)


def sharpness(img):
    a = np.asarray(img, np.float64)
    gy, gx = np.gradient(a)
    return float(np.mean(gx ** 2 + gy ** 2))


# ---------------------------------------------------------------------------
# Read the window
# ---------------------------------------------------------------------------

def read_window(t_hi=20.05):
    """Return raw float frames, their z-scored registration representation, the
    per-frame de-banded energy (for the blink gate), fps and chain. Registration
    is on RAW (z-scored) frames because that is where NCC is highest (~0.5)."""
    ch = khz2d.chain(); fps = float(ch["fps"])
    raw, reg, energy = {}, {}, {}
    for f, r in khz2d._read_frames():
        if f / fps > t_hi:
            break
        if not ch["ok"][f] or f < 1:
            continue
        rf = r.astype(np.float32)
        raw[f] = rf
        reg[f] = zs(rf)
        energy[f] = float(np.mean(np.abs(deband(rf))))
    return raw, reg, energy, fps, ch


# ---------------------------------------------------------------------------
# Registration + supersampled register-and-average
# ---------------------------------------------------------------------------

def _ncc(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 1000:
        return np.nan
    aa = a[m] - a[m].mean(); bb = b[m] - b[m].mean()
    d = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float((aa * bb).sum() / d) if d > 0 else np.nan


def register_all(reg_imgs, ref_eq, max_shift, upsample=10):
    """Sub-pixel global shift of every (z-scored) frame to ref_eq + aligned NCC.
    Returns dict f -> (sy, sx, ncc)."""
    out = {}
    for f in sorted(reg_imgs):
        shift, _, _ = phase_cross_correlation(ref_eq, reg_imgs[f],
                                              upsample_factor=upsample,
                                              normalization=None)
        sy, sx = float(shift[0]), float(shift[1])
        if abs(sy) > max_shift or abs(sx) > max_shift:
            out[f] = (sy, sx, np.nan); continue
        al = ndi.shift(reg_imgs[f], (sy, sx), order=1, mode="constant",
                       cval=np.nan)
        out[f] = (sy, sx, _ncc(ref_eq, al))
    return out


def dense_warp(reg_img, ref_eq, sy, sx, tile=140, step=70, max_res=9.0,
               smooth=45.0):
    """Dense 2D residual displacement field (Dy, Dx) of a globally-aligned frame
    vs the reference, by block matching on a regular tile grid -> cubic-resized
    to full resolution -> smoothed. Unlike 1D vertical-strip registration this
    captures torsion + field warp (shift varying in BOTH axes), which is what
    leaves wavy/doubled fine vessels in the macula. Returns native (Dy, Dx)
    float32 fields = the extra shift that aligns the frame to the reference."""
    H, W = ref_eq.shape
    al = ndi.shift(reg_img, (sy, sx), order=1, mode="constant", cval=0.0)
    r0s = list(range(0, max(1, H - tile + 1), step))
    c0s = list(range(0, max(1, W - tile + 1), step))
    Dy_c = np.zeros((len(r0s), len(c0s)), np.float32)
    Dx_c = np.zeros((len(r0s), len(c0s)), np.float32)
    for i, r0 in enumerate(r0s):
        for j, c0 in enumerate(c0s):
            a = al[r0:r0 + tile, c0:c0 + tile]
            b = ref_eq[r0:r0 + tile, c0:c0 + tile]
            if a.std() < 1e-3 or b.std() < 1e-3:
                continue
            sh, _, _ = phase_cross_correlation(b, a, upsample_factor=10,
                                               normalization=None)
            if abs(sh[0]) <= max_res and abs(sh[1]) <= max_res:
                Dy_c[i, j] = sh[0]; Dx_c[i, j] = sh[1]
    Dy = cv2.resize(Dy_c, (W, H), interpolation=cv2.INTER_CUBIC)
    Dx = cv2.resize(Dx_c, (W, H), interpolation=cv2.INTER_CUBIC)
    Dy = ndi.gaussian_filter(Dy, smooth); Dx = ndi.gaussian_filter(Dx, smooth)
    return Dy.astype(np.float32), Dx.astype(np.float32)


def local_contrast(img, k=9):
    """Per-pixel local std (signal-presence map). Flat FOV-dropout regions ->
    near 0, structured retina -> high. Normalized by its own median."""
    a = img.astype(np.float64)
    m = cv2.boxFilter(a, -1, (k, k))
    v = np.sqrt(np.maximum(cv2.boxFilter(a * a, -1, (k, k)) - m * m, 0.0))
    med = np.median(v[v > 0]) if (v > 0).any() else 1.0
    return np.clip(v / (med + 1e-9), 0.0, 3.0)


def accumulate(src, reg, ref_shape, keep, U=1, weight_pow=1.0, warp_fn=None,
               qmap=None):
    """Shift-and-add the `src[f]` frames onto a U x finer canvas using the
    registration `reg[f]=(sy,sx,ncc)`, for f in `keep`. NCC^weight_pow weighting.
    If `warp_fn(f)` returns a dense native (Dy, Dx) field, apply it on top of the
    global shift (non-rigid alignment, SR-preserving). If `qmap[f]` is given,
    weight every pixel by that frame's local-contrast map (suppresses FOV-
    dropout / flat regions). Returns (composite, weight-sum, frame-coverage)."""
    H, W = ref_shape
    CH, CW = H * U, W * U
    acc = np.zeros((CH, CW)); cnt = np.zeros((CH, CW)); ncov = np.zeros((CH, CW))
    Rg, Cg = np.meshgrid(np.arange(CH), np.arange(CW), indexing="ij")
    rbase = (Rg / float(U)).ravel(); cbase = (Cg / float(U)).ravel()
    for f in keep:
        sy, sx, ncc = reg[f][:3]
        w = max(ncc, 0.0) ** weight_pow
        if w <= 0:
            continue
        r = rbase - sy; c = cbase - sx
        if warp_fn is not None:
            DyDx = warp_fn(f)
            if DyDx is not None:
                Dy, Dx = DyDx
                rr = np.clip(rbase, 0, H - 1); cc = np.clip(cbase, 0, W - 1)
                r = r - map_coordinates(Dy, [rr, cc], order=1, mode="nearest")
                c = c - map_coordinates(Dx, [rr, cc], order=1, mode="nearest")
        vals = map_coordinates(src[f], [r, c], order=3, mode="constant",
                               cval=0.0).reshape(CH, CW)
        inb = ((r >= 0) & (r <= H - 1) & (c >= 0) & (c <= W - 1)).reshape(CH, CW)
        wpix = np.full(CH * CW, w)
        if qmap is not None and f in qmap:
            wpix = wpix * map_coordinates(qmap[f], [r, c], order=1,
                                          mode="constant", cval=0.0)
        wpix = wpix.reshape(CH, CW)
        acc[inb] += vals[inb] * wpix[inb]
        cnt[inb] += wpix[inb]
        ncov[inb] += 1.0
    comp = np.where(cnt > 0, acc / np.maximum(cnt, 1e-9), 0.0)
    return comp, cnt, ncov


# ---------------------------------------------------------------------------
# Finishing (gentle, natural — NOT the over-processed chain)
# ---------------------------------------------------------------------------

def _norm_blur(img, m, sigma):
    """Coverage-normalized Gaussian blur (ignores uncovered pixels)."""
    a = np.where(m, img, 0.0).astype(np.float64)
    num = cv2.GaussianBlur(a, (0, 0), sigma)
    den = cv2.GaussianBlur(m.astype(np.float64), (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def fix_scan_line(img, m, sigma=9.0):
    """Remove the horizontal fast-axis scan line + faint row banding via a gentle
    high-pass on the per-row median DC profile (subtract only the row-to-row mean
    deviation, never structure). Robust median -> a single horizontal vessel
    barely registers, so this is structure-preserving."""
    a = img.astype(np.float64).copy()
    rm = np.array([np.median(a[i, m[i]]) if m[i].any() else np.nan
                   for i in range(a.shape[0])])
    rm = khz2d.fill_nan(rm)
    hp = rm - ndi.gaussian_filter1d(rm, sigma, mode="nearest")
    return a - hp[:, None]


def finish(comp, ncov, scale=1, flatten_sigma=58.0, unsharp=0.7,
           unsharp_sigma=3.0, p=(1.0, 99.6), gamma=1.0):
    """Clean, natural finishing — the OPPOSITE of the over-processed chain that
    crunched the earlier renders. Homomorphic illumination flattening (divide by
    a coverage-normalized smooth background -> kills the hotspot/vignette while
    staying smooth), scan-line fix, a single MILD large-radius unsharp to lift
    vessels (no CLAHE, no deconvolution, no destripe), then a gentle global
    stretch. Returns uint8 with uncovered pixels black."""
    m = ncov > 0
    bg = _norm_blur(comp, m, flatten_sigma * scale)
    flat = comp / np.maximum(bg, 1e-3)
    flat = fix_scan_line(flat, m)
    if unsharp > 0:
        blur = cv2.GaussianBlur(flat, (0, 0), unsharp_sigma * scale)
        flat = flat + unsharp * (flat - blur)
    lo, hi = np.percentile(flat[m], p)
    x = np.clip((flat - lo) / (hi - lo + 1e-9), 0, 1)
    if gamma != 1.0:
        x = x ** gamma
    u8 = (x * 255).astype(np.uint8)
    u8[~m] = 0
    return u8


def stretch(a, m=None, p=(1.0, 99.0)):
    a = np.asarray(a, np.float64)
    mm = np.ones(a.shape, bool) if m is None else m
    lo, hi = np.percentile(a[mm], p)
    x = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    u8 = (x * 255).astype(np.uint8)
    u8[~mm] = 0
    return u8


def crop_cov(img, cnt, frac=0.7):
    keep = cnt >= frac * cnt.max()
    ys, xs = np.where(keep)
    if len(ys) == 0:
        return img, cnt
    r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return img[r0:r1, c0:c1], cnt[r0:r1, c0:c1]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-hi", type=float, default=20.05)
    ap.add_argument("--U", type=int, default=2, help="super-resolution factor")
    ap.add_argument("--max-shift", type=float, default=110.0,
                    help="keep frames within this many px of the reference patch")
    ap.add_argument("--ncc-thr", type=float, default=0.45)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--warp", choices=["dense", "none"], default="none",
                    help="none = global-rigid registration only (clean, default); "
                         "dense = 2D block-matching warp (can over-warp noisy "
                         "low-contrast regions -> avoid unless motion is large)")
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--qweight", type=int, default=0,
                    help="1 = per-pixel local-contrast weighting (suppress "
                         "FOV-dropout regions); 0 = plain average (cleaner)")
    ap.add_argument("--unsharp", type=float, default=0.7,
                    help="mild unsharp amount for the finish (0 = none)")
    ap.add_argument("--flatten-sigma", type=float, default=58.0,
                    help="homomorphic illumination-flattening radius (px)")
    args = ap.parse_args()

    print("reading window ...")
    raw, regimg, energy, fps, ch = read_window(args.t_hi)
    frames = sorted(raw)
    print(f"  {len(frames)} ok frames in 0-{args.t_hi:.0f}s")

    # blink / low-energy rejection
    en = np.array([energy[f] for f in frames]); med = np.median(en)
    good = [f for f in frames if energy[f] > 0.5 * med]
    print(f"  {len(good)}/{len(frames)} frames pass blink/energy gate")

    # reference = most structurally-sharp good frame in the early dwell
    early = [f for f in good if f <= frames[0] + 80] or good
    ref0 = max(early, key=lambda f: sharpness(deband(raw[f])))
    ref_eq = regimg[ref0]; ref_shape = ref_eq.shape
    print(f"  initial reference = frame {ref0} (sharpest in dwell)")

    # iterative reference refinement (register -> average -> re-register), all
    # on RAW z-scored frames (NCC ~0.5; CLAHE/deband would crater it).
    regmap = None; keep = [ref0]
    for p in range(args.passes):
        t0 = time.time()
        regmap = register_all(regimg, ref_eq, args.max_shift)
        nccs = np.array([regmap[f][2] for f in good])
        keep = [f for f in good
                if np.isfinite(regmap[f][2]) and regmap[f][2] >= args.ncc_thr]
        print(f"  [pass {p}] registered {len(good)} -> kept {len(keep)} "
              f"(ncc>={args.ncc_thr}; med ncc {np.nanmedian(nccs):.2f}) "
              f"({time.time()-t0:.0f}s)")
        comp1, _, _ = accumulate(regimg, regmap, ref_shape, keep, U=1)
        ref_eq = zs(comp1)

    # dense non-rigid refinement vs the final reference (removes torsion / field
    # warp -> the wavy/doubled fine vessels). Cached per frame.
    warp_fn = None
    if args.warp == "dense":
        t0 = time.time()
        wcache = {}

        def warp_fn(f):
            if f not in wcache:
                wcache[f] = dense_warp(regimg[f], ref_eq, regmap[f][0],
                                       regmap[f][1], tile=args.tile,
                                       step=args.tile // 2)
            return wcache[f]
        for f in keep:           # warm the cache + report magnitude
            warp_fn(f)
        amp = np.median([np.nanmax(np.abs(wcache[f][1])) for f in keep
                         if wcache[f] is not None])
        print(f"  dense warp on {len(keep)} frames "
              f"(median max |Dx| {amp:.1f}px) ({time.time()-t0:.0f}s)")

    # final super-resolved accumulation of the RAW frames
    qmap = {f: local_contrast(raw[f]) for f in keep} if args.qweight else None
    comp, cnt, ncov = accumulate(raw, regmap, ref_shape, keep, U=args.U,
                                 warp_fn=warp_fn, qmap=qmap)
    comp, ncov = crop_cov(comp, ncov, frac=0.5)   # crop on GEOMETRIC coverage
    th = int(round(0.03 * comp.shape[0]))          # drop thin top-of-sweep comb
    comp, ncov = comp[th:], ncov[th:]
    print(f"  final composite {comp.shape} at U={args.U}; "
          f"median depth {np.median(ncov[ncov>0]):.0f} frames/px")

    # clean, natural finishing (homomorphic flat-field + scan-line fix + mild
    # unsharp). The registered deep average is already smooth/sharp; we add the
    # LEAST processing needed, not the most.
    reg_only = stretch(comp, ncov > 0)                     # registered avg, raw
    hero = finish(comp, ncov, scale=args.U, unsharp=args.unsharp,
                  flatten_sigma=args.flatten_sigma)

    # comparators on the SAME frames: single frame + unregistered mean
    m_full = np.ones(ref_shape, bool)
    ones = np.ones(ref_shape)
    single = finish(raw[ref0], ones, scale=1, unsharp=args.unsharp)
    naive_acc = np.zeros(ref_shape)
    for f in keep:
        naive_acc += raw[f]
    naive = finish(naive_acc / max(len(keep), 1), ones, scale=1,
                   unsharp=args.unsharp)

    for name, im in (("single frame", single), ("naive mean", naive),
                     ("registered", reg_only), ("finished (hero)", hero)):
        print(f"  {name:16s} sharp={sharpness(im):.3e}")

    # --- diagnostic figure ---
    fig = plt.figure(figsize=(16, 8)); fig.patch.set_facecolor("black")
    gs = fig.add_gridspec(1, 4, wspace=0.03)
    panels = [(single, f"1 raw SLO frame\n(frame {ref0})"),
              (naive, f"unregistered mean\n({len(keep)} frames -> ghosted)"),
              (reg_only, f"registered average\n({len(keep)} frames, drift-free)"),
              (hero, f"+ flat-field + mild unsharp\n({args.U}x, final)")]
    for i, (im, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, color="w", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("SLO register-and-average (test1) — drift-free image "
                 "registration + deep averaging + gentle natural finishing",
                 color="w", fontsize=13, fontweight="bold")
    p1 = os.path.join(RESULTS, "slo_recon_hires.png")
    fig.savefig(p1, dpi=140, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  wrote {p1}")

    # --- hero ---
    p2 = os.path.join(RESULTS, "slo_recon_hires_hero.png")
    figh = plt.figure(figsize=(hero.shape[1] / 150, hero.shape[0] / 150), dpi=150)
    axh = figh.add_axes([0, 0, 1, 1]); axh.axis("off")
    axh.imshow(hero, cmap="gray", vmin=0, vmax=255)
    figh.savefig(p2, dpi=150, facecolor="black"); plt.close(figh)
    cv2.imwrite(p2.replace(".png", "_native.png"), hero)
    print(f"  wrote {p2} (+ _native.png)")

    np.savez(os.path.join(CACHE, "slo_recon_hires_summary.npz"),
             ref=np.int64(ref0), n_keep=np.int64(len(keep)), U=np.int64(args.U),
             max_shift=np.float64(args.max_shift), hero=hero, depth=cnt)
    return hero


if __name__ == "__main__":
    main()
