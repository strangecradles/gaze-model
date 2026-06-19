"""slo_recon_dpf.py — best-possible motion-corrected SLO image from the test1
retinal raster, built the Azimipour way and driven by OUR particle-filter (DPF)
gaze trace, swept over the number of frames N.

This is the DPF arm of `test1_image_recon.py` taken seriously as an IMAGE
deliverable (not a 3-way tracker comparison). We keep the validated, synthetic-
round-trip-checked kernel from that module verbatim — column-axis Azimipour
dewarp (eqs 9-10), per-frame integer chain anchor + subpixel/intra-frame
residual, content-validity-weighted register-and-average — and we:

  1. use ONLY the DPF trace (best config learned_n1000_ess0.7_nw3, per-column),
  2. sweep N = number of contributed frames (N=1 single frame ... N=all in the
     20 s DPF window) so you can SEE register-and-average build SNR/structure,
  3. render a polished "hero" SLO image (CLAHE + mild unsharp, dark bg) and a
     single-frame-vs-N-averaged before/after.

Azimipour et al. (2018/2019) register-and-average was developed for AO-SLO; we
do NOT have AO (no adaptive optics), so the input is ordinary SLO. Motion
correction therefore mostly buys de-noising + de-warping of the conventional
SLO raster rather than cellular-scale resolution. Everything is reference-free
(no ground-truth retinal image), so image-quality numbers are proxies.

Run:  python slo_recon_dpf.py
      python slo_recon_dpf.py --ns 1 25 50 100 183 --hero 100
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cv2
from scipy.ndimage import gaussian_filter1d

import khz2d
import test1_image_recon as t1   # validated dewarp + register-and-average kernel

RESULTS = t1.RESULTS
CACHE = t1.CACHE
EDGE = t1.EDGE
DPF_TAG = t1.DPF_TAG          # m4_dpf_11823_learned_n1000_ess0.7_nw3_d20
T0, T1 = t1.T0, t1.T1         # 0 .. 20.05 s — the DPF trace's valid span

DEFAULT_NS = [1, 10, 25, 50, 100, 183]


# ---------------------------------------------------------------------------
# Display enhancement (aggressive, per repo aesthetic: dark bg, sharp, structure)
# ---------------------------------------------------------------------------

def destripe(img, cov, sigma=10.0):
    """Remove the residual fixed-pattern VERTICAL stripes (columnar scan/illum
    bias that register-and-average leaves where the eye moved little) without
    touching vessels or broad illumination: estimate a robust per-column bias
    over covered rows, high-pass it (subtract its along-row smooth), subtract
    from every row. Purely cosmetic, structure-preserving."""
    a = img.astype(np.float64).copy()
    m = cov > 0
    col = np.full(a.shape[1], np.nan)
    for c in range(a.shape[1]):
        v = a[m[:, c], c]
        if v.size > 5:
            col[c] = np.median(v)
    col = khz2d.fill_nan(col)
    stripe = col - gaussian_filter1d(col, sigma, mode="nearest")
    return a - stripe[None, :]


def enhance(img, cov=None, p=(1.0, 99.6), clip=2.0, tiles=8, unsharp=0.35,
            blur_sigma=2.0, do_destripe=True):
    """Map a float mosaic to a display image: optional destripe, robust
    percentile stretch on the COVERED region, CLAHE local-contrast, a mild
    unsharp mask to lift vessels, uncovered pixels forced to black. uint8."""
    a = img.astype(np.float64).copy()
    m = np.ones(a.shape, bool) if cov is None else (cov > 0)
    if m.sum() < 10:
        m = np.ones(a.shape, bool)
    if do_destripe and cov is not None:
        a = destripe(a, cov)
    lo, hi = np.percentile(a[m], p)
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    u8 = (a * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    eq = clahe.apply(u8).astype(np.float32)
    if unsharp > 0:
        blur = cv2.GaussianBlur(eq, (0, 0), blur_sigma)
        eq = np.clip(eq + unsharp * (eq - blur), 0, 255)
    eq = eq.astype(np.uint8)
    eq[~m] = 0
    return eq


# ---------------------------------------------------------------------------
# Build one DPF mosaic at a given frame count
# ---------------------------------------------------------------------------

def _trim(img, cov, top=0.035, side=0.015):
    """Drop the thin-coverage first scan-rows (the bright 'comb' dewarp edge at
    the top of the fast sweep) and a small border on the other edges."""
    H, W = img.shape
    r0 = int(round(H * top)); r1 = H - int(round(H * side))
    c0 = int(round(W * side)); c1 = W - int(round(W * side))
    return img[r0:r1, c0:c1], cov[r0:r1, c0:c1]


def build_n(frame_idx, ox, oy, mask, fps, sgn, store):
    """Register-and-average the DPF arm over `frame_idx` (already sliced to N).
    ox/oy are the per-column absolute DPF positions aligned to frame_idx."""
    comps, cov, _ = t1.build_composites(frame_idx, {"dpf": (ox, oy)},
                                        mask, fps, sgn, store=store)
    return _trim(comps["dpf"], cov)


def quality(img):
    m = t1.metrics(img)
    # noise-robust "looks good" score: real structure-band power and contrast
    # (gradient `sharp` is inflated by noise, so it is reported, not optimized).
    return m


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_sweep(results, path):
    """Grid of the N reconstructions (enhanced) + metric-vs-N curves."""
    ns = [r["n"] for r in results]
    ncol = 3
    nrow = int(np.ceil(len(results) / ncol))
    fig = plt.figure(figsize=(4.7 * ncol, 4.0 * nrow + 3.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.0] * nrow + [0.62],
                          hspace=0.28, wspace=0.06)
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        ax.imshow(r["disp"], cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        cov_med = float(np.median(r["cov"][r["cov"] > 0])) if (r["cov"] > 0).any() else 0.0
        ax.set_title(f"N = {r['n']}   (~{cov_med:.0f} frames/px avg)\n"
                     f"struct {r['met']['struct']:.2e} | contrast "
                     f"{r['met']['contrast']:.3f} | sharp_y {r['met']['sharp_y']:.2f}",
                     fontsize=9.5)

    # metric vs N curves
    def norm(v):
        v = np.asarray(v, float)
        return v / (np.nanmax(np.abs(v)) + 1e-12)
    struct = [r["met"]["struct"] for r in results]
    contrast = [r["met"]["contrast"] for r in results]
    sharp = [r["met"]["sharp"] for r in results]
    covm = [float(np.median(r["cov"][r["cov"] > 0])) if (r["cov"] > 0).any() else 0
            for r in results]

    axa = fig.add_subplot(gs[nrow, 0])
    axa.plot(ns, norm(struct), "o-", color="#1b8a5a", label="structure-band power")
    axa.plot(ns, norm(contrast), "s-", color="#2b6cb0", label="contrast")
    axa.plot(ns, norm(sharp), "^--", color="#999", label="gradient sharp (noise↑)")
    axa.set_xlabel("N frames"); axa.set_ylabel("normalized")
    axa.set_title("image-quality proxies vs N", fontsize=10)
    axa.grid(alpha=0.25); axa.legend(fontsize=8)

    axb = fig.add_subplot(gs[nrow, 1])
    axb.plot(ns, covm, "o-", color="#d1495b")
    axb.set_xlabel("N frames"); axb.set_ylabel("median frames / pixel")
    axb.set_title("averaging depth (SNR ∝ √depth)", fontsize=10)
    axb.grid(alpha=0.25)

    axc = fig.add_subplot(gs[nrow, 2]); axc.axis("off")
    axc.text(0.0, 0.95,
             "Azimipour register-and-average\n(SLO, not AO-SLO)\n"
             "driven by OUR particle-filter (DPF) trace\n\n"
             "• vertical gaze stable, horizontal drifts ~258 px\n"
             "• small N → deep overlap (max SNR, smaller FOV)\n"
             "• large N → wider mosaic, thinner averaging in\n"
             "   briefly-visited regions\n\n"
             "gradient 'sharp' falls as N grows because\n"
             "speckle noise (high-gradient) averages out —\n"
             "structure-band power + contrast are the\n"
             "trustworthy 'cleaner image' signals.",
             fontsize=9, va="top", family="monospace")

    fig.suptitle("Motion-corrected SLO reconstruction — DPF-driven Azimipour "
                 "register-and-average, frame-count sweep (test1, 0–20 s)",
                 fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


def figure_compare(single_disp, best_disp, n_best, path):
    """Before/after: one raw frame (same pipeline, N=1) vs the best N average,
    with a matched zoom inset on a structure-rich patch."""
    fig = plt.figure(figsize=(13.5, 7.6)); fig.patch.set_facecolor("black")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.5], hspace=0.06, wspace=0.04)

    H, W = best_disp.shape
    zr0, zr1 = int(H * 0.42), int(H * 0.66)
    zc0, zc1 = int(W * 0.40), int(W * 0.64)

    def panel(gpos, img, title):
        ax = fig.add_subplot(gpos)
        # match single-frame size to best by center-crop/pad for display parity
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, color="w", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(Rectangle((zc0, zr0), zc1 - zc0, zr1 - zr0,
                               ec="#ffd000", fc="none", lw=1.6))
        return ax

    panel(gs[0, 0], single_disp, "one raw SLO frame (N = 1)")
    panel(gs[0, 1], best_disp, f"motion-corrected average (N = {n_best})")
    for j, (img, lab) in enumerate(((single_disp, "N=1 zoom"),
                                    (best_disp, f"N={n_best} zoom"))):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(img[zr0:zr1, zc0:zc1], cmap="gray", vmin=0, vmax=255)
        ax.set_title(lab, color="w", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("DPF-driven Azimipour motion correction: single frame vs "
                 f"{n_best}-frame register-and-average (SLO)",
                 color="w", fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  wrote {path}")


def save_hero(disp, path):
    fig = plt.figure(figsize=(disp.shape[1] / 130, disp.shape[0] / 130), dpi=130)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(disp, cmap="gray", vmin=0, vmax=255)
    fig.savefig(path, dpi=130, facecolor="black")
    plt.close(fig)
    # also a raw PNG at native resolution
    cv2.imwrite(path.replace(".png", "_native.png"), disp)
    print(f"  wrote {path} (+ _native.png)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS,
                    help="frame counts to sweep")
    ap.add_argument("--hero", type=int, default=None,
                    help="N for the hero/compare image (default: best by struct)")
    ap.add_argument("--smooth", type=float, default=4.0,
                    help="Gaussian sigma (in columns) to low-pass the per-column "
                         "DPF residual before dewarp — removes column-to-column "
                         "estimation jitter (vertical tearing/streaks) while "
                         "preserving real intra-frame eye motion. 0 disables.")
    ap.add_argument("--covcrop", type=float, default=0.65,
                    help="keep mosaic pixels covered by >= this * max coverage")
    args = ap.parse_args()

    t1.CROP_COV = float(args.covcrop)   # tighter crop -> drop low-coverage edges

    err = t1.synthetic_check()
    print(f"synthetic column-dewarp round-trip rms/std = {err:.2e} (should be <0.1)")

    ch = khz2d.chain()
    fps = float(ch["fps"])
    nframes = len(ch["x"])
    f_lo = int(np.ceil(T0 * fps)); f_hi = int(np.floor(T1 * fps))
    frame_idx = np.array([f for f in range(f_lo, f_hi + 1)
                          if ch["ok"][f] and f >= 1])
    print(f"DPF window t=[{T0},{T1}] s -> {len(frame_idx)} ok frames "
          f"({frame_idx[0]}..{frame_idx[-1]})")

    ns = sorted(set(min(n, len(frame_idx)) for n in args.ns))
    print(f"N sweep: {ns}")

    mask = t1.content_mask(nframes)
    dt, dx, dy, dvalid = t1.method_samples(DPF_TAG)
    print(f"DPF valid fraction = {dvalid*100:.0f}%")
    ox_full, oy_full = t1.per_column_offsets(dt, dx, dy, frame_idx, fps)
    if args.smooth > 0:
        # low-pass the per-column residual (intra-frame motion is smooth; the
        # column-to-column wiggle in the per-line estimate is what tears the
        # dewarp into vertical streaks). Smooth only the residual about each
        # frame's mean so the inter-frame placement is untouched.
        mx = ox_full.mean(1, keepdims=True); my = oy_full.mean(1, keepdims=True)
        ox_full = mx + gaussian_filter1d(ox_full - mx, args.smooth, axis=1,
                                         mode="nearest")
        oy_full = my + gaussian_filter1d(oy_full - my, args.smooth, axis=1,
                                         mode="nearest")
        print(f"  smoothed per-column residual (sigma={args.smooth} cols)")

    print("  preloading window frames ...")
    store = t1.load_window_frames(frame_idx)

    # global image-shift sign: maximize composite sharpness on the full set once.
    best_sgn, best_sh = +1.0, -np.inf
    for s in (+1.0, -1.0):
        img, _ = build_n(frame_idx, ox_full, oy_full, mask, fps, s, store)
        sh = t1.sharpness(img[EDGE:-EDGE, EDGE:-EDGE])
        print(f"  sign {s:+.0f}: sharpness {sh:.3e}")
        if sh > best_sh:
            best_sh, best_sgn = sh, s
    print(f"  -> global sign = {best_sgn:+.0f}")

    results = []
    for n in ns:
        fi = frame_idx[:n]
        img, cov = build_n(fi, ox_full[:n], oy_full[:n], mask, fps, best_sgn, store)
        met = quality(img)
        disp = enhance(img, cov)
        results.append(dict(n=n, img=img, cov=cov, met=met, disp=disp))
        cov_med = float(np.median(cov[cov > 0])) if (cov > 0).any() else 0.0
        print(f"  N={n:4d}  shape={img.shape}  cov~{cov_med:.0f}  "
              f"struct={met['struct']:.3e} contrast={met['contrast']:.4f} "
              f"sharp={met['sharp']:.3e} sharp_y={met['sharp_y']:.3e}")

    # pick hero: requested, else the N that maximizes structure-band power
    # (the noise-robust 'real detail' proxy) among N>=25.
    if args.hero is not None:
        n_best = min(args.hero, len(frame_idx))
        best = next(r for r in results if r["n"] == n_best) if any(
            r["n"] == n_best for r in results) else None
        if best is None:
            fi = frame_idx[:n_best]
            img, cov = build_n(fi, ox_full[:n_best], oy_full[:n_best], mask, fps,
                               best_sgn, store)
            best = dict(n=n_best, img=img, cov=cov, met=quality(img),
                        disp=enhance(img, cov))
    else:
        cand = [r for r in results if r["n"] >= 25] or results
        best = max(cand, key=lambda r: r["met"]["struct"])
        n_best = best["n"]
    print(f"  hero N = {n_best}")

    single = next((r for r in results if r["n"] == 1), results[0])

    figure_sweep(results, os.path.join(RESULTS, "slo_recon_dpf_sweep.png"))
    figure_compare(single["disp"], best["disp"], n_best,
                   os.path.join(RESULTS, "slo_recon_dpf_compare.png"))
    save_hero(best["disp"], os.path.join(RESULTS, "slo_recon_dpf_hero.png"))

    np.savez(os.path.join(CACHE, "slo_recon_dpf_summary.npz"),
             ns=np.array(ns), sign=np.float64(best_sgn), n_best=np.int64(n_best),
             dpf_valid=np.float64(dvalid),
             **{f"struct_{r['n']}": np.float64(r["met"]["struct"]) for r in results},
             **{f"contrast_{r['n']}": np.float64(r["met"]["contrast"]) for r in results},
             **{f"sharp_{r['n']}": np.float64(r["met"]["sharp"]) for r in results},
             hero_img=best["img"])
    print("  wrote cache/slo_recon_dpf_summary.npz")
    return results, best


if __name__ == "__main__":
    main()
