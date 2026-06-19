"""aoslo_image_recon.py — image reconstruction / intra-frame motion correction
head-to-head: OUR particle-filter (DPF) eye trace vs the Azimipour-2018 strip-
registration trace, driving the SAME dewarp + averaging pipeline.

This builds on aoslo_headtohead.py (which recovers the eye traces for both
methods and ports the strip registration to py3). Here we take those traces and
run the paper's intra-frame motion-correction / reconstruction step, then ask:
does OUR tracking produce a MEANINGFULLY better motion-corrected image?

Reference (reconstruction math, equations cited inline):
  Azimipour, Migacz, Zawadzki, Werner, Jonnal, "Intraframe motion correction for
  raster-scanned adaptive optics images using strip-based cross-correlation lag
  biases," PLoS One 13(10):e0206052, 2018.
  Code/data: R. Jonnal, github.com/rjonnal/intraframe_motion_correction.

Generative model (create_simulated_images.py, verified bit-exact, err = 0):
  frame[f][idx, c] = bilinear(mosaic, row = my0 + idx + gy[f,idx],
                                      col = mx0 + gx[f,idx] + c)
  so each line is the clean mosaic sampled along a path warped by eye motion.

Reconstruction we run (faithful to the paper):
  * Per-frame, per-strip lags  s_x, s_y          (eqs 1-2; via aoslo_headtohead)
  * Single-reference paper pipeline (REAL):
      inter-row lag differences  s_hat = diff(s)  (eqs 3-4)
      outlier reject |s_hat|>2px -> NaN
      lag biases  delta_r = nanmean_over_frames(s_hat)   (eqs 5-6)
      integrate   x_hat_t = cumsum(delta)               (eqs 7-8)
      dewarp reference: corrected[r,c] = ref(r + y_hat_t[r], c + x_hat_t[r]) (eqs 9-10)
  * Register-and-average composite (SIMULATED, primary, has GT object):
      for each frame, dewarp by its per-line eye trace (x_hat = -gx_est,
      y_hat = -gy_est) onto the object grid, then average all frames.
      The ONLY thing that differs between the SOTA and OURS arms is the trace.

Interpolation: the paper uses scipy griddata(method='cubic'); on a regular grid
that is equivalent to a cubic-spline resample, so we use
scipy.ndimage.map_coordinates(order=3) (identical math, ~100x faster). Applied
identically to both arms, so it cannot favor either.

Run:  python aoslo_image_recon.py            # uses caches from aoslo_headtohead
      python aoslo_image_recon.py --rebuild  # recompute traces too
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.registration import phase_cross_correlation

import aoslo_headtohead as h2h

HERE = h2h.HERE
EXT = h2h.EXT
RESULTS = h2h.RESULTS
CACHE = h2h.CACHE
ARCMIN_PER_PX = h2h.ARCMIN_PER_PX
N_LINES = h2h.N_LINES
MX0 = h2h.MX0
MY0 = h2h.MY0


# ---------------------------------------------------------------------------
# Dewarp (eqs 9-10) — the shared reconstruction kernel
# ---------------------------------------------------------------------------

def dewarp_frame(frame, x_hat, y_hat, order=3):
    """Resample `frame` onto a stabilized grid (Azimipour eqs 9-10).

    corrected[r, c] = frame( row = r + y_hat[r], col = c + x_hat[r] )

    `x_hat`, `y_hat` are per-row STABILIZING shifts. For the generative model
    above, the motion is removed by x_hat = -gx_est, y_hat = -gy_est (the script
    plots -x_hat_t as the recovered eye trace, confirming this sign). order=3 is a
    cubic-spline resample, equivalent to the paper's griddata(method='cubic')."""
    R, C = frame.shape
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    rows = rr + np.asarray(y_hat)[:, None]
    cols = cc + np.asarray(x_hat)[:, None]
    out = map_coordinates(frame, [rows.ravel(), cols.ravel()], order=order,
                          mode="nearest")
    return out.reshape(R, C)


def lag_bias_reference(s_x, s_y, ref, max_inter_row=2.0, order=3):
    """Faithful port of demonstrate_registration.py eqs 3-10.

    s_x, s_y: (n_frames, n_rows) per-strip lags of every frame vs the reference.
    Returns (x_hat_t, y_hat_t, corrected_reference). The reference's own intra-
    frame motion is recovered as the across-frame mean of inter-row lag biases."""
    s_hat_x = np.diff(np.hstack((s_x[:, 0:1], s_x)), axis=1)        # eqs 3-4
    s_hat_y = np.diff(np.hstack((s_y[:, 0:1], s_y)), axis=1)
    s_hat_x[np.abs(s_hat_x) > max_inter_row] = np.nan              # outlier reject
    s_hat_y[np.abs(s_hat_y) > max_inter_row] = np.nan
    delta_x_r = np.nanmean(s_hat_x, axis=0)                        # eqs 5-6
    delta_y_r = np.nanmean(s_hat_y, axis=0)
    x_hat_t = np.cumsum(delta_x_r)                                 # eqs 7-8
    y_hat_t = np.cumsum(delta_y_r)
    corrected = dewarp_frame(ref, x_hat_t, y_hat_t, order=order)   # eqs 9-10
    return x_hat_t, y_hat_t, corrected


# ---------------------------------------------------------------------------
# Composite (register-and-average) — shared between arms
# ---------------------------------------------------------------------------

def composite_from_traces(frames, gx_est, gy_est, order=3):
    """Dewarp every frame by its own per-line eye trace then average.

    gx_est, gy_est: (F, R) recovered eye position per line. Stabilizing shift is
    the negative trace (x_hat = -gx_est). Returns the averaged composite (R, C)."""
    F, R, C = frames.shape
    acc = np.zeros((R, C))
    for f in range(F):
        acc += dewarp_frame(frames[f], -gx_est[f], -gy_est[f], order=order)
    return acc / F


# ---------------------------------------------------------------------------
# Image-quality metrics
# ---------------------------------------------------------------------------

def _align_to(ref, img, crop=12):
    """Remove an arbitrary rigid global offset (the eye-trace origin is free):
    estimate a sub-pixel shift via phase correlation, apply it, then crop the
    border. Applied IDENTICALLY to every composite, so it cannot bias the
    comparison."""
    shift, _, _ = phase_cross_correlation(ref, img, upsample_factor=20,
                                          normalization=None)
    R, C = img.shape
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    aligned = map_coordinates(img, [(rr - shift[0]).ravel(), (cc - shift[1]).ravel()],
                              order=3, mode="nearest").reshape(R, C)
    return aligned[crop:R - crop, crop:C - crop]


def radial_psd(img):
    """Radially-averaged power spectrum of a Hann-windowed, mean-removed image.
    Returns (freq cyc/px, power)."""
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
    freq = np.arange(radial.size) / float(R)      # cycles/pixel
    return freq, radial


def cone_peak(img, flo=0.10, fhi=0.45, f_ref=None):
    """Cone-mosaic spectral signature: peak in the radial PSD within the cone
    band. Returns (peak_freq cyc/px, prominence) where prominence = peak power /
    median band power (modulation depth of the cone ring). If `f_ref` given,
    evaluate prominence at that fixed frequency band instead of the argmax."""
    freq, radial = radial_psd(img)
    band = (freq >= flo) & (freq <= fhi)
    fb, pb = freq[band], radial[band]
    base = np.median(pb) + 1e-12
    if f_ref is None:
        i = int(np.argmax(pb))
        return float(fb[i]), float(pb[i] / base)
    j = int(np.argmin(np.abs(fb - f_ref)))
    return float(fb[j]), float(pb[j] / base)


def sharpness(img):
    """Gradient energy (mean squared Sobel-like gradient magnitude). Higher =
    sharper. Reference-free."""
    gy, gx = np.gradient(img.astype(np.float64))
    return float(np.mean(gx ** 2 + gy ** 2))


def rms_contrast(img):
    """RMS contrast = std / mean (Michelson-like, reference-free)."""
    m = img.mean()
    return float(img.std() / (abs(m) + 1e-12))


def gt_metrics(comp, obj, f_cone):
    """Full-reference metrics vs the clean object (simulated only)."""
    a = _align_to(obj, comp)
    b = obj[12:obj.shape[0] - 12, 12:obj.shape[1] - 12]
    dr = float(b.max() - b.min())
    return dict(
        psnr=float(peak_signal_noise_ratio(b, a, data_range=dr)),
        ssim=float(structural_similarity(b, a, data_range=dr)),
        cone_prom=cone_peak(a, f_ref=f_cone)[1],
        sharp=sharpness(a),
        contrast=rms_contrast(a),
    )


def ref_free_metrics(comp, f_cone):
    """Reference-free metrics (real frames, no GT)."""
    a = comp[12:comp.shape[0] - 12, 12:comp.shape[1] - 12]
    return dict(
        cone_prom=cone_peak(a, f_ref=f_cone)[1],
        sharp=sharpness(a),
        contrast=rms_contrast(a),
    )


# ---------------------------------------------------------------------------
# Effect sizes: split frames into groups, build paired sub-composites
# ---------------------------------------------------------------------------

def _paired_stats(d):
    """Mean, 95% bootstrap CI, Cohen's d_z, and paired-t p-value for a vector of
    paired differences d (OURS - SOTA per group)."""
    d = np.asarray(d, float)
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    dz = mean / sd if sd > 0 else np.inf
    # bootstrap CI of the mean
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(5000)])
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    # paired t -> two-sided p
    t = mean / (sd / np.sqrt(n)) if sd > 0 else np.inf
    try:
        from scipy.stats import t as tdist
        p = float(2 * tdist.sf(abs(t), n - 1))
    except Exception:
        p = float("nan")
    return dict(mean=mean, ci=ci, dz=float(dz), p=p, n=n)


def subcomposite_effect(frames, traces, obj, f_cone, n_groups=20):
    """Split frames into n_groups disjoint groups; for each group build a paired
    composite per method and measure PSNR/SSIM. Returns per-method metric arrays
    plus paired OURS-minus-SOTA effect sizes. `traces` maps method -> (gx,gy)."""
    F = frames.shape[0]
    idx = np.array_split(np.arange(F), n_groups)
    out = {m: dict(psnr=[], ssim=[]) for m in traces}
    for g in idx:
        sub = frames[g]
        for m, (gx, gy) in traces.items():
            c = composite_from_traces(sub, gx[g], gy[g])
            mt = gt_metrics(c, obj, f_cone)
            out[m]["psnr"].append(mt["psnr"]); out[m]["ssim"].append(mt["ssim"])
    for m in out:
        out[m]["psnr"] = np.array(out[m]["psnr"])
        out[m]["ssim"] = np.array(out[m]["ssim"])
    eff = {}
    if "ours" in out and "sota" in out:
        eff["psnr"] = _paired_stats(out["ours"]["psnr"] - out["sota"]["psnr"])
        eff["ssim"] = _paired_stats(out["ours"]["ssim"] - out["sota"]["ssim"])
    return out, eff


# ---------------------------------------------------------------------------
# Simulated reconstruction (primary: has GT object)
# ---------------------------------------------------------------------------

def run_simulated(rebuild=False, n_particles=250):
    print("\n=== SIMULATED reconstruction (register-and-average vs GT object) ===")
    frames, gx, gy, mosaic = h2h.load_simulated()
    obj = mosaic[MY0:MY0 + N_LINES, MX0:MX0 + N_LINES]            # motion_free

    sx, sy, scorr, sxi, syi = h2h.sota_simulated(frames, mosaic, rebuild=rebuild)
    dpf_gx, dpf_gy, dpf_ncc = h2h.run_dpf_simulated(
        frames, mosaic, n_particles=n_particles, rebuild=rebuild)

    # SOTA lag sign relative to GT (same convention as the head-to-head)
    acc = h2h.accuracy_table(sx, sy, dpf_gx, dpf_gy, gx, gy)
    ssx, ssy = acc["_sota_sign"]
    acc_i = h2h.accuracy_table(sxi, syi, dpf_gx, dpf_gy, gx, gy)
    six, siy = acc_i["_sota_sign"]

    # cone reference frequency from the clean object
    f_cone = cone_peak(obj[12:-12, 12:-12])[0]
    print(f"  cone-band peak frequency of object: {f_cone:.3f} cyc/px")

    print("  building composites (200 frames each) ...")
    comps = {
        "naive": frames.mean(axis=0),
        "sota_int": composite_from_traces(frames, six * sxi, siy * syi),
        "sota": composite_from_traces(frames, ssx * sx, ssy * sy),
        "ours": composite_from_traces(frames, dpf_gx, dpf_gy),
        "ideal_gt": composite_from_traces(frames, gx, gy),
    }
    metrics = {k: gt_metrics(v, obj, f_cone) for k, v in comps.items()}
    obj_metrics = dict(cone_prom=cone_peak(obj[12:-12, 12:-12], f_ref=f_cone)[1],
                       sharp=sharpness(obj[12:-12, 12:-12]),
                       contrast=rms_contrast(obj[12:-12, 12:-12]))

    print("\n  composite | PSNR(dB) | SSIM   | cone-prom | sharp    | contrast")
    for k in ["naive", "sota_int", "sota", "ours", "ideal_gt"]:
        m = metrics[k]
        print(f"  {k:9s} | {m['psnr']:7.3f}  | {m['ssim']:.4f} | "
              f"{m['cone_prom']:8.2f}  | {m['sharp']:.2e} | {m['contrast']:.4f}")
    print(f"  {'OBJECT':9s} |    inf   | 1.0000 | {obj_metrics['cone_prom']:8.2f}  | "
          f"{obj_metrics['sharp']:.2e} | {obj_metrics['contrast']:.4f}")

    print("\n  effect size (20 disjoint 10-frame sub-composites, OURS - SOTA):")
    groups, eff = subcomposite_effect(
        frames, {"sota": (ssx * sx, ssy * sy), "ours": (dpf_gx, dpf_gy)},
        obj, f_cone, n_groups=20)
    for mk in ("psnr", "ssim"):
        e = eff[mk]
        unit = "dB" if mk == "psnr" else ""
        print(f"    {mk.upper():4s}  d(OURS-SOTA) = {e['mean']:+.3f} {unit}  "
              f"95%CI [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]  "
              f"Cohen's dz={e['dz']:+.2f}  p={e['p']:.2e}  (n={e['n']})")

    return dict(obj=obj, comps=comps, metrics=metrics, obj_metrics=obj_metrics,
                f_cone=f_cone, groups=groups, eff=eff)


# ---------------------------------------------------------------------------
# Real reconstruction: (a) paper-faithful corrected reference, (b) composite
# ---------------------------------------------------------------------------

def _strip_lags_to_ref(frames, ref, strip_width=13):
    """Per-frame, per-row strip lags of each frame vs a reference frame (eqs 1-2).
    Returns (s_x, s_y) (F, R)."""
    f_ref = np.fft.fft2(ref)
    F, R, _ = frames.shape
    sx = np.zeros((F, R)); sy = np.zeros((F, R))
    t0 = time.time()
    for f in range(F):
        xl, yl, _ = h2h._strip_register_frame(frames[f], f_ref, ref.shape,
                                               strip_width, subpixel=True)
        sx[f] = xl; sy[f] = yl
        if f % 5 == 0:
            print(f"    [strip->ref] {f}/{F} ({time.time()-t0:.0f}s)")
    return sx, sy


def _best_sign_composite(frames, ex, ey):
    """Build a register-and-average composite, choosing the single global sign
    convention (+/-) of the trace that maximizes composite sharpness. This is a
    coordinate convention, decided independently and identically per method."""
    best = None; best_s = 1.0; best_sharp = -np.inf
    for s in (1.0, -1.0):
        c = composite_from_traces(frames, s * ex, s * ey)
        sh = sharpness(c[12:-12, 12:-12])
        if sh > best_sharp:
            best_sharp, best_s, best = sh, s, c
    return best, best_s


def run_real(n_frames=12, rebuild=False):
    print("\n=== REAL reconstruction (no GT: paper corrected-ref + composite) ===")
    frames = h2h.load_real(n=n_frames)
    ref = frames[0]
    R = ref.shape[0]

    # reuse the head-to-head real cache for per-frame traces (sx/sy strip lags,
    # ex/ey DPF positions), all relative to frame 0.
    real = h2h.real_comparison(n_frames=n_frames, rebuild=rebuild)
    sx, sy, ex, ey = real["sx"], real["sy"], real["ex"], real["ey"]

    # cone frequency from the raw (naive) average
    naive = frames.mean(axis=0)
    f_cone = cone_peak(naive[12:-12, 12:-12])[0]
    print(f"  cone-band peak frequency (naive avg): {f_cone:.3f} cyc/px")

    # (b) register-and-average composites; pick global sign per method
    sota_comp, sgn_s = _best_sign_composite(frames, sx, sy)
    ours_comp, sgn_o = _best_sign_composite(frames, ex, ey)
    comps = {"naive": naive, "sota": sota_comp, "ours": ours_comp}
    metrics = {k: ref_free_metrics(v, f_cone) for k, v in comps.items()}

    print("\n  composite | cone-prom | sharp    | contrast")
    for k in ["naive", "sota", "ours"]:
        m = metrics[k]
        print(f"  {k:6s} | {m['cone_prom']:8.2f}  | {m['sharp']:.2e} | {m['contrast']:.4f}")

    # effect size on real (no GT): disjoint sub-composites, reference-free
    n_groups = min(6, n_frames // 2)
    groups = np.array_split(np.arange(n_frames), n_groups)
    rs = {"sharp_sota": [], "sharp_ours": [], "cone_sota": [], "cone_ours": []}
    for g in groups:
        cs = composite_from_traces(frames[g], sgn_s * sx[g], sgn_s * sy[g])
        co = composite_from_traces(frames[g], sgn_o * ex[g], sgn_o * ey[g])
        rs["sharp_sota"].append(ref_free_metrics(cs, f_cone)["sharp"])
        rs["sharp_ours"].append(ref_free_metrics(co, f_cone)["sharp"])
        rs["cone_sota"].append(ref_free_metrics(cs, f_cone)["cone_prom"])
        rs["cone_ours"].append(ref_free_metrics(co, f_cone)["cone_prom"])
    eff = {
        "sharp": _paired_stats(np.array(rs["sharp_ours"]) - np.array(rs["sharp_sota"])),
        "cone": _paired_stats(np.array(rs["cone_ours"]) - np.array(rs["cone_sota"])),
    }
    print(f"\n  effect size ({n_groups} disjoint sub-composites, OURS - SOTA, no GT):")
    print(f"    sharpness d={eff['sharp']['mean']:+.2e} dz={eff['sharp']['dz']:+.2f} "
          f"p={eff['sharp']['p']:.2e}")
    print(f"    cone-prom d={eff['cone']['mean']:+.2f} dz={eff['cone']['dz']:+.2f} "
          f"p={eff['cone']['p']:.2e}")

    return dict(comps=comps, metrics=metrics, f_cone=f_cone, ref=ref,
                sgn_sota=sgn_s, sgn_ours=sgn_o, n_frames=n_frames, eff=eff)


def run_paper_repro():
    """Reproduce the paper's exact corrected-reference on the native
    slo_frames_real_small set (256x256, reference 065) using the authors' OWN
    cached strip lags in external/.../tmp/. Confirms our eqs 3-10 port is
    faithful to demonstrate_registration.py."""
    base = os.path.join(EXT, "slo_frames_real_small")
    tmp = os.path.join(EXT, "tmp")
    sxf = os.path.join(tmp, "slo_frames_real_small_s_x_065.npy")
    syf = os.path.join(tmp, "slo_frames_real_small_s_y_065.npy")
    if not (os.path.exists(sxf) and os.path.exists(base)):
        print("  [paper-repro] native cache/data missing, skipping")
        return None
    s_x = np.load(sxf); s_y = np.load(syf)
    ref = np.load(os.path.join(base, "065.npy"))
    x_hat, y_hat, corrected = lag_bias_reference(s_x, s_y, ref)
    print(f"  [paper-repro] corrected ref {corrected.shape}; recovered trace "
          f"x[{x_hat.min():.2f},{x_hat.max():.2f}] y[{y_hat.min():.2f},{y_hat.max():.2f}]")
    return dict(ref=ref, corrected=corrected, x_hat=x_hat, y_hat=y_hat)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _imshow(ax, img, title):
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def make_sim_figure(sim, path):
    obj, comps, metrics = sim["obj"], sim["comps"], sim["metrics"]
    mo = sim["obj_metrics"]
    fig = plt.figure(figsize=(15, 8.5)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.28, wspace=0.12)

    panels = [("GT object", obj, None),
              ("naive average\n(no correction)", comps["naive"], metrics["naive"]),
              ("SOTA strip-reg recon", comps["sota"], metrics["sota"]),
              ("OUR DPF recon", comps["ours"], metrics["ours"])]
    for i, (title, img, m) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        sub = title if m is None else (
            f"{title}\nPSNR {m['psnr']:.2f} dB | SSIM {m['ssim']:.3f}")
        _imshow(ax, img, sub)

    # bar metrics: PSNR, SSIM, cone prominence
    order = ["naive", "sota_int", "sota", "ours", "ideal_gt"]
    labels = ["naive", "SOTA int", "SOTA sub-px", "OURS", "ideal (GT)"]
    colors = ["#888888", "#e0a0a0", "#d1495b", "#1b8a5a", "#222222"]

    ax = fig.add_subplot(gs[1, 0])
    vals = [metrics[k]["psnr"] for k in order]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("PSNR (dB)"); ax.set_title("Reconstruction PSNR vs GT", fontsize=11)
    ax.tick_params(axis="x", rotation=35, labelsize=8); ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(gs[1, 1])
    vals = [metrics[k]["ssim"] for k in order]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("SSIM"); ax.set_title("Reconstruction SSIM vs GT", fontsize=11)
    ax.tick_params(axis="x", rotation=35, labelsize=8); ax.grid(alpha=0.25, axis="y")
    ax.set_ylim(0, 1.0)

    ax = fig.add_subplot(gs[1, 2])
    vals = [metrics[k]["cone_prom"] for k in order] + [mo["cone_prom"]]
    ax.bar(labels + ["OBJECT"], vals, color=colors + ["#3a6ea5"])
    ax.set_ylabel("cone-ring prominence"); ax.set_title("Cone power @ object freq", fontsize=11)
    ax.tick_params(axis="x", rotation=35, labelsize=8); ax.grid(alpha=0.25, axis="y")

    # difference maps SOTA vs OURS (|recon - object|)
    ax = fig.add_subplot(gs[1, 3])
    a_s = _align_to(obj, comps["sota"]); a_o = _align_to(obj, comps["ours"])
    b = obj[12:-12, 12:-12]
    d = np.abs(a_o - b) - np.abs(a_s - b)   # negative (blue) = OURS closer to GT
    lim = np.percentile(np.abs(d), 99)
    im = ax.imshow(d, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_title("|OURS-GT| - |SOTA-GT|\n(blue = OURS closer)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("AO-SLO intra-frame motion correction (Azimipour 2018 pipeline) — "
                 "SOTA strip-reg vs OUR DPF trace, SIMULATED set (GT object available)",
                 fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def make_real_figure(realr, path):
    comps, metrics = realr["comps"], realr["metrics"]
    fig = plt.figure(figsize=(14, 6.5)); fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.9], wspace=0.18)
    for i, (k, title) in enumerate([("naive", "naive average\n(no correction)"),
                                    ("sota", "SOTA strip-reg recon"),
                                    ("ours", "OUR DPF recon")]):
        ax = fig.add_subplot(gs[0, i])
        m = metrics[k]
        _imshow(ax, comps[k], f"{title}\nsharp {m['sharp']:.2e} | cone {m['cone_prom']:.1f}")
    ax = fig.add_subplot(gs[0, 3])
    labels = ["naive", "SOTA", "OURS"]
    vals = [metrics[k]["sharp"] for k in ["naive", "sota", "ours"]]
    ax.bar(labels, vals, color=["#888888", "#d1495b", "#1b8a5a"])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1e}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("gradient sharpness"); ax.set_title("composite sharpness", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"AO-SLO reconstruction on REAL frames (no GT, {realr['n_frames']} "
                 "frames, register+average) — SOTA strip-reg vs OUR DPF trace",
                 fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n-particles", type=int, default=250)
    ap.add_argument("--real-frames", type=int, default=12)
    a = ap.parse_args()

    sim = run_simulated(rebuild=a.rebuild, n_particles=a.n_particles)
    make_sim_figure(sim, os.path.join(RESULTS, "aoslo_image_quality.png"))

    repro = run_paper_repro()
    realr = run_real(n_frames=a.real_frames, rebuild=a.rebuild)
    make_real_figure(realr, os.path.join(RESULTS, "aoslo_image_quality_real.png"))

    # stash a compact summary for the md writer
    np.savez(os.path.join(CACHE, "image_quality_summary.npz"),
             f_cone=sim["f_cone"],
             **{f"sim_{k}_{mk}": mv for k, m in sim["metrics"].items()
                for mk, mv in m.items()},
             **{f"obj_{mk}": mv for mk, mv in sim["obj_metrics"].items()},
             **{f"real_{k}_{mk}": mv for k, m in realr["metrics"].items()
                for mk, mv in m.items()},
             real_sgn_sota=realr["sgn_sota"], real_sgn_ours=realr["sgn_ours"])
    return sim, realr, repro


if __name__ == "__main__":
    main()
