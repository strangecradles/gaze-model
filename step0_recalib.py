"""step0_recalib.py — calibrate the Step 0 ablation speckle to real localization noise.

Critique caveat 4: the ablation's injected per-line noise (OBS_NOISE_SIG=0.15) was
uncontrolled, so its ~0.06' output floor was not quantitatively load-bearing. Here we:

  1. Measure the REAL single-line argmax scatter on people data: rdx (per-line
     horizontal localization) minus a smoothed track, std in raster px -> arcmin.
  2. Find the synthetic OBS_NOISE_SIG whose single-line NCC-argmax std (atlas rows ->
     arcmin) matches that real scatter, using the SAME observation model as the filter
     (decoder.render + filter._fine + NCC argmax).
  3. Report the calibrated amplitude. The ablation is then re-run with
     OBS_NOISE_SIG=<calibrated> (see results/step0_recalib.md). If realistic-amplitude
     white noise STILL yields ~0.06' output, white noise is excluded as the 3' source.

Usage:
  python3 step0_recalib.py --person Igor
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter

import calib
import data
import decoder
import filter as flt
import khz2d
import noise as noisemod
import people_fov_pf as pf

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RATE = 14594.0
LINE_LEN = 200
TRUE_PERP = 300.0
TRUE_ALONG = 600.0


def real_line_scatter_px(person: str) -> float:
    """Std of per-line argmax (rdx) about a smoothed track, in raster px."""
    sub = pf.subject_by_name(person)
    lm = pf.build_line_measurements(sub)
    m = pf.fov_mask(lm)
    rdx = lm["rdx"].astype(float)
    fs = float(lm["line_rate"])
    filled = khz2d.fill_nan(np.where(m, rdx, np.nan))
    sm = gaussian_filter1d(median_filter(filled, 7), fs * 0.01)
    resid = (rdx - sm)[m]
    return float(np.nanstd(resid))


def synth_argmax_std_rows(obs_sig: float, n: int = 400, seed: int = 0,
                          span: float = 12.0, hp_sigma: float = flt.HP_SIGMA) -> float:
    """Std (atlas rows) of single-line NCC-argmax localization at a given noise amp."""
    atlas = data.load_atlas().ref_map
    cand = np.arange(TRUE_PERP - span, TRUE_PERP + span + 1e-6, 0.5)
    templ_hp = np.stack([
        flt._fine(decoder.render(torch.tensor(c, dtype=torch.float64),
                                 torch.tensor(TRUE_ALONG, dtype=torch.float64),
                                 atlas, LINE_LEN, col_step=1.0).detach().cpu().numpy(),
                  hp_sigma)
        for c in cand])
    clean = decoder.render(torch.tensor(TRUE_PERP, dtype=torch.float64),
                           torch.tensor(TRUE_ALONG, dtype=torch.float64),
                           atlas, LINE_LEN, col_step=1.0).detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    est = np.empty(n)
    for i in range(n):
        noisy = noisemod.apply_noise(clean + rng.normal(0, obs_sig, LINE_LEN), RATE, rng=rng)
        obs_hp = flt._fine(noisy, hp_sigma)
        ncc = (templ_hp * obs_hp[None, :]).mean(1)
        k = int(np.argmax(ncc))
        sub = 0.0
        if 0 < k < len(cand) - 1:
            a, b, c = ncc[k - 1], ncc[k], ncc[k + 1]
            den = a - 2 * b + c
            if abs(den) > 1e-12:
                sub = float(np.clip(0.5 * (a - c) / den, -1, 1))
        est[i] = cand[k] + sub * (cand[1] - cand[0])
    return float(np.std(est))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Igor")
    args = ap.parse_args()

    real_px = real_line_scatter_px(args.person)
    real_arcmin = real_px * pf.ARC_PER_PX
    target_rows = real_arcmin / calib.ARCMIN_PER_ROW
    print(f"[{args.person}] real single-line scatter: {real_px:.3f} px = "
          f"{real_arcmin:.3f}' = {target_rows:.3f} atlas rows")

    sigs = np.array([0.05, 0.10, 0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 2.0])
    stds = np.array([synth_argmax_std_rows(s) for s in sigs])
    for s, st in zip(sigs, stds):
        print(f"  obs_sig={s:.2f} -> argmax std {st:.3f} rows ({st*calib.ARCMIN_PER_ROW:.3f}')")
    # monotone-ish; interpolate to the target
    order = np.argsort(stds)
    calibrated = float(np.interp(target_rows, stds[order], sigs[order]))
    print(f"\nCalibrated OBS_NOISE_SIG = {calibrated:.3f}  "
          f"(matches real {target_rows:.3f}-row single-line scatter)")

    os.makedirs(RESULTS, exist_ok=True)
    md = os.path.join(RESULTS, "step0_recalib.md")
    with open(md, "w") as f:
        f.write("# Step 0 speckle recalibration\n\n")
        f.write(f"Real single-line localization scatter ({args.person}): "
                f"**{real_arcmin:.3f}'** ({real_px:.3f} px, {target_rows:.3f} atlas rows).\n\n")
        f.write(f"Calibrated `OBS_NOISE_SIG` = **{calibrated:.3f}** "
                "(synthetic single-line argmax std matched to the real value).\n\n")
        f.write("| obs_sig | argmax std (rows) | argmax std (′) |\n|--:|--:|--:|\n")
        for s, st in zip(sigs, stds):
            f.write(f"| {s:.2f} | {st:.3f} | {st*calib.ARCMIN_PER_ROW:.3f} |\n")
        f.write(f"\nRe-run the ablation at the calibrated amplitude:\n\n"
                f"```\nOBS_NOISE_SIG={calibrated:.3f} python3 -m pytest "
                f"tests/test_jitter_ablation.py -s -q\n```\n\n"
                "If the worst-config output still sits at ~0.06′ (≈0.13 rows), filter-internal "
                "white noise is conclusively excluded as the ~3′ floor source.\n")
    print(f"Wrote {md}")
    print(f"\nNext: OBS_NOISE_SIG={calibrated:.3f} python3 -m pytest "
          f"tests/test_jitter_ablation.py -s -q")


if __name__ == "__main__":
    main()
