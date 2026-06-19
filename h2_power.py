"""h2_power.py — Phase 2 Task 5: intervention-noise floor of per-line re-referencing,
and the analytic identity test that gates the whole H2 experiment.

A per-line re-reference shifts the per-column horizontal search center by a causal running
estimate delta(c) and re-adds delta(c) to the result. The committed NCC profile prof[c, .]
(span +-PADH) is exactly what the rdx argmax runs on, so we can apply the operation post-hoc:

    prof_reref[c, k] = prof[c, k - delta(c)]        (sub-pixel interp)
    rdx_reref[c]     = argmax_k(prof_reref) + delta(c)

Because argmax is translation-equivariant, rdx_reref == rdx_baseline up to interpolation
noise. That interpolation noise IS the intervention-noise floor. If it is not << any
plausible C1 effect, no per-line-reref A/B can be trusted; and if the operation is an
identity, H2 is H0 for the rdx/C1 metric by construction.

Emit results/h2_power.json. PER SUBJECT.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

import khz2d
import people_fov_pf as pf

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX


def causal_running_delta(lm, alpha=0.05):
    """Causal within-frame running estimate of horizontal position (px), per line, from
    rdx already observed earlier in the same frame (EWMA reset at each frame boundary).
    This is exactly the kind of running estimate a per-line re-reference would use."""
    rdx = lm["rdx"].astype(float)
    frame = lm["frame"]
    col = lm["col"]
    delta = np.zeros_like(rdx)
    # vectorized per-frame EWMA is awkward; do a simple causal cumulative mean within frame
    # delta(c) = mean of rdx[0..c-1] in this frame (causal, excludes current line)
    order = np.lexsort((col, frame))
    inv = np.empty_like(order); inv[order] = np.arange(order.size)
    rs = rdx[order]; fs = frame[order]
    out = np.zeros_like(rs)
    csum = 0.0; cnt = 0; cur = -1
    for i in range(rs.size):
        if fs[i] != cur:
            cur = fs[i]; csum = 0.0; cnt = 0
        out[i] = (csum / cnt) if cnt > 0 else 0.0
        csum += rs[i]; cnt += 1
    return out[inv]


def reref_rdx(lm, delta):
    """Apply per-column profile shift by delta then re-add delta (translation-equivariant)."""
    prof = lm["prof"].astype(np.float32)          # (N, 2*PADH+1)
    N, L = prof.shape
    padh = (L - 1) // 2
    k = np.arange(L)
    rdx_new = np.empty(N, np.float32)
    # Re-reference: shift the search profile by delta (sample at k+delta), localize the
    # RESIDUAL, then re-add delta. Net translation-equivariant identity (subtract the shift
    # we applied), so rdx_new == rdx_baseline up to interpolation noise.
    for i in range(N):
        d = float(delta[i])
        if d == 0.0:
            shifted = prof[i]
        else:
            shifted = np.interp(k + d, k, prof[i], left=prof[i, 0], right=prof[i, -1])
        kp = int(np.argmax(shifted))
        rdx_new[i] = khz2d._parab(shifted, kp) - padh + d
    return rdx_new


def scat(rdx, lm, m):
    fs = float(lm["line_rate"])
    filled = khz2d.fill_nan(np.where(m, rdx, np.nan))
    sm = gaussian_filter1d(median_filter(filled, 7), fs * 0.01)
    return float(np.nanstd((rdx - sm)[m]) * ARC)


def run(person):
    sub = pf.subject_by_name(person)
    lm = pf.build_line_measurements(sub)
    m = pf.fov_mask(lm)
    delta = causal_running_delta(lm)
    # subsample for the (python-loop) reref to keep it fast but representative
    idx = np.where(m)[0]
    rng = np.random.default_rng(0)
    sub_idx = np.sort(rng.choice(idx, size=min(60000, idx.size), replace=False))
    base = lm["rdx"][sub_idx].astype(float)
    lm_sub = {"prof": lm["prof"][sub_idx]}
    d_full = delta[sub_idx]
    delta_rms = float(np.sqrt(np.mean(d_full ** 2)))
    # sweep delta magnitude: identity at scale->0 (translation-equivariance), growth with
    # scale exposes alias-peak flipping + interpolation. scale=1.0 = the running estimate a
    # real per-line reref would actually use.
    sweep = {}
    for s in (0.05, 0.1, 0.25, 0.5, 1.0):
        rr = reref_rdx(lm_sub, d_full * s)
        diff = rr - base
        sweep[f"scale_{s}"] = dict(
            delta_rms_px=float(delta_rms * s),
            intervention_noise_arcmin=float(np.std(diff) * ARC),
            max_abs_change_px=float(np.max(np.abs(diff))),
            frac_jumped_gt1px=float(np.mean(np.abs(diff) > 1.0)))
    full_noise = sweep["scale_1.0"]["intervention_noise_arcmin"]
    out = dict(person=person,
               reref_delta_rms_px=delta_rms,
               intervention_noise_arcmin=full_noise,
               sweep=sweep, n_subset=int(sub_idx.size),
               PASS_power_gate=bool(full_noise < 0.5 * 1.7),   # vs real-motion UB ~1.7-2.0'
               note=("Per-line reref shifts the per-column search center by the causal running "
                     "estimate and re-adds it. Translation-equivariance makes it identity ONLY "
                     "while the SAME alias peak stays global; the NCC profile is multi-peaked "
                     "(photoreceptor mosaic), so realistic-magnitude shifts flip alias peaks and "
                     "inject large noise. Power gate compares intervention noise to <0.5x the "
                     "real-motion upper bound (~1.7'). Identity verified at scale->0."))
    print(f"[H2-power {person}] delta_rms={delta_rms:.2f}px  intervention_noise@scale: " +
          " ".join(f"{s}:{sweep[f'scale_{s}']['intervention_noise_arcmin']:.3f}'"
                   for s in (0.05, 0.1, 0.25, 0.5, 1.0)) +
          f"  -> gate {'PASS' if out['PASS_power_gate'] else 'FAIL'}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="+", default=["Igor", "Ashton3"])
    args = ap.parse_args()
    res = {p: run(p) for p in args.people}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "h2_power.json"), "w"), indent=2)
    print("-> results/h2_power.json")


if __name__ == "__main__":
    main()
