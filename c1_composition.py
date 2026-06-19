"""c1_composition.py — Phase A: MEASURE the composition of C1 (reference-specific floor).

C1 = spread of per-line localizations across reference frames N-1/N-2/N-3. We split its
variance into four measured sub-components (NOT assumed) per subject, never pooled:

  dev(L,k) = rdx(L,k) - mean_k rdx(L,.)          per-reference deviation (px)
  V = pooled var of dev over FOV (L,k)            = the C1 variance budget

  ALIAS      : heavy discrete tail |dev| > 0.5*alias_spacing (mosaic-peak flips). Variance share.
  DISTORTION : core |dev| explained by reference frame k's OWN intra-frame motion |inc[f-k]|.
  CHAIN      : age-gap dependence of pairwise disagreement var(rdx_i-rdx_j) vs |i-j| hops.
  TEMPLATE   : motion-independent core var that correlates with NCC peak height/contrast (low SNR).

Fractions are NON-orthogonal (reported with that caveat). Emit results/c1_composition.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

import people_fov_pf as pf
import floor_multiref as fm

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX


def _r2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    if x.size < 100 or x.std() < 1e-9:
        return dict(r2=0.0, slope=0.0, n=int(x.size))
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = s * x + b
    ss = np.sum((y - yh) ** 2); st = np.sum((y - y.mean()) ** 2)
    return dict(r2=float(1 - ss / st) if st > 0 else 0.0, slope=float(s),
                intercept=float(b), n=int(x.size))


def alias_spacing(sub, m_lm) -> float:
    """Estimate mosaic alias spacing (px) from per-line NCC profile autocorrelation."""
    lm = pf.build_line_measurements(sub)
    prof = lm["prof"][m_lm].astype(np.float32)
    idx = np.random.default_rng(0).choice(prof.shape[0], size=min(4000, prof.shape[0]), replace=False)
    P = prof[idx]
    P = P - P.mean(1, keepdims=True)
    ac = np.array([np.correlate(p, p, "full")[len(p) - 1:] for p in P]).mean(0)
    # first secondary maximum after the zero-lag peak
    from scipy.signal import find_peaks
    pk, _ = find_peaks(ac)
    return float(pk[0]) if pk.size else 6.0


def run(person) -> dict:
    sub = pf.subject_by_name(person)
    mr = fm.build_multiref(sub)
    ch = pf.build_chain(sub)
    inc_mag = np.sqrt(ch["inc_x"] ** 2 + ch["inc_y"] ** 2)            # px/frame per frame
    R = np.stack([mr["rdx1"], mr["rdx2"], mr["rdx3"]], 0).astype(float)
    Q = np.stack([mr["qh1"], mr["qh2"], mr["qh3"]], 0).astype(float)
    con = mr["con"]; frame = mr["frame"]
    m = (Q > pf.Q_FOV).all(0) & (con > pf.CONTRAST_FRAC * np.median(con)) & np.isfinite(R).all(0)
    lm_mask = pf.fov_mask(pf.build_line_measurements(sub))
    spacing = alias_spacing(sub, lm_mask)

    dev = R - R.mean(0)                                               # (3, N)
    # flatten masked (L,k)
    flat_dev = dev[:, m].ravel()
    qk = Q[:, m].ravel()
    # ref k motion: inc_mag at frame f-k
    fr = frame[m]
    motion = np.stack([inc_mag[np.clip(fr - k, 0, inc_mag.size - 1)] for k in (1, 2, 3)], 0).ravel()
    V = float(np.var(flat_dev))
    C1_px = float(np.sqrt(np.mean((dev[:, m] ** 2)) * (3.0 / 2.0)))   # ~ RMS std across refs
    thr = 0.5 * spacing

    # ALIAS: heavy-tail variance share
    tail = np.abs(flat_dev) > thr
    V_alias = float(np.sum(flat_dev[tail] ** 2) / flat_dev.size)
    frac_alias = V_alias / V if V > 0 else 0.0
    # is the alias-jump RATE motion-driven? (per-frame alias rate vs |inc|) — this is where the
    # R^2=0.33 motion-scaling lives if motion drives mosaic-flips rather than smooth distortion.
    fr_all = frame[m]
    uf = np.unique(fr_all)
    tail_byline = np.abs(dev[:, m]).max(0) > thr           # per masked line: any ref flipped
    rate = np.array([tail_byline[fr_all == f].mean() for f in uf])
    fmot = inc_mag[np.clip(uf, 0, inc_mag.size - 1)]
    alias_vs_motion = _r2(fmot, rate)
    core = ~tail
    dc = flat_dev[core]; qc = qk[core]; mc = motion[core]
    V_core = float(np.var(dc))

    # DISTORTION: core dev^2 explained by reference motion
    dist = _r2(mc, dc ** 2)
    frac_dist = float(dist["r2"] * V_core / V) if V > 0 else 0.0
    # also |dev| vs motion for a sign-stable readout
    dist_abs = _r2(mc, np.abs(dc))

    # TEMPLATE: core dev^2 explained by NCC peak height (low qh -> high dev), motion-independent
    templ = _r2(1.0 / np.maximum(qc, 1e-3), dc ** 2)
    # the part of template R2 NOT shared with motion: partial via residualizing dc^2 on motion first
    frac_templ = float(templ["r2"] * V_core / V) if V > 0 else 0.0

    # CHAIN: age-gap dependence of pairwise disagreement variance
    d12 = (R[0] - R[1])[m]; d23 = (R[1] - R[2])[m]; d13 = (R[0] - R[2])[m]
    v_gap1 = float(0.5 * (np.var(d12) + np.var(d23)))                 # 1-hop pairs
    v_gap2 = float(np.var(d13))                                       # 2-hop pair
    # var(rdx_i-rdx_j)=2*eps_var + chain_var*gap ; slope per hop = v_gap2 - v_gap1
    chain_var_per_hop = max(v_gap2 - v_gap1, 0.0)
    # C1 var ~ (1/3)*sum pairwise var /  (K-1)... express chain share via per-line std model:
    frac_chain = float(chain_var_per_hop / (2.0 * V)) if V > 0 else 0.0

    out = dict(
        person=person, C1_px=C1_px, C1_arcmin=C1_px * ARC, V_px2=V,
        alias_spacing_px=spacing, alias_thresh_px=thr,
        components=dict(
            alias=dict(frac=frac_alias, tail_frac_of_Lk=float(np.mean(tail)),
                       rate_vs_motion_r2=alias_vs_motion["r2"],
                       rate_vs_motion_slope=alias_vs_motion["slope"],
                       note=">0.5*alias_spacing |dev| (discrete mosaic-peak flips); "
                            "rate_vs_motion = per-frame alias rate regressed on |inc| (motion-driven?)"),
            distortion=dict(frac=frac_dist, r2_dev2_vs_motion=dist["r2"], slope=dist["slope"],
                            r2_absdev_vs_motion=dist_abs["r2"],
                            note="core |dev| explained by reference frame's own |inc| motion"),
            template=dict(frac=frac_templ, r2_dev2_vs_inv_qh=templ["r2"],
                          note="core dev^2 explained by 1/NCC-peak-height (SNR); motion-independent"),
            chain=dict(frac=frac_chain, var_gap1_px2=v_gap1, var_gap2_px2=v_gap2,
                       chain_var_per_hop_px2=chain_var_per_hop,
                       note="age-gap slope of pairwise disagreement variance"),
        ),
        caveat="fractions are NON-orthogonal (distortion & alias both motion-driven; template & "
               "distortion may share variance). Read as attributions, not a partition.")
    comp = out["components"]
    dom = max(comp, key=lambda k: comp[k]["frac"])
    out["dominant"] = dom
    print(f"[C1-comp {person}] C1={C1_px*ARC:.2f}' spacing={spacing:.1f}px | "
          f"alias={comp['alias']['frac']:.2f} dist={comp['distortion']['frac']:.2f} "
          f"(R2={comp['distortion']['r2_absdev_vs_motion']:.2f}) templ={comp['template']['frac']:.2f} "
          f"chain={comp['chain']['frac']:.2f} -> DOMINANT={dom}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="+", default=["Igor", "Ashton3"])
    args = ap.parse_args()
    res = {p: run(p) for p in args.people}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "c1_composition.json"), "w"), indent=2)
    print("-> results/c1_composition.json")


if __name__ == "__main__":
    main()
