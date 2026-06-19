"""phaseA_metric.py — Phase A: re-metric to flip-rate + core-RMS, and locate the flips.

Metric of record = flip-rate + core-RMS on the PF OUTPUT (never std). Compares harness rdx
(chain_x+rdx) vs PF output (x_px) against a flip-resistant track, per subject, never pooled.

Labels (cross-reference, from lines_multiref3 — NOT even/odd):
  alias-flip-prone : refs N-1/N-2/N-3 DISAGREE >=3px on the line (mosaic ambiguity, inconsistent).
  real-microsaccade: refs AGREE (<1.5px) AND a >=3px jump from the robust track (eye moved).
  clean            : no disagreement, no jump (typical sub-px).

Metric (for estimate in {harness rdx, PF x_px}):
  flip-rate            = frac of alias-flip-prone lines where |est-track| >= 3px (carried the flip)
  core-RMS (arcmin)    = RMS(est-track) over clean lines (typical precision)
  microsacc-preserved  = frac of real-microsaccade lines where |est-track| >= 3px (kept the eye move)

Emit results/flip_labels.json + results/metric_rdx_vs_pf.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

import people_fov_pf as pf
import floor_multiref as fm

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX
FLIP_PX = 3.0          # mosaic half-spacing threshold (mosaic ~6px)
AGREE_PX = 1.5         # references "agree" below this


def _load(person):
    sub = pf.subject_by_name(person)
    lm = pf.build_line_measurements(sub)
    mr = fm.build_multiref(sub)
    z = np.load(os.path.join(sub.cache_dir, "m4_dpf_physics.npz"))
    ch = pf.build_chain(sub)
    return sub, lm, mr, {k: z[k] for k in z.files}, ch


def run(person):
    sub, lm, mr, pfz, ch = _load(person)
    fs = float(lm["line_rate"])
    frame = lm["frame"]; chain_x = lm["chain_x"]
    R = np.stack([mr["rdx1"], mr["rdx2"], mr["rdx3"]], 0).astype(float)
    Q = np.stack([mr["qh1"], mr["qh2"], mr["qh3"]], 0).astype(float)
    con = mr["con"]
    refs_locked = (Q > pf.Q_FOV).all(0) & np.isfinite(R).all(0)
    fovm = refs_locked & (con > pf.CONTRAST_FRAC * np.median(con))

    common = chain_x[frame] + R.mean(0)                 # px, flip-diluted
    harness_pos = chain_x[frame] + lm["rdx"].astype(float)
    pf_pos = pfz["x_px"].astype(float)
    pf_valid = pfz["valid"].astype(bool)

    # flip-resistant track: median(7) rejects isolated flips, then 10 ms gaussian
    import khz2d
    filled = khz2d.fill_nan(np.where(fovm, common, np.nan))
    track = gaussian_filter1d(median_filter(filled, 7), fs * 0.01)

    pairwise_max = np.zeros(R.shape[1])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pairwise_max = np.maximum(pairwise_max, np.abs(R[i] - R[j]))
    jump = np.abs(common - track) >= FLIP_PX

    alias_flip_prone = fovm & (pairwise_max >= FLIP_PX)
    real_microsacc = fovm & (pairwise_max < AGREE_PX) & jump
    clean = fovm & (pairwise_max < AGREE_PX) & (~jump)

    # main-sequence sanity on real microsaccades: peak-velocity vs amplitude correlation
    ms_idx = np.where(real_microsacc)[0]
    amp = np.abs(common - track)[ms_idx] * ARC                       # arcmin
    vel = np.abs(np.gradient(common))[ms_idx] * ARC * fs / 60.0      # arcmin/s -> deg/s approx
    main_seq_r = float(np.corrcoef(amp, vel)[0, 1]) if ms_idx.size > 50 else float("nan")

    def metric(est_pos, valid):
        dep = np.abs(est_pos - track)
        afp = alias_flip_prone & valid
        cl = clean & valid
        ms = real_microsacc & valid
        return dict(
            flip_rate=float(np.mean(dep[afp] >= FLIP_PX)) if afp.any() else float("nan"),
            core_rms_arcmin=float(np.sqrt(np.nanmean((est_pos - track)[cl] ** 2)) * ARC)
            if cl.any() else float("nan"),
            microsacc_preserved=float(np.mean(dep[ms] >= FLIP_PX)) if ms.any() else float("nan"),
            n_flipprone=int(afp.sum()), n_clean=int(cl.sum()), n_microsacc=int(ms.sum()))

    m_harness = metric(harness_pos, fovm)
    m_pf = metric(pf_pos, fovm & pf_valid)

    labels = dict(person=person,
                  n_total=int(fovm.size), n_fov=int(fovm.sum()),
                  frac_alias_flip_prone=float(alias_flip_prone.sum() / fovm.sum()),
                  frac_real_microsacc=float(real_microsacc.sum() / fovm.sum()),
                  frac_clean=float(clean.sum() / fovm.sum()),
                  main_sequence_r=main_seq_r,
                  flip_px_threshold=FLIP_PX, agree_px_threshold=AGREE_PX)
    metricout = dict(person=person, harness_rdx=m_harness, pf_output=m_pf,
                     pf_suppresses_flips=bool(np.isfinite(m_pf["flip_rate"])
                                              and m_pf["flip_rate"] < 0.5 * m_harness["flip_rate"]),
                     note=("CRUX: pf flip-rate << harness => PF suppresses the flips; "
                           "pf flip-rate ~ harness => flips survive into the deliverable."))
    print(f"[PhaseA {person}] labels: flip-prone {labels['frac_alias_flip_prone']*100:.1f}% "
          f"microsacc {labels['frac_real_microsacc']*100:.2f}% clean {labels['frac_clean']*100:.1f}% "
          f"main-seq r={main_seq_r:.2f}")
    print(f"   harness: flip-rate={m_harness['flip_rate']:.3f} core-RMS={m_harness['core_rms_arcmin']:.3f}' "
          f"ms-preserved={m_harness['microsacc_preserved']:.2f}")
    print(f"   PF out : flip-rate={m_pf['flip_rate']:.3f} core-RMS={m_pf['core_rms_arcmin']:.3f}' "
          f"ms-preserved={m_pf['microsacc_preserved']:.2f}  -> PF_suppresses={metricout['pf_suppresses_flips']}")
    return labels, metricout


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="+", default=["Igor", "Ashton3"])
    args = ap.parse_args()
    labels = {}; metrics = {}
    for p in args.people:
        lb, mt = run(p)
        labels[p] = lb; metrics[p] = mt
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(labels, open(os.path.join(RESULTS, "flip_labels.json"), "w"), indent=2)
    json.dump(metrics, open(os.path.join(RESULTS, "metric_rdx_vs_pf.json"), "w"), indent=2)
    print("-> results/flip_labels.json results/metric_rdx_vs_pf.json")


if __name__ == "__main__":
    main()
