"""m5_reg_sweep.py — M5 MAP smoother regularization sweep (closes critique caveat 2).

The pf_dejitter.md conclusion "M5 doesn't help" was premature: M5's regularization
(w_dyn, beta) was never swept. An under-regularized batch smoother interpolates
THROUGH per-line noise (β=4 over-trusts each line), which can make HF worse than the
forward filter. Crank the dynamics weight up / drop the data trust and re-measure.

If a strongly-regularized M5 STILL floors at ~3' HF, that is the final nail: the floor
is an observation/motion floor, not an estimator-bandwidth one.

Usage:
  python3 m5_reg_sweep.py --person Ashton3
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

import people_fov_m5 as m5
import people_fov_pf as pf

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# (tag, kwargs) — base is the original M5 operating point; the rest strengthen the prior.
CONFIGS = [
    ("base",        dict(w_dyn=0.5, beta=4.0)),
    ("dyn4",        dict(w_dyn=4.0, beta=4.0)),
    ("dyn8",        dict(w_dyn=8.0, beta=4.0)),
    ("beta1",       dict(w_dyn=0.5, beta=1.0)),
    ("dyn8_beta1",  dict(w_dyn=8.0, beta=1.0)),
]


def _metrics(label, run, refs):
    valid = run["valid"].astype(bool)
    rate = float(run["rate"])
    ev = pf.evaluate(run["t"], run["x_px"], run["y_px"], valid, rate, refs)
    hx = ev["cal_x"]
    p_low, p_hf = pf.psd_band_ratio(hx, rate, valid)
    return dict(
        label=label, prec_x=ev["prec_x"], r_dot_x=ev["r_dot_x"],
        jitter_30=pf.frame_jitter_30fps(hx, rate, valid),
        psd_hf=p_hf, valid_frac=ev["valid_frac"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Ashton3")
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()

    sub = pf.subject_by_name(args.person)
    lm = pf.build_line_measurements(sub)
    refs = pf.compute_refs(sub, lm)

    # M4 baseline (forward filter) for comparison.
    m4_path = os.path.join(sub.cache_dir, "m4_dpf_physics.npz")
    if not os.path.exists(m4_path):
        raise FileNotFoundError(f"missing {m4_path} — run the PF first")
    z = np.load(m4_path)
    m4 = _metrics("M4", {k: z[k] for k in z.files}, refs)

    rows = [dict(person=args.person, config="M4_fwd", w_dyn="", beta="", **{
        k: m4[k] for k in ("prec_x", "jitter_30", "r_dot_x", "psd_hf", "valid_frac")})]
    print(f"[{args.person}] M4_fwd  prec_x={m4['prec_x']:.3f}'  j30={m4['jitter_30']:.3f}'  "
          f"hf={m4['psd_hf']:.3f}  r_dot={m4['r_dot_x']:.3f}")

    for tag, kw in CONFIGS:
        run = m5.m5_map_people(args.person, iters=args.iters, rebuild=True,
                               cache_tag=f"m5_{tag}", **kw)
        mm = _metrics(f"M5_{tag}", run, refs)
        rows.append(dict(person=args.person, config=f"M5_{tag}",
                         w_dyn=kw["w_dyn"], beta=kw["beta"], **{
                             k: mm[k] for k in ("prec_x", "jitter_30", "r_dot_x",
                                                "psd_hf", "valid_frac")}))
        ratio = mm["prec_x"] / m4["prec_x"] if m4["prec_x"] else np.nan
        print(f"[{args.person}] M5_{tag:11s} w_dyn={kw['w_dyn']} beta={kw['beta']}  "
              f"prec_x={mm['prec_x']:.3f}'  j30={mm['jitter_30']:.3f}'  "
              f"hf={mm['psd_hf']:.3f}  r_dot={mm['r_dot_x']:.3f}  prec/M4={ratio:.2f}")

    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, f"m5_reg_sweep_{args.person}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {csv_path}")

    best = min((r for r in rows if r["config"].startswith("M5")),
               key=lambda r: r["prec_x"] if np.isfinite(r["prec_x"]) else 1e9)
    print(f"Best M5: {best['config']} prec_x={best['prec_x']:.3f}' "
          f"(M4 forward {m4['prec_x']:.3f}')")
    if best["prec_x"] >= m4["prec_x"] - 0.1:
        print("=> Even strongly-regularized M5 does NOT beat the forward filter's HF floor "
              "-> the floor is observation/motion-limited, not estimator-bandwidth-limited.")


if __name__ == "__main__":
    main()
