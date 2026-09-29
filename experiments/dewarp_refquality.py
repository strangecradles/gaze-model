"""dewarp_refquality.py — Task 5: reference-quality control.

Separates "intra-frame distortion" from "single-frame reference noise" by comparing
rdx_scatter using:
  - single previous-frame atlas         (committed baseline; == ref_frames=1)
  - K-frame chain-averaged reference     (ref_frames=K; sqrt(K) lower template noise)

All on a COMMON FOV mask. If the averaged reference lowers rdx_scatter, the floor is
single-frame reference noise rather than intra-frame distortion of one frame.

Emit results/dewarp_refquality.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

import people_fov_pf as pf
from dewarp_diag import residual_px

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX


def run(person: str = "Igor", ks=(3, 5)) -> dict:
    sub = pf.subject_by_name(person)
    variants = {"single_frame": pf.build_line_measurements(sub)}
    for k in ks:
        variants[f"refavg{k}"] = pf.build_line_measurements(sub, ref_frames=k)
    # common mask across all variants
    m = None
    for lm in variants.values():
        mm = pf.fov_mask(lm)
        m = mm if m is None else (m & mm)
    base = None
    rows = {}
    for name, lm in variants.items():
        r = residual_px(lm, m)
        s = float(np.nanstd(r[m]) * ARC)
        if base is None:
            base = s
        rows[name] = dict(rdx_scatter_arcmin=s,
                          reduction_vs_single=float((base - s) / base) if base else 0.0)
        print(f"[refquality {person}] {name:14s} {s:.3f}'  "
              f"(Δ vs single {(base - s) / base * 100:+.1f}%)")
    out = dict(person=person, capture=sub.stem, n_lines_common=int(m.sum()),
               variants=rows,
               note=("ref_frames=1 == committed single-frame baseline; refavgK = K-frame "
                     "chain-averaged reference. Drop with K => floor is single-frame "
                     "reference noise, not single-frame intra-frame distortion."))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Igor")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5])
    args = ap.parse_args()
    out = run(args.person, tuple(args.ks))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "dewarp_refquality.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("-> results/dewarp_refquality.json")


if __name__ == "__main__":
    main()
