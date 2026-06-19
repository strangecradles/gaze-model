"""mosaic_prior_ab.py — power gate + A/B of the belief-level --mosaic-prior on the PF output.

Reuses the mosaic_ab scoring (Phase-A cross-reference labels + flip-resistant track): flip-rate,
core-RMS, microsacc-preservation, r-vs-dot. Per subject, never pooled.

OFF = committed m4_dpf_physics.npz (mosaic_prior False == byte-identical baseline).
ON  = m4_mosaic_prior[_dN].npz (rebuilt with mosaic_prior=True).

Signal-safety PASS = (i) flip-rate down (CI<0), (ii) core-RMS <= 1.05x OFF, (iii) r-vs-dot >=
OFF-0.01. (microsacc-preservation reported but NOT a hard gate: Phase A showed the >=3px-jump
"microsaccade" label contains no real microsaccades.)
"""
from __future__ import annotations

import json
import os

import numpy as np

import people_fov_pf as pf
import mosaic_ab as mab

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def build_on(person, sigma_rows=3.0, reseed_hold=8, ncc_thr=0.35, dur_s=None):
    sub = pf.subject_by_name(person)
    nt = "" if abs(ncc_thr - 0.35) < 1e-9 else f"_ncc{int(round(ncc_thr*100))}"
    base = "m4_mosaic_prior" if dur_s is None else f"m4_mosaic_prior_d{int(dur_s)}"
    path = os.path.join(sub.cache_dir, f"{base}{nt}.npz")
    if not os.path.exists(path):
        lm = pf.build_line_measurements(sub); ch = pf.build_chain(sub)
        pf.run_m4(sub, lm, ch, cache_path=path, rebuild=True, dur_s=dur_s,
                  mosaic_prior=True, mosaic_prior_sigma_rows=sigma_rows,
                  mosaic_prior_reseed_hold=reseed_hold, mosaic_prior_ncc_thr=ncc_thr)
    return {k: v for k, v in np.load(path).items()}, path


def run(person, sigma_rows=3.0, ncc_thr=0.35, dur_s=None):
    L = mab.labels_and_track(person)
    off = {k: v for k, v in np.load(os.path.join(L["sub"].cache_dir, "m4_dpf_physics.npz")).items()}
    on, _ = build_on(person, sigma_rows=sigma_rows, ncc_thr=ncc_thr, dur_s=dur_s)
    n = on["x_px"].size                                   # restrict to built lines (short runs)
    # restrict label/track arrays to the first n lines
    Ls = {k: (v[:n] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == off["x_px"].size else v)
          for k, v in L.items()}
    ox, ov = off["x_px"][:n].astype(float), off["valid"][:n].astype(bool)
    nx, nv = on["x_px"].astype(float), on["valid"].astype(bool)
    s_off = mab.score(Ls, ox, ov); s_on = mab.score(Ls, nx, nv)
    full = (dur_s is None)
    r_off = mab.r_dot(Ls, ox, ov) if full else float("nan")
    r_on = mab.r_dot(Ls, nx, nv) if full else float("nan")
    ci = mab.boot_flip_delta(Ls, ox, ov, nx, nv) if full else None
    out = dict(person=person, sigma_rows=sigma_rows, dur_s=dur_s, n_lines=int(n),
               off=dict(**s_off, r_dot_x=r_off), on=dict(**s_on, r_dot_x=r_on),
               flip_rate_delta=float(s_on["flip_rate"] - s_off["flip_rate"]),
               core_rms_delta=float(s_on["core_rms"] - s_off["core_rms"]),
               flip_delta_ci=ci)
    if full:
        out["SIGNAL_SAFE_PASS"] = bool(s_on["flip_rate"] < s_off["flip_rate"] and ci[1] < 0
                                       and s_on["core_rms"] <= s_off["core_rms"] * 1.05
                                       and r_on >= r_off - 0.01)
    print(f"[prior-AB {person} sig{sigma_rows}{'' if full else ' dur'+str(dur_s)}] "
          f"flip {s_off['flip_rate']:.3f}->{s_on['flip_rate']:.3f} "
          f"(Δ{out['flip_rate_delta']:+.3f}{'' if not ci else ' CI'+str([round(c,3) for c in ci])}) "
          f"core {s_off['core_rms']:.3f}->{s_on['core_rms']:.3f}' "
          f"ms {s_off['microsacc_preserved']:.2f}->{s_on['microsacc_preserved']:.2f}"
          f"{'' if not full else f' r {r_off:.3f}->{r_on:.3f} PASS='+str(out.get(chr(83)+'IGNAL_SAFE_PASS'))}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="+", default=["Igor", "Ashton3"])
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--ncc", type=float, default=0.35)
    ap.add_argument("--dur", type=float, default=None)
    args = ap.parse_args()
    res = {p: run(p, sigma_rows=args.sigma, ncc_thr=args.ncc, dur_s=args.dur) for p in args.people}
    os.makedirs(RESULTS, exist_ok=True)
    tag = "mosaic_prior_ab" if args.dur is None else "mosaic_prior_power"
    json.dump(res, open(os.path.join(RESULTS, f"{tag}.json"), "w"), indent=2)
    print(f"-> results/{tag}.json")


if __name__ == "__main__":
    main()
