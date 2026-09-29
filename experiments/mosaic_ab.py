"""mosaic_ab.py — Phase B Tasks 5-6: power gate + A/B of --mosaic-reject on the PF output.

Metric of record = flip-rate + core-RMS on the PF OUTPUT, scored against the Phase-A
cross-reference labels + flip-resistant track (NOT even/odd). Per subject, never pooled.

OFF  = committed m4_dpf_physics.npz (mosaic_reject False == byte-identical baseline).
ON   = m4_mosaic.npz (rebuilt with mosaic_reject=True; built here if missing).

Signal-safety gate PASS requires: (i) alias-flip-rate DOWN, (ii) real-microsaccade preserved
within +-5%, (iii) core-RMS not inflated (ON vs OFF, same track -> bias cancels),
(iv) r-vs-dot >= baseline-0.01. Emit results/mosaic_power.json + results/mosaic_ab.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

import khz2d
import people_fov_pf as pf
import floor_multiref as fm

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX
FLIP_PX = 3.0
AGREE_PX = 1.5


def labels_and_track(person):
    sub = pf.subject_by_name(person)
    lm = pf.build_line_measurements(sub)
    mr = fm.build_multiref(sub)
    ch = pf.build_chain(sub)
    fs = float(lm["line_rate"]); frame = lm["frame"]; cx = lm["chain_x"]
    R = np.stack([mr["rdx1"], mr["rdx2"], mr["rdx3"]], 0).astype(float)
    Q = np.stack([mr["qh1"], mr["qh2"], mr["qh3"]], 0).astype(float)
    con = mr["con"]
    fovm = (Q > pf.Q_FOV).all(0) & np.isfinite(R).all(0) & (con > pf.CONTRAST_FRAC * np.median(con))
    common = cx[frame] + R.mean(0)
    track = gaussian_filter1d(median_filter(khz2d.fill_nan(np.where(fovm, common, np.nan)), 7), fs * 0.01)
    pwm = np.zeros(R.shape[1])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pwm = np.maximum(pwm, np.abs(R[i] - R[j]))
    jump = np.abs(common - track) >= FLIP_PX
    afp = fovm & (pwm >= FLIP_PX)
    ms = fovm & (pwm < AGREE_PX) & jump
    cl = fovm & (pwm < AGREE_PX) & (~jump)
    return dict(sub=sub, lm=lm, ch=ch, fs=fs, frame=frame, track=track,
                fovm=fovm, afp=afp, ms=ms, clean=cl)


def score(L, x_px, valid):
    track = L["track"]; dep = np.abs(x_px - track)
    afp = L["afp"] & valid; cl = L["clean"] & valid; msk = L["ms"] & valid
    return dict(
        flip_rate=float(np.mean(dep[afp] >= FLIP_PX)),
        core_rms=float(np.sqrt(np.nanmean((x_px - track)[cl] ** 2)) * ARC),
        microsacc_preserved=float(np.mean(dep[msk] >= FLIP_PX)),
        n_flip=int(afp.sum()), n_clean=int(cl.sum()), n_ms=int(msk.sum()))


def r_dot(L, x_px, valid):
    refs = pf.compute_refs(L["sub"], L["lm"])
    posy = L["lm"]["chain_y"][L["frame"]] + L["lm"]["lam_v"].astype(float)
    ev = pf.evaluate(L["lm"]["t"].astype(float), x_px, posy, valid & L["fovm"],
                     L["fs"], refs, smooth_ms=2.0)
    return float(ev["r_dot_x"])


def boot_flip_delta(L, off_x, off_v, on_x, on_v, nboot=600, seed=0):
    rng = np.random.default_rng(seed)
    frame = L["frame"]; afp_off = L["afp"] & off_v; afp_on = L["afp"] & on_v
    track = L["track"]
    uf = np.unique(frame[L["afp"]])
    # per-frame flip indicators
    doff = (np.abs(off_x - track) >= FLIP_PX)
    don = (np.abs(on_x - track) >= FLIP_PX)
    deltas = np.empty(nboot)
    for b in range(nboot):
        fs_b = rng.choice(uf, uf.size, replace=True)
        sel = np.isin(frame, fs_b)
        ao = afp_off & sel; an = afp_on & sel
        fo = doff[ao].mean() if ao.any() else np.nan
        fn = don[an].mean() if an.any() else np.nan
        deltas[b] = fn - fo
    return [float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))]


def build_on(person, window_rows=3.0, dur_s=None):
    sub = pf.subject_by_name(person)
    tag = "m4_mosaic" if dur_s is None else f"m4_mosaic_d{int(dur_s)}"
    path = os.path.join(sub.cache_dir, f"{tag}.npz")
    if not os.path.exists(path):
        lm = pf.build_line_measurements(sub); ch = pf.build_chain(sub)
        pf.run_m4(sub, lm, ch, cache_path=path, rebuild=True, dur_s=dur_s,
                  mosaic_reject=True, mosaic_window_rows=window_rows)
    return {k: v for k, v in np.load(path).items()}


def run(person):
    L = labels_and_track(person)
    off = {k: v for k, v in np.load(os.path.join(L["sub"].cache_dir, "m4_dpf_physics.npz")).items()}
    on = build_on(person)
    ox, ov = off["x_px"].astype(float), off["valid"].astype(bool)
    nx, nv = on["x_px"].astype(float), on["valid"].astype(bool)
    s_off = score(L, ox, ov); s_on = score(L, nx, nv)
    r_off = r_dot(L, ox, ov); r_on = r_dot(L, nx, nv)
    ci = boot_flip_delta(L, ox, ov, nx, nv)
    g_flip = bool(s_on["flip_rate"] < s_off["flip_rate"] and ci[1] < 0)
    g_ms = bool(s_on["microsacc_preserved"] >= 0.95 * s_off["microsacc_preserved"])
    g_core = bool(s_on["core_rms"] <= s_off["core_rms"] * 1.05)
    g_r = bool(r_on >= r_off - 0.01)
    out = dict(person=person,
               off=dict(**s_off, r_dot_x=r_off), on=dict(**s_on, r_dot_x=r_on),
               flip_rate_delta=float(s_on["flip_rate"] - s_off["flip_rate"]),
               flip_delta_ci=ci,
               gate_flip_down=g_flip, gate_microsacc=g_ms, gate_core=g_core, gate_rdot=g_r,
               SIGNAL_SAFE_PASS=bool(g_flip and g_ms and g_core and g_r))
    print(f"[mosaic-AB {person}] flip {s_off['flip_rate']:.3f}->{s_on['flip_rate']:.3f} "
          f"(Δ{out['flip_rate_delta']:+.3f} CI{[round(c,3) for c in ci]}) "
          f"core {s_off['core_rms']:.3f}->{s_on['core_rms']:.3f}' "
          f"ms {s_off['microsacc_preserved']:.2f}->{s_on['microsacc_preserved']:.2f} "
          f"r {r_off:.3f}->{r_on:.3f} -> PASS={out['SIGNAL_SAFE_PASS']}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="+", default=["Igor", "Ashton3"])
    args = ap.parse_args()
    res = {p: run(p) for p in args.people}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "mosaic_ab.json"), "w"), indent=2)
    # power = the flip-rate delta with CI (powered iff CI excludes 0 and core unchanged)
    power = {p: dict(flip_rate_delta=res[p]["flip_rate_delta"], flip_delta_ci=res[p]["flip_delta_ci"],
                     core_off=res[p]["off"]["core_rms"], core_on=res[p]["on"]["core_rms"],
                     powered=bool(res[p]["flip_delta_ci"][1] < 0 and res[p]["gate_core"]))
             for p in res}
    json.dump(power, open(os.path.join(RESULTS, "mosaic_power.json"), "w"), indent=2)
    print("-> results/mosaic_ab.json results/mosaic_power.json")


if __name__ == "__main__":
    main()
