"""h2_crossref.py — definitive signal-safety test for multi-frame reference averaging.

Localize each current line against TWO staggered averaged references:
  refA = mean(N-1, N-2, N-3)   refB = mean(N-2, N-3, N-4)
Their coherence (rdxA vs rdxB) is SIMULTANEOUS (same line/instant) and NON-atlas-shared
(different averaged references) — the only valid coherence split. Compare the crossover to
the single-frame cross-reference (rdx1 vs rdx2, Phase 1). If averaging preserves the
coherent (real-motion) band, refavg is signal-safe; if the crossover drops, averaging blurs
real HF motion (the registration-blur risk). PER SUBJECT.

Emit results/h2_crossref.json + cache lines_avgrefpair.npz.
"""
from __future__ import annotations

import json
import os
from collections import deque

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

import data
import people_fov_pf as pf
import floor_multiref as fm
import floor_decompose as fdc

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX
CROPV = pf.CROPV

DEFS = ([1, 2, 3], [2, 3, 4])     # refA lags, refB lags (staggered)


def _avg_ref(raw_hist, ch, f, lags, shape):
    """Mean of chain-aligned raw frames at the given lags, deband+crop -> reference image."""
    H, W = shape
    px, py = float(ch["x"][f]), float(ch["y"][f])
    acc = np.zeros((H, W), np.float32); n = 0
    for j in lags:
        fk = f - j
        if fk < 1 or not bool(ch["ok"][fk]) or j > len(raw_hist):
            continue
        dx = -(px - float(ch["x"][fk])); dy = -(py - float(ch["y"][fk]))
        acc += fm._warp_ref(raw_hist[-j], dx, dy, (H, W)); n += 1
    if n == 0:
        return None
    return data._deband(acc / n)[CROPV:H - CROPV]


def build(sub, rebuild=False):
    cache = os.path.join(sub.cache_dir, "lines_avgrefpair.npz")
    if os.path.exists(cache) and not rebuild:
        return {k: v for k, v in np.load(cache).items()}
    ch = pf.build_chain(sub); fps = float(ch["fps"])
    maxlag = max(max(DEFS[0]), max(DEFS[1]))
    raw_hist = deque(maxlen=maxlag)
    T, FR, COL, CON, RA, RB, QA, QB = [], [], [], [], [], [], [], []
    for f, raw in pf._read_frames(sub):
        if len(raw_hist) < maxlag:
            raw_hist.append(raw)
            if len(raw_hist) < maxlag:
                continue
        H, W = raw.shape
        cur_db = data._deband(raw)[CROPV:H - CROPV]
        ok_f = bool(ch["ok"][f])
        ra = np.full(W, np.nan, np.float32); rb = np.full(W, np.nan, np.float32)
        qa = np.zeros(W, np.float32); qb = np.zeros(W, np.float32)
        if ok_f:
            refA = _avg_ref(raw_hist, ch, f, DEFS[0], (H, W))
            refB = _avg_ref(raw_hist, ch, f, DEFS[1], (H, W))
            if refA is not None:
                ra, qa = fm._rdx_against(cur_db, refA)
            if refB is not None:
                rb, qb = fm._rdx_against(cur_db, refB)
        con = raw[CROPV:H - CROPV].std(0)
        T.append((f / fps + (np.arange(W) + 0.5) / W / fps).astype(np.float64))
        FR.append(np.full(W, f, np.int32)); COL.append(np.arange(W, dtype=np.int16))
        CON.append(con.astype(np.float32))
        RA.append(ra); RB.append(rb); QA.append(qa); QB.append(qb)
        raw_hist.append(raw)
        if f % 200 == 0:
            print(f"  [{sub.name} avgrefpair] frame {f}")
    out = dict(t=np.concatenate(T), frame=np.concatenate(FR), col=np.concatenate(COL),
               con=np.concatenate(CON), rdxA=np.concatenate(RA), rdxB=np.concatenate(RB),
               qhA=np.concatenate(QA), qhB=np.concatenate(QB),
               line_rate=np.float64(fps * 808), fps=np.float64(fps))
    np.savez(cache, **out)
    return out


def analyze(person):
    sub = pf.subject_by_name(person)
    d = build(sub)
    fs = float(d["line_rate"]); t = d["t"].astype(float); col = d["col"]
    con = d["con"]; con_med = np.median(con)
    m = (d["qhA"] > pf.Q_FOV) & (d["qhB"] > pf.Q_FOV) & (con > pf.CONTRAST_FRAC * con_med) \
        & np.isfinite(d["rdxA"]) & np.isfinite(d["rdxB"])
    avg = fdc.crossref_coherence(t, d["rdxA"].astype(float), d["rdxB"].astype(float), m, fs)
    # single-frame cross-ref from Phase 1 multiref cache for comparison
    mr = fm.build_multiref(sub)
    cons = mr["con"]; cm = np.median(cons)
    ms = (mr["qh1"] > pf.Q_FOV) & (mr["qh2"] > pf.Q_FOV) & (cons > pf.CONTRAST_FRAC * cm) \
        & np.isfinite(mr["rdx1"]) & np.isfinite(mr["rdx2"])
    single = fdc.crossref_coherence(mr["t"].astype(float), mr["rdx1"].astype(float),
                                    mr["rdx2"].astype(float), ms, fs)
    cross_down = bool(np.isfinite(avg["crossover_hz"]) and np.isfinite(single["crossover_hz"])
                      and avg["crossover_hz"] < single["crossover_hz"] - 1e-6)
    out = dict(person=person,
               single_ref_crossref=single, avg_ref_crossref=avg,
               crossover_single_hz=single["crossover_hz"], crossover_avg_hz=avg["crossover_hz"],
               averaging_blurs_real_motion=cross_down,
               signal_safe=bool(not cross_down),
               note=("Valid signal-safety: cross-ref coherence of two STAGGERED averaged "
                     "references (simultaneous, non-atlas-shared). If avg crossover < single "
                     "crossover, averaging blurs real HF motion (registration blur)."))
    print(f"[crossref {person}] single cross={single['crossover_hz']:.0f}Hz "
          f"coh(0-10/10-50)={single['coh_0_10hz']:.2f}/{single['coh_10_50hz']:.2f} | "
          f"avg cross={avg['crossover_hz']:.0f}Hz coh={avg['coh_0_10hz']:.2f}/{avg['coh_10_50hz']:.2f} "
          f"-> signal_safe={out['signal_safe']}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="+", default=["Igor", "Ashton3"])
    args = ap.parse_args()
    res = {p: analyze(p) for p in args.people}
    json.dump(res, open(os.path.join(RESULTS, "h2_crossref.json"), "w"), indent=2)
    print("-> results/h2_crossref.json")


if __name__ == "__main__":
    main()
