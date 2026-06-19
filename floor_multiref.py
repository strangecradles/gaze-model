"""floor_multiref.py — Phase 1 Task 1: multi-reference disagreement (C1).

For every line we localize the current frame against EACH of the chain-aligned previous
frames N-1, N-2, N-3 SEPARATELY (no averaging), all expressed in the current frame's
coordinate. Their disagreement is the reference-specific error C1; their mean is the
cross-reference COMMON track (real position + reference-common systematic C2 + chain).

  C1 = RMS over FOV lines of std_j(rdx_j)            [arcmin]
  common[line] = mean_j rdx_j                         [px, in current-frame coords]

`_rdx_against` is a faithful replica of people_fov_pf.build_line_measurements' per-frame
core (verified: j=1 reproduces the committed rdx). PER SUBJECT only — never pooled.

Emit results/floor_multiref.json and cache cache/people_fov/<sub>/lines_multiref.npz.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque

import cv2
import numpy as np

import data
import khz2d
import people_fov_pf as pf

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX
CROPV = pf.CROPV; RDY_MAX = pf.RDY_MAX; PADH = pf.PADH
Q_FOV = pf.Q_FOV; CONTRAST_FRAC = pf.CONTRAST_FRAC


def _rdx_against(cur_db: np.ndarray, prv_db: np.ndarray):
    """Faithful copy of the build_line_measurements core: localize cur vs one reference.
    Returns (rdx, qh) in px (current-frame coords). Vertical/along path recomputed per
    reference (each reference gets its own lam alignment), horizontal argmax on ±PADH."""
    Hc, W = cur_db.shape
    curz = khz2d._zcols(cur_db); prvz = khz2d._zcols(prv_db)
    npad = Hc + 2 * RDY_MAX
    A = np.fft.rfft(curz, n=npad, axis=0)
    B = np.fft.rfft(prvz, n=npad, axis=0)
    cc = np.fft.irfft(np.conj(A) * B, n=npad, axis=0)
    ccb = np.concatenate([cc[-RDY_MAX:], cc[:RDY_MAX + 1]], 0)
    pk = np.argmax(ccb, 0)
    lam = np.empty(W, np.float32)
    for c in range(W):
        lam[c] = khz2d._parab(ccb[:, c], int(pk[c])) - RDY_MAX
    idx = np.arange(Hc, dtype=np.float32)[:, None] - lam[None, :]
    i0 = np.clip(np.floor(idx).astype(np.int32), 0, Hc - 2)
    frac = np.clip(idx - i0, 0.0, 1.0)
    cols = np.arange(W)[None, :].repeat(Hc, 0)
    cural = cur_db[i0, cols] * (1 - frac) + cur_db[i0 + 1, cols] * frac
    curalz = khz2d._zcols(cural)
    P = (prvz.T @ curalz) / Hc
    di = np.arange(-PADH, PADH + 1)
    src = np.arange(W)[None, :] + di[:, None]
    bad = (src < 0) | (src >= W)
    src = np.clip(src, 0, W - 1)
    prof = P[src, np.arange(W)[None, :]]
    prof[bad] = -1.0
    prof = prof.T
    pk2 = np.argmax(prof, 1)
    rdx = np.empty(W, np.float32); qh = np.empty(W, np.float32)
    for c in range(W):
        k = int(pk2[c])
        rdx[c] = khz2d._parab(prof[c], k) - PADH
        qh[c] = prof[c, k]
    return rdx, qh


def _warp_ref(raw_ref: np.ndarray, dx: float, dy: float, shape) -> np.ndarray:
    H, W = shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(raw_ref, M, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def build_multiref(sub: pf.Subject, K: int = 3, rebuild: bool = False) -> dict:
    cache = os.path.join(sub.cache_dir, f"lines_multiref{K}.npz")
    if os.path.exists(cache) and not rebuild:
        return {k: v for k, v in np.load(cache).items()}
    ch = pf.build_chain(sub)
    fps = float(ch["fps"])
    posx = ch["x"]; posy = ch["y"]
    T, FR, COL, CON = [], [], [], []
    RDXJ = [[] for _ in range(K)]      # per-reference rdx
    QHJ = [[] for _ in range(K)]
    raw_hist: deque = deque(maxlen=K)
    nf = 0
    for f, raw in pf._read_frames(sub):
        nf = f + 1
        if not raw_hist:
            raw_hist.append(raw); continue
        H, W = raw.shape
        ok_f = bool(ch["ok"][f])
        cur_db = data._deband(raw)[CROPV:H - CROPV]
        Hc = cur_db.shape[0]
        rdxs = np.full((K, W), np.nan, np.float32); qhs = np.zeros((K, W), np.float32)
        if ok_f:
            for j in range(1, len(raw_hist) + 1):
                fk = f - j
                if fk < 1 or not bool(ch["ok"][fk]):
                    continue
                dx = -(float(posx[f]) - float(posx[fk]))
                dy = -(float(posy[f]) - float(posy[fk]))
                refw = _warp_ref(raw_hist[-j], dx, dy, (H, W))
                prv_db = data._deband(refw)[CROPV:H - CROPV]
                rdxj, qhj = _rdx_against(cur_db, prv_db)
                rdxs[j - 1] = rdxj; qhs[j - 1] = qhj
        con = raw[CROPV:H - CROPV].std(0)
        tline = f / fps + (np.arange(W) + 0.5) / W / fps
        T.append(tline.astype(np.float64)); FR.append(np.full(W, f, np.int32))
        COL.append(np.arange(W, dtype=np.int16)); CON.append(con.astype(np.float32))
        for j in range(K):
            RDXJ[j].append(rdxs[j]); QHJ[j].append(qhs[j])
        raw_hist.append(raw)
        if f % 200 == 0:
            print(f"  [{sub.name} multiref] frame {f}")
    out = dict(t=np.concatenate(T), frame=np.concatenate(FR), col=np.concatenate(COL),
               con=np.concatenate(CON),
               line_rate=np.float64(fps * 808), fps=np.float64(fps), K=np.int64(K))
    for j in range(K):
        out[f"rdx{j+1}"] = np.concatenate(RDXJ[j])
        out[f"qh{j+1}"] = np.concatenate(QHJ[j])
    np.savez(cache, **out)
    return out


def analyze(sub_name: str, K: int = 3) -> dict:
    sub = pf.subject_by_name(sub_name)
    mr = build_multiref(sub, K=K)
    W = 808
    con = mr["con"]; con_med = np.median(con)
    R = np.stack([mr[f"rdx{j+1}"].astype(float) for j in range(K)], 0)   # (K, N)
    Q = np.stack([mr[f"qh{j+1}"].astype(float) for j in range(K)], 0)
    # per-line validity: every reference must overlap/lock (qh>Q_FOV) + contrast
    good_ref = Q > Q_FOV
    line_valid = (good_ref.all(0)) & (con > CONTRAST_FRAC * con_med) & np.isfinite(R).all(0)
    n_valid = int(line_valid.sum())
    Rv = R[:, line_valid]
    # C1 = RMS over lines of the cross-reference std (per line)
    per_line_std = Rv.std(0, ddof=1)
    C1_px = float(np.sqrt(np.mean(per_line_std ** 2)))
    C1_arc = C1_px * ARC
    common = R.mean(0)                       # cross-reference common track (px, cur coords)
    out = dict(person=sub_name, K=K, n_valid=n_valid,
               n_total=int(line_valid.size),
               C1_arcmin=C1_arc, C1_px=C1_px,
               note=("C1 = RMS over FOV lines of std across refs N-1..N-K (separate, "
                     "not averaged); slight OVER-estimate (cross-frame content drift). "
                     "common track saved to lines_multiref cache as rdx mean."))
    print(f"[multiref {sub_name}] C1 = {C1_arc:.3f}' (n_valid={n_valid}/{line_valid.size}, "
          f"K={K})")
    return out, common, line_valid, mr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="+", default=["Igor", "Ashton3"])
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()
    res = {}
    for p in args.people:
        o, _, _, _ = analyze(p, args.K)
        res[p] = o
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "floor_multiref.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("-> results/floor_multiref.json")


if __name__ == "__main__":
    main()
