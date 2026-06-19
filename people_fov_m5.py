"""people_fov_m5.py — batch MAP smoother adapter for people_data_fov captures.

Runs M5-style offline trajectory smoothing on the per-line profile cache
(Viterbi init + torch Adam MAP). Compare jitter vs M4 on Ashton3.

Usage:
  python people_fov_m5.py --person Ashton3
  python people_fov_m5.py --person Ashton3 --compare
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

import calib
import khz2d
import khz2d_methods as km
import losses
import people_fov_pf as pf

PADH = khz2d.PADH


def _kalman_y(lm: dict, rate: float, pxdeg: float) -> tuple[np.ndarray, np.ndarray]:
    fr = lm["frame"]
    okl = lm["ok"].astype(bool)
    con_ok = lm["con"] > khz2d.CONTRAST_FRAC * np.median(lm["con"])
    meas_y = np.where(okl & con_ok, lm["chain_y"][fr] + lm["lam_v"], np.nan)
    sigma_a = 3000.0 * pxdeg
    ey, uy = km._kf_axis(lm["t"], meas_y, lm["qv"], 0.20, 1.5, sigma_a,
                         gate_px=15.0, reacq_steps=int(rate * 0.005))

    def recent(u, win):
        idx = np.arange(len(u)); last = np.where(u, idx, -10**9)
        last = np.maximum.accumulate(last)
        return (idx - last) <= win

    win = int(rate * 0.030)
    valid = recent(uy, win) & np.isfinite(ey)
    return ey, valid


def _viterbi_d(lm: dict, rate: float, beta: float = 8.0,
               trans_gamma: float = 0.05) -> np.ndarray:
    pxdeg = 126.0 / 60.0  # horizontal px per arcmin (people raster scale)
    vmax_px = 826.0 * pxdeg
    Wt = int(np.ceil(vmax_px / rate)) + 2
    nS = 2 * PADH + 1
    fr = lm["frame"]; col = lm["col"]
    okl = lm["ok"].astype(bool)
    con_ok = lm["con"] > khz2d.CONTRAST_FRAC * np.median(lm["con"])
    use = okl & con_ok & (lm["qh"] > 0.05)
    prof = lm["prof"]
    N = prof.shape[0]
    inc_x = np.diff(lm["chain_x"], prepend=lm["chain_x"][0])
    off = np.arange(-Wt, Wt + 1)
    pen = (trans_gamma * np.abs(off)).astype(np.float32)
    score = np.zeros(nS, np.float32)
    bp = np.empty((N, nS), np.int8)
    NEG = np.float32(-1e9)
    pad = np.full(Wt, NEG, np.float32)
    for i in range(N):
        if i > 0 and col[i] == 0:
            k = int(round(inc_x[fr[i]]))
            if abs(k) >= nS:
                score = np.zeros(nS, np.float32)
            elif k > 0:
                score = np.concatenate([score[k:], np.full(k, score.min(), np.float32)])
            elif k < 0:
                score = np.concatenate([np.full(-k, score.min(), np.float32), score[:k]])
        sw = np.lib.stride_tricks.sliding_window_view(
            np.concatenate([pad, score, pad]), 2 * Wt + 1) - pen[None, :]
        am = np.argmax(sw, 1)
        score = sw[np.arange(nS), am]
        bp[i] = (am - Wt).astype(np.int8)
        if use[i]:
            score = score + beta * prof[i].astype(np.float32)
    d = np.empty(N, np.float32)
    j = int(np.argmax(score))
    for i in range(N - 1, -1, -1):
        d[i] = j - PADH
        j = j + int(bp[i, j])
        if i > 0 and col[i] == 0:
            k = int(round(inc_x[fr[i]]))
            j = int(np.clip(j + k, 0, nS - 1))
        j = int(np.clip(j, 0, nS - 1))
    pr = prof.astype(np.float32)
    ji = (d + PADH).astype(int)
    iN = np.arange(N)
    inner = (ji > 0) & (ji < nS - 1) & use
    a = pr[iN, np.clip(ji - 1, 0, nS - 1)]; b = pr[iN, ji]
    cc = pr[iN, np.clip(ji + 1, 0, nS - 1)]
    den = a - 2 * b + cc
    subs = np.where(inner & (np.abs(den) > 1e-12),
                    0.5 * (a - cc) / np.where(den == 0, 1, den), 0.0)
    return d + np.clip(subs, -1, 1)


def px_per_deg_people(lm: dict, refs: dict) -> float:
    f = pf.fov_mask(lm)
    pos = lm["chain_x"][lm["frame"]] + lm["rdx"]
    dot_x = np.interp(lm["t"], refs["dot_t"] + refs["off"], refs["dot_x"])
    m = f & np.isfinite(pos)
    if m.sum() > 20:
        A = np.c_[pos[m], np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(A, dot_x[m], rcond=None)
        pxdeg = abs(60.0 / coef[0]) if abs(coef[0]) > 1e-9 else 126.0 / 60.0
    else:
        pxdeg = 126.0 / 60.0
    return float(pxdeg)


def m5_map_people(name: str, *, iters: int = 300, lr: float = 0.3,
                  w_data: float = 1.0, w_dyn: float = 0.5, w_anchor: float = 2.0,
                  beta: float = 4.0, rebuild: bool = False,
                  cache_tag: str = "m5_map") -> dict:
    sub = pf.subject_by_name(name)
    out_path = os.path.join(sub.cache_dir, f"{cache_tag}.npz")
    if os.path.exists(out_path) and not rebuild:
        z = np.load(out_path)
        return {k: z[k] for k in z.files}

    lm = pf.build_line_measurements(sub)
    refs = pf.compute_refs(sub, lm)
    rate = float(lm["line_rate"])
    pxdeg = px_per_deg_people(lm, refs)
    px2row = float(calib.ROWS_PER_DEG) / pxdeg
    fr = lm["frame"]
    chx = lm["chain_x"][fr].astype(np.float64)
    okl = lm["ok"].astype(bool)
    con_ok = lm["con"] > khz2d.CONTRAST_FRAC * np.median(lm["con"])
    use = okl & con_ok & (lm["qh"] > 0.10)
    N = len(chx)
    nS = 2 * PADH + 1

    d_init = _viterbi_d(lm, rate)
    prof = torch.from_numpy(lm["prof"].astype(np.float32))
    w = torch.from_numpy(use.astype(np.float32))
    d = torch.tensor(d_init.astype(np.float64), dtype=torch.float64, requires_grad=True)
    chx_t = torch.from_numpy(chx)
    tt = torch.from_numpy(lm["t"].astype(np.float64))
    anchor_rows = (lm["chain_x"] * px2row).astype(np.float64)
    anchor_t = lm["chain_t"].astype(np.float64)
    iN = torch.arange(N)

    def pooled(x, k=64):
        n = (x.shape[0] // k) * k
        return x[:n].reshape(-1, k).mean(1)

    opt = torch.optim.Adam([d], lr=lr)
    t0 = time.time()
    for it in range(iters):
        opt.zero_grad()
        idx = torch.clamp(d + PADH, 0.0, nS - 1.001)
        i0 = idx.floor().long(); frac = (idx - i0).float()
        val = prof[iN, i0] * (1 - frac) + prof[iN, torch.clamp(i0 + 1, max=nS - 1)] * frac
        l_data = -(beta * val * w).sum() / w.sum()
        pos_rows = (d + chx_t) * px2row
        l_dy = losses.l_dyn(pos_rows, tt)
        l_an = losses.l_anchor(pooled(pos_rows), pooled(tt), anchor_rows, anchor_t)
        loss = w_data * l_data + w_dyn * l_dy + w_anchor * l_an
        loss.backward()
        opt.step()
        if it % 50 == 0:
            print(f"  [m5 {name}] it {it}: data {float(l_data.detach()):+.4f} "
                  f"({time.time()-t0:.0f}s)")

    d_np = d.detach().numpy()
    pos_x = chx + d_np
    ey, yvalid = _kalman_y(lm, rate, pxdeg)
    valid = use & yvalid
    out = dict(t=lm["t"], x_px=pos_x, y_px=ey, valid=valid,
               rate=np.float64(rate), d=d_np.astype(np.float64))
    np.savez(out_path, **out)
    return out


def compare_m4_m5(name: str) -> None:
    sub = pf.subject_by_name(name)
    refs = pf.compute_refs(sub, pf.build_line_measurements(sub))
    m4_path = os.path.join(sub.cache_dir, "m4_dpf_physics.npz")
    m5_path = os.path.join(sub.cache_dir, "m5_map.npz")
    if not os.path.exists(m4_path):
        raise FileNotFoundError(f"missing {m4_path} — run PF first")
    if not os.path.exists(m5_path):
        m5_map_people(name)

    def _metrics(label, path):
        z = np.load(path)
        run = {k: z[k] for k in z.files}
        valid = run["valid"].astype(bool)
        rate = float(run["rate"])
        ev = pf.evaluate(run["t"], run["x_px"], run["y_px"], valid, rate, refs)
        hx = ev["cal_x"]
        p_low, p_hf = pf.psd_band_ratio(hx, rate, valid)
        return dict(
            label=label, prec_x=ev["prec_x"], r_dot_x=ev["r_dot_x"],
            jitter_30=pf.frame_jitter_30fps(hx, rate, valid),
            psd_pursuit=p_low, psd_hf=p_hf, valid_frac=ev["valid_frac"],
        )

    m4 = _metrics("M4", m4_path)
    m5 = _metrics("M5", m5_path)
    print(f"\n=== {name}: M4 vs M5 jitter comparison ===")
    for k in ("prec_x", "jitter_30", "r_dot_x", "psd_hf", "valid_frac"):
        print(f"  {k:12s}  M4={m4[k]:.3f}  M5={m5[k]:.3f}  "
              f"ratio={m5[k]/m4[k]:.2f}" if m4[k] and m5[k] else f"  {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Ashton3")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()
    m5_map_people(args.person, iters=args.iters, rebuild=args.rebuild)
    if args.compare:
        compare_m4_m5(args.person)


if __name__ == "__main__":
    main()
