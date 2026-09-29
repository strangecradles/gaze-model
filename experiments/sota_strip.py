"""sota_strip.py — faithful Stevenson & Roorda strip-based registration to a
COMPOSITE (synthetic) reference frame, the TSLO/AOSLO state-of-the-art.

References
----------
- Stevenson SB & Roorda A (2005), "Correcting for miniature eye movements in
  high resolution scanning laser ophthalmoscopy", Proc. SPIE 5688.
- Sheehy CK, Yang Q, Arathorn DW, Tiruveedhula P, de Boer JF, Roorda A (2012),
  "High-speed, image-based eye tracking with a scanning laser ophthalmoscope",
  Biomed. Opt. Express 3(10):2611-2622.  (TSLO; ~960 Hz at 0.66' in AOSLO.)
- Bowers NR, Boehm AE, Roorda A (2019)/(2020) refinements; the "robust strip
  based digital image registration" pipeline (Roorda lab, BOE).
- Liu Z, et al. (2024), substrip variant (two sub-windows at strip ends).

Algorithm (faithful, see results/sota_comparison.md for the spec + simplifications):
  1. Pre-processing: per-frame CLAHE contrast enhancement; de-band; blink /
     low-intensity frame rejection; distortion-frame rejection via consecutive
     full-frame match quality (the chain `q`) below a threshold.
  2. Composite reference: accepted low-distortion frames are averaged, each
     placed at its global (full-frame-registered) position, into an oversized
     composite (a retinal mosaic = THE synthetic reference). Registration is to
     this FIXED composite, not to the previous frame (the defining difference
     from incremental tracking).
  3. Strip registration: each frame is divided into strips of S adjacent columns
     (a strip is parallel to the fast scanner, which on this raster is a column).
     Each strip is NCC-matched (TM_CCOEFF_NORMED) against the composite within a
     local search window; sub-pixel peak by 2D parabolic interpolation; the strip
     is accepted iff its NCC peak exceeds a threshold.
  4. High-rate trace: per-strip (x, y) offsets in temporal order -> eye trace at
     (808 / S) * 14.633 Hz.

The output obeys the standard khz2d method contract (t, x_px, y_px, valid, rate)
so khz2d.load_method / khz2d.evaluate / khz2d.summarize work unchanged.

GLOBAL COORDINATE / "reference selection" simplification (stated, not a handicap):
the composite is built in the coordinate of khz2d.chain() — the robust
incremental full-frame (strip-median) registration that every method already
uses as its 20 Hz absolute anchor. Using it as the mosaic coordinate is the
standard montaging step and gives the SOTA the SAME coarse anchor as M1/M4,
isolating the only thing under test: the high-rate per-strip residual estimator
(composite-reference NCC vs previous-frame NCC vs particle filter).
"""
from __future__ import annotations

import os
import time
import numpy as np
import cv2

import data
import khz2d

# composite-build thresholds
COMP_Q = 0.45        # only frames with chain match-quality >= this enter the composite
LOWINT_FRAC = 0.5    # reject frame if mean |deband| < this * median (blink / dropout)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

COMP_CACHE = os.path.join(khz2d.CACHE, "sota_composite.npz")


def _enhance(raw: np.ndarray) -> np.ndarray:
    """Per-frame contrast enhancement: CLAHE on the raw frame, then de-band,
    then z-score (zero-mean/unit-std) for TM_CCOEFF_NORMED stability."""
    u8 = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    eq = _CLAHE.apply(u8).astype(np.float32)
    g = data._deband(eq)
    return ((g - g.mean()) / (g.std() + 1e-9)).astype(np.float32)


def build_composite(rebuild: bool = False):
    """Average accepted low-distortion frames at their global positions into an
    oversized composite (the synthetic reference). Returns dict with the
    composite image and the margins (My, Mx) mapping chain coords -> composite
    pixels: composite[r + chain_y + My, c + chain_x + Mx]."""
    if os.path.exists(COMP_CACHE) and not rebuild:
        z = np.load(COMP_CACHE)
        return {k: z[k] for k in z.files}
    ch = khz2d.chain()
    cx = ch["x"]; cy = ch["y"]; ok = ch["ok"].astype(bool); q = ch["q"]

    # first pass: per-frame enhanced frame mean-abs for blink rejection,
    # and collect frame dims.
    enh = {}
    means = []
    H = W = None
    for f, raw in khz2d._read_frames():
        if f >= len(cx):
            break
        zf = _enhance(raw)
        if H is None:
            H, W = zf.shape
        enh[f] = zf
        means.append(np.mean(np.abs(data._deband(raw))))
    means = np.asarray(means)
    med = np.median(means)
    not_blink = means > LOWINT_FRAC * med

    use = ok & not_blink & (q >= COMP_Q)
    use[0] = use[0] and ok[0]  # frame 0 has no chain increment; keep if ok

    # canvas sizing from the global positions of the USED frames
    ix = np.round(cx).astype(int); iy = np.round(cy).astype(int)
    uf = np.where(use)[0]
    Mx = int(-min(0, ix[uf].min()) + 4)
    My = int(-min(0, iy[uf].min()) + 4)
    CW = int(Mx + ix[uf].max() + W + 4)
    CH = int(My + iy[uf].max() + H + 4)

    acc = np.zeros((CH, CW), np.float32)
    cnt = np.zeros((CH, CW), np.float32)
    for f in uf:
        oy = iy[f] + My; ox = ix[f] + Mx
        acc[oy:oy + H, ox:ox + W] += enh[f]
        cnt[oy:oy + H, ox:ox + W] += 1.0
    comp = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), 0.0).astype(np.float32)

    # anchor (reference) frame = highest-quality used frame near the middle (report only)
    mid = len(cx) // 2
    cand = uf[np.argsort(-(q[uf] - 0.001 * np.abs(uf - mid)))]
    anchor = int(cand[0]) if len(cand) else int(uf[len(uf) // 2])

    out = dict(comp=comp, My=np.int64(My), Mx=np.int64(Mx),
               H=np.int64(H), W=np.int64(W), n_used=np.int64(len(uf)),
               n_total=np.int64(len(cx)), anchor=np.int64(anchor),
               cov_frac=np.float32((cnt > 0).mean()))
    np.savez(COMP_CACHE, **out)
    print(f"  [composite] {len(uf)}/{len(cx)} frames -> {CH}x{CW} "
          f"(anchor frame {anchor}, coverage {(cnt>0).mean()*100:.0f}%)")
    return out


def _parab2d(r):
    """Sub-pixel (dy, dx) offset of the peak of correlation map r (separable
    parabolic through the integer peak), plus the peak value."""
    j, i = np.unravel_index(int(np.argmax(r)), r.shape)  # j=row, i=col
    peak = float(r[j, i])
    dy = khz2d._parab(r[:, i], j) - j if 0 < j < r.shape[0] - 1 else 0.0
    dx = khz2d._parab(r[j, :], i) - i if 0 < i < r.shape[1] - 1 else 0.0
    return j, i, dy, dx, peak


def sota_roorda(S: int, pad: int = 60, ncc_thr: float = 0.35,
                dur_s: float | None = None, rebuild: bool = False):
    """Composite-reference strip registration at strip width S.

    Each frame's S-column strips are NCC-matched to the FIXED composite within
    +-pad px of their predicted (chain) position; accepted iff peak NCC>ncc_thr.
    Output rate = (808 // S) * fps.  dur_s caps the run (line rate S=1 is heavy).
    """
    tag = f"sota_s{S}" + (f"_d{int(dur_s)}" if dur_s is not None else "")
    c = khz2d.load_method(tag)
    if c is not None and not rebuild:
        return c
    comp = build_composite()
    # zero-pad the composite by `pad` so the +-pad strip search never runs off
    # the canvas edge (margins are shifted accordingly).
    C = np.pad(comp["comp"], pad)
    My = int(comp["My"]) + pad; Mx = int(comp["Mx"]) + pad
    CH, CW = C.shape
    ch = khz2d.chain()
    cx = ch["x"]; cy = ch["y"]; ok = ch["ok"].astype(bool); fps = float(ch["fps"])

    T, X, Y, V, NCC = [], [], [], [], []
    t0 = time.time()
    nfr = len(cx)
    for f, raw in khz2d._read_frames():
        if f >= nfr:
            break
        if dur_s is not None and f / fps > dur_s:
            break
        if not ok[f]:
            continue
        zf = _enhance(raw)
        H, W = zf.shape
        nstrip = W // S
        bry = int(round(cy[f])) + My          # predicted composite row of frame top
        brx = int(round(cx[f])) + Mx          # predicted composite col of frame col0
        for s in range(nstrip):
            c0 = s * S
            tmpl = zf[:, c0:c0 + S]
            ry0 = bry - pad; rx0 = brx + c0 - pad
            ry1 = bry + H + pad; rx1 = brx + c0 + S + pad
            t_line = (f + (c0 + S / 2.0) / W) / fps
            if ry0 < 0 or rx0 < 0 or ry1 > CH or rx1 > CW:
                T.append(t_line); X.append(np.nan); Y.append(np.nan)
                V.append(False); NCC.append(0.0)
                continue
            region = C[ry0:ry1, rx0:rx1]
            r = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
            j, i, dy, dx, peak = _parab2d(r)
            # composite top-left of the matched strip:
            mr = ry0 + j + dy
            mc = rx0 + i + dx
            # eye position = matched_mosaic_pos - nominal_image_pos - margins
            y_px = mr - My
            x_px = mc - Mx - c0
            # store the measured position regardless of threshold so the run
            # can be re-thresholded from the cached max_ncc without re-running.
            T.append(t_line); X.append(x_px); Y.append(y_px)
            V.append(peak > ncc_thr); NCC.append(peak)
        if f % 100 == 0:
            el = time.time() - t0
            print(f"  [sota S={S}] frame {f}/{nfr} ({el:.0f}s) "
                  f"valid~{np.mean(V[-nstrip:])*100:.0f}%")
    T = np.asarray(T); X = np.asarray(X); Y = np.asarray(Y)
    V = np.asarray(V) & np.isfinite(X) & np.isfinite(Y)
    rate = (808 // S) * fps
    return khz2d.save_method(tag, T, X, Y, V, rate,
                             extra=dict(max_ncc=np.asarray(NCC), S=np.int64(S),
                                        pad=np.int64(pad),
                                        ncc_thr=np.float64(ncc_thr)))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--pad", type=int, default=40)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--composite", action="store_true", help="(re)build composite only")
    a = ap.parse_args()
    if a.composite:
        build_composite(rebuild=True)
        raise SystemExit
    r = sota_roorda(a.S, pad=a.pad, ncc_thr=a.thr, dur_s=a.dur, rebuild=a.rebuild)
    ev = khz2d.evaluate(r["t"], r["x_px"], r["y_px"], r["valid"].astype(bool),
                        float(r["rate"]), f"sota_s{a.S}", smooth_ms=2)
    print(khz2d.summarize(ev))
