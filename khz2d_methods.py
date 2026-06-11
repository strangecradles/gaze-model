"""khz2d_methods.py — the candidate kHz 2D gaze reconstruction methods (M0-M5).

Every method consumes the shared harness measurements (khz2d.line_measurements)
and/or its own streaming pass over the test1 raster, and emits the standard
contract dict(t, x_px, y_px, valid, rate) saved under cache/khz2d_<name>.npz.

  M0  chain          : the 2D SLO frames alone (incremental registration, ~15 Hz)
                       — the "slow axis only" baseline every kHz method must beat.
  M1  strips S=...   : TSLO-style 2D template matching of S-column strips against
                       the previous frame (the validated gaze_anim_raster method),
                       swept S=20..1 -> 0.6..11.8 kHz.
  M2  kalman         : per-line channel-split fusion. Vertical = trusted 1D shift;
                       horizontal = per-line profile argmax; each fused with the
                       frame-anchored chain by a constant-velocity Kalman filter
                       with quality-scaled measurement noise + innovation gating
                       (mislocked alias peaks get rejected, coast + reacquire).
  M3  viterbi        : global MAP decode of the per-line horizontal profile
                       sequence: states = residual offsets, transitions capped by
                       the saccade main-sequence VMAX, anchored to the chain at
                       every frame boundary. The offline-optimal alias resolver.
  M4  dpf            : the existing particle filter (filter.py, G10-G14 verbatim)
                       stepped per effective line; the frozen-physics decoder
                       renders from the most recent SLO frame (per-step atlas
                       window in gaze-residual coordinates), trusted along =
                       vertical channel, 20 Hz coarse anchor = the chain.
  M5  map            : batch MAP smoother — torch gradient descent on
                       data(profile NCC) + losses.l_dyn + losses.l_anchor over
                       the full trajectory at line rate (decoder-render data term
                       precomputed on the offset grid = the same L_recon NCC).

Horizontal (slow axis) is the hard/aliased axis; M3/M5 reuse M2's vertical
channel (the trusted one) so the comparison isolates the horizontal question.
"""
from __future__ import annotations

import os
import time
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d, median_filter

import data
import calib
import khz2d

PADH = khz2d.PADH


def _lm():
    return khz2d.line_measurements()


def px_per_deg(lm=None) -> float:
    """Horizontal px/deg of the test1 raster, measured by the affine fit of the
    raw per-line series to the dot (harness-level, method-independent)."""
    p = os.path.join(khz2d.CACHE, "khz2d_pxdeg.npy")
    if os.path.exists(p):
        return float(np.load(p))
    lm = lm or _lm()
    f = khz2d.fov_mask(lm)
    pos = lm["chain_x"][lm["frame"]] + lm["rdx"]
    ev = khz2d.evaluate(lm["t"], pos, lm["chain_y"][lm["frame"]] + lm["lam_v"],
                        f, float(lm["line_rate"]), "raw", smooth_ms=2)
    v = 60.0 / abs(ev["gain_x"])
    np.save(p, v)
    return float(v)


# ---------------------------------------------------------------------------
# M0 — slow-axis-only baseline
# ---------------------------------------------------------------------------


def m0_chain():
    name = "m0_chain"
    c = khz2d.load_method(name)
    if c is not None:
        return c
    ch = khz2d.chain()
    return khz2d.save_method(name, ch["t"], ch["x"], ch["y"],
                             ch["ok"].astype(bool), ch["fps"])


# ---------------------------------------------------------------------------
# M1 — strip tracking (gaze_anim_raster method), strip-width sweep
# ---------------------------------------------------------------------------


def _nz(x):
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def m1_strips(S: int, pad: int = 80, max_frames: int | None = None,
              rebuild: bool = False):
    """TSLO-style strip tracking with S-column strips (joint 2D matchTemplate
    against the previous frame). Faithful port of gaze_anim_raster.track."""
    name = f"m1_s{S}"
    c = khz2d.load_method(name)
    if c is not None and not rebuild:
        return c
    cap = cv2.VideoCapture(data._slo_path(khz2d.WHICH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    DY, DX, Q, CON, T = [], [], [], [], []
    cumy = cumx = 0.0
    prev = None
    f = 0
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok or (max_frames is not None and f >= max_frames):
            break
        raw = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = data._deband(raw)
        cur = _nz(g)
        if prev is not None:
            H, W = cur.shape
            nstrip = W // S
            refp = cv2.copyMakeBorder(prev, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=0)
            fy = np.zeros(nstrip); fx = np.zeros(nstrip)
            fq = np.zeros(nstrip); fc = np.zeros(nstrip)
            for s in range(nstrip):
                cc = s * S
                strip = cur[:, cc:cc + S]
                reg = refp[:, cc:cc + S + 2 * pad]
                r = cv2.matchTemplate(reg, strip, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                fy[s] = loc[1] - pad; fx[s] = loc[0] - pad; fq[s] = mx
                fc[s] = raw[:, cc:cc + S].std()
            my = np.median(fy); mxd = np.median(fx)
            cumy += my; cumx += mxd
            DY.extend(cumy + (fy - my)); DX.extend(cumx + (fx - mxd))
            Q.extend(fq); CON.extend(fc)
            T.extend((f + (np.arange(nstrip) * S + S / 2) / W) / fps)
        prev = cur
        f += 1
        if f == 6 and max_frames is None:
            per = (time.time() - t0) / 5.0
            est = per * 1024
            print(f"  [m1 S={S}] ~{per:.2f}s/frame -> est {est/60:.1f} min total")
    cap.release()
    DY = np.array(DY); DX = np.array(DX); Q = np.array(Q); CON = np.array(CON)
    T = np.array(T)
    rate = (808 // S) * fps
    fov = (Q > 0.35) & (CON > khz2d.CONTRAST_FRAC * np.median(CON))
    return khz2d.save_method(name, T, DX, DY, fov, rate,
                             extra=dict(q=Q, con=CON, S=np.int64(S)))


# ---------------------------------------------------------------------------
# M2 — channel-split constant-velocity Kalman fusion
# ---------------------------------------------------------------------------


def _kf_axis(t, meas, q, q_floor, sigma0, sigma_a, gate_px, reacq_steps,
             q_strong=0.45):
    """Scalar constant-velocity KF over an absolute per-line measurement stream.

    Quality-gated (q <= q_floor skipped), innovation-gated (rejects alias
    mislocks), with median-reacquisition from recent STRONG measurements after
    ``reacq_steps`` consecutive rejections (no indefinite coast).
    Returns (est, ok_updated); reacquisitions do NOT count as updates."""
    n = len(meas)
    est = np.empty(n); upd = np.zeros(n, bool)
    x = np.array([np.nan, 0.0])
    P = np.diag([1e4, 1e6])
    rej = 0
    hist = []
    tprev = t[0]
    for i in range(n):
        dt = max(t[i] - tprev, 1e-6); tprev = t[i]
        # predict
        x[0] += x[1] * dt
        P[0, 0] += 2 * P[0, 1] * dt + P[1, 1] * dt * dt + 0.25 * sigma_a**2 * dt**4
        P[0, 1] += P[1, 1] * dt + 0.5 * sigma_a**2 * dt**3
        P[1, 0] = P[0, 1]
        P[1, 1] += sigma_a**2 * dt**2
        z = meas[i]; qq = q[i]
        good = np.isfinite(z) and qq > q_floor
        if good and qq > q_strong:
            hist.append(z)
            if len(hist) > 9:
                hist.pop(0)
        if np.isnan(x[0]) and good:
            x[0] = z; P[0, 0] = sigma0**2
            est[i] = x[0]; upd[i] = True
            continue
        if good:
            sig = sigma0 / max(qq, 0.1)
            innov = z - x[0]
            s = P[0, 0] + sig**2
            if abs(innov) < max(gate_px, 6.0 * np.sqrt(s)):
                k0 = P[0, 0] / s; k1 = P[0, 1] / s
                x[0] += k0 * innov; x[1] += k1 * innov
                P00 = (1 - k0) * P[0, 0]
                P01 = (1 - k0) * P[0, 1]
                P11 = P[1, 1] - k1 * P[0, 1]
                P = np.array([[P00, P01], [P01, P11]])
                upd[i] = True
                rej = 0
            else:
                rej += 1
                if rej > reacq_steps and len(hist) >= 5:
                    x[0] = float(np.median(hist)); x[1] = 0.0
                    P = np.diag([sigma0**2 * 25, 1e6])
                    rej = 0
        est[i] = x[0]
    return est, upd


def m2_kalman(sigma_a_deg: float = 3000.0, sigma0: float = 1.5,
              gate_px: float = 15.0, rebuild: bool = False):
    name = "m2_kalman"
    c = khz2d.load_method(name)
    if c is not None and not rebuild:
        return c
    lm = _lm()
    rate = float(lm["line_rate"])
    fr = lm["frame"]
    okl = lm["ok"].astype(bool)
    con_ok = lm["con"] > khz2d.CONTRAST_FRAC * np.median(lm["con"])
    meas_x = np.where(okl & con_ok, lm["chain_x"][fr] + lm["rdx"], np.nan)
    meas_y = np.where(okl & con_ok, lm["chain_y"][fr] + lm["lam_v"], np.nan)
    pxdeg = px_per_deg(lm)
    sigma_a = sigma_a_deg * pxdeg        # px/s^2 process (accel) noise
    ex, ux = _kf_axis(lm["t"], meas_x, lm["qh"], khz2d.Q_FOV, sigma0, sigma_a,
                      gate_px=gate_px, reacq_steps=int(rate * 0.005))
    ey, uy = _kf_axis(lm["t"], meas_y, lm["qv"], 0.20, sigma0, sigma_a,
                      gate_px=gate_px, reacq_steps=int(rate * 0.005))

    # valid where each axis got an accepted update within the last 30 ms
    def recent(u, win):
        idx = np.arange(len(u)); last = np.where(u, idx, -10**9)
        last = np.maximum.accumulate(last)
        return (idx - last) <= win
    win = int(rate * 0.030)
    valid = recent(ux, win) & recent(uy, win) & np.isfinite(ex) & np.isfinite(ey)
    return khz2d.save_method(name, lm["t"], ex, ey, valid, rate)


# ---------------------------------------------------------------------------
# M3 — Viterbi decode of the horizontal profile sequence
# ---------------------------------------------------------------------------


def m3_viterbi(beta: float = 8.0, trans_gamma: float = 0.05,
               rebuild: bool = False):
    name = "m3_viterbi"
    c = khz2d.load_method(name)
    if c is not None and not rebuild:
        return c
    lm = _lm()
    rate = float(lm["line_rate"])
    pxdeg = px_per_deg(lm)
    vmax_px = 826.0 * pxdeg                      # main-sequence ceiling, px/s
    Wt = int(np.ceil(vmax_px / rate)) + 2        # per-line transition cap (px)
    nS = 2 * PADH + 1
    fr = lm["frame"]; col = lm["col"]
    okl = lm["ok"].astype(bool)
    con_ok = lm["con"] > khz2d.CONTRAST_FRAC * np.median(lm["con"])
    use = okl & con_ok & (lm["qh"] > 0.05)
    prof = lm["prof"]                            # (N, nS) float16
    N = prof.shape[0]
    inc_x = np.diff(lm["chain_x"], prepend=lm["chain_x"][0])

    # transition: from state j we may come from j+d, d in [-Wt, Wt]
    off = np.arange(-Wt, Wt + 1)
    pen = (trans_gamma * np.abs(off)).astype(np.float32)
    score = np.zeros(nS, np.float32)
    bp = np.empty((N, nS), np.int8)
    NEG = np.float32(-1e9)
    pad = np.full(Wt, NEG, np.float32)
    t0 = time.time()
    for i in range(N):
        if i > 0 and col[i] == 0:                # frame boundary: d -> d - inc_x
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
        if i % 200000 == 0 and i:
            print(f"  [m3] {i}/{N} ({time.time()-t0:.0f}s)")
    # backtrace
    d = np.empty(N, np.float32)
    j = int(np.argmax(score))
    for i in range(N - 1, -1, -1):
        d[i] = j - PADH
        j = j + int(bp[i, j])
        if i > 0 and col[i] == 0:                # undo boundary shift
            k = int(round(inc_x[fr[i]]))
            j = int(np.clip(j + k, 0, nS - 1))
        j = int(np.clip(j, 0, nS - 1))
    # subpixel refinement on the chosen ridge
    pr = prof.astype(np.float32)
    ji = (d + PADH).astype(int)
    iN = np.arange(N)
    inner = (ji > 0) & (ji < nS - 1) & use
    a = pr[iN, np.clip(ji - 1, 0, nS - 1)]; b = pr[iN, ji]
    cc = pr[iN, np.clip(ji + 1, 0, nS - 1)]
    den = a - 2 * b + cc
    subs = np.where(inner & (np.abs(den) > 1e-12), 0.5 * (a - cc) / np.where(den == 0, 1, den), 0.0)
    d = d + np.clip(subs, -1, 1)

    pos_x = lm["chain_x"][fr] + d
    m2 = m2_kalman()
    valid = use & (lm["qh"] > 0.15) & m2["valid"]
    return khz2d.save_method(name, lm["t"], pos_x, m2["y_px"], valid, rate,
                             extra=dict(d=d))


# ---------------------------------------------------------------------------
# M4 — the existing particle filter (filter.py) fused with the 20 Hz SLO
# ---------------------------------------------------------------------------


def m4_dpf(eff_rate: float = 1182.0, dur_s: float | None = None,
           likelihood: str = "physics", n_particles: int = 300,
           padw: int = 100, line_len: int = 200, seed: int = 0,
           rebuild: bool = False):
    """Run filter.py's ParticleFilter per effective line.

    Per-step atlas = a (2*padw+1)-column window of the chain-aligned PREVIOUS
    frame, transposed so atlas rows = slow-axis columns. The window is centred
    on the current scan column, so the perp state is the GAZE RESIDUAL
    d + padw (scan ramp removed); at frame boundaries the particle cloud is
    shifted by -chain increment (coordinate re-reference). Trusted along =
    vertical channel; coarse anchor = the chain (residual 0) — i.e. the 20 Hz
    2D SLO absolute fix. This is the G10-G14 machinery, unmodified, fused with
    the slow 2D SLO.
    """
    import filter as flt
    tag = f"m4_dpf_{int(eff_rate)}" + ("" if likelihood == "physics" else "_learned")
    c = khz2d.load_method(tag)
    if c is not None and not rebuild:
        return c
    lm = _lm()
    rate = float(lm["line_rate"])
    block = max(1, int(round(rate / eff_rate)))
    eff = rate / block
    head = None
    if likelihood == "learned":
        import train
        head = train.load_head()
    ch = khz2d.chain()
    con_med = np.median(lm["con"])

    T_, X_, Y_, V_, NCC_ = [], [], [], [], []
    rng = np.random.default_rng(seed)
    pf = None
    prev_db = None
    H = None
    t_start = time.time()
    nfr = len(ch["t"])
    for f, raw in khz2d._read_frames():
        if f >= nfr:
            break
        g = data._deband(raw)
        if H is None:
            H = g.shape[0]
        gc = g[khz2d.CROPV:H - khz2d.CROPV]
        if prev_db is None:
            prev_db = gc; continue
        if dur_s is not None and f / float(ch["fps"]) > dur_s:
            break
        ok_f = bool(ch["ok"][f])
        sx = -float(ch["inc_x"][f]); sy = -float(ch["inc_y"][f])
        if ok_f:
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            prevw = cv2.warpAffine(prev_db, M, (prev_db.shape[1], prev_db.shape[0]),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        else:
            prevw = prev_db
        Hc, W = gc.shape
        col_step = Hc / float(line_len)
        atlasT = prevw.T                                  # (W cols, Hc) rows = slow axis
        # frame-boundary coordinate re-reference for the particle cloud
        if pf is not None and ok_f:
            pf.state.pos_perp = pf.state.pos_perp - float(ch["inc_x"][f])
            pf.state.pos_along = pf.state.pos_along - float(ch["inc_y"][f])
        i0 = (f - 1) * 808                                # line-cache offset of this frame
        nb = W // block
        for b in range(nb):
            c0 = b * block; c1 = c0 + block
            cmid = (c0 + c1 - 1) / 2.0
            sl = slice(i0 + c0, i0 + c1)
            qv = lm["qv"][sl]; lam = lm["lam_v"][sl]
            con = lm["con"][sl].mean()
            tline = float(lm["t"][sl][len(lam) // 2])
            good = ok_f and con > khz2d.CONTRAST_FRAC * con_med and np.median(qv) > 0.1
            if not good or pf is None:
                if pf is None and good:
                    # initial lock from the raw measurements
                    d0 = float(np.median(lm["rdx"][sl]))
                    a0 = float(np.median(lam))
                    st = flt.init_filter(n_particles, padw + d0, a0, 15.0, 4.0, rng=rng)
                    win = atlasT[max(0, int(cmid) - padw): int(cmid) + padw + 1]
                    pf = flt.ParticleFilter(st, win, line_len, col_step=col_step,
                                            reseed_perp_sigma=30.0,
                                            likelihood=likelihood, learned_head=head)
                T_.append(tline); X_.append(np.nan); Y_.append(np.nan)
                V_.append(False); NCC_.append(0.0)
                continue
            line = gc[:, c0:c1].mean(1)
            obs = np.interp(np.arange(line_len) * col_step,
                            np.arange(Hc), line)
            lo = int(cmid) - padw
            win = atlasT[max(0, lo): lo + 2 * padw + 1]
            if win.shape[0] < 2 * padw + 1:               # pad at frame edges
                padrows = np.zeros((2 * padw + 1 - win.shape[0], Hc), np.float32)
                win = np.vstack([padrows, win] if lo < 0 else [win, padrows])
            pf.atlas = win.astype(np.float64)
            along_meas = float(np.median(lam))
            post = pf.step(obs, along_meas, block / rate, rng, coarse_anchor=float(padw))
            d_est = post.est_perp - padw
            T_.append(tline)
            X_.append(float(ch["x"][f]) + d_est)
            Y_.append(float(ch["y"][f]) + post.est_along)
            V_.append(post.max_ncc > 0.12)
            NCC_.append(post.max_ncc)
        prev_db = gc
        if f % 100 == 0:
            el = time.time() - t_start
            print(f"  [m4 {int(eff)}Hz {likelihood}] frame {f}/{nfr} ({el:.0f}s)")
    T_ = np.array(T_); X_ = np.array(X_); Y_ = np.array(Y_)
    V_ = np.array(V_) & np.isfinite(X_) & np.isfinite(Y_)
    return khz2d.save_method(tag, T_, X_, Y_, V_, eff,
                             extra=dict(max_ncc=np.array(NCC_)))


# ---------------------------------------------------------------------------
# M5 — batch MAP smoother (torch) at line rate
# ---------------------------------------------------------------------------


def m5_map(iters: int = 300, lr: float = 0.3, w_data: float = 1.0,
           w_dyn: float = 0.5, w_anchor: float = 2.0, beta: float = 4.0,
           rebuild: bool = False):
    """Offline MAP: minimise -beta*<profile NCC at d> + w_dyn*l_dyn + w_anchor*
    l_anchor over the full horizontal trajectory (torch Adam, init = Viterbi).

    The profile data term IS the L_recon NCC (decoder render vs observed line)
    precomputed on the offset grid; l_dyn / l_anchor are losses.py verbatim
    (positions converted px -> atlas rows via the measured scale so VMAX and
    the alias normalisation apply unchanged).
    """
    import torch
    import losses
    name = "m5_map"
    c = khz2d.load_method(name)
    if c is not None and not rebuild:
        return c
    lm = _lm()
    m3 = m3_viterbi()
    rate = float(lm["line_rate"])
    pxdeg = px_per_deg(lm)
    px2row = float(calib.ROWS_PER_DEG) / pxdeg
    fr = lm["frame"]
    chx = lm["chain_x"][fr].astype(np.float64)
    okl = lm["ok"].astype(bool)
    con_ok = lm["con"] > khz2d.CONTRAST_FRAC * np.median(lm["con"])
    use = okl & con_ok & (lm["qh"] > 0.10)
    N = len(chx)
    nS = 2 * PADH + 1

    prof = torch.from_numpy(lm["prof"].astype(np.float32))      # (N, nS)
    w = torch.from_numpy(use.astype(np.float32))
    d = torch.tensor(np.asarray(m3["d"], np.float64), dtype=torch.float64,
                     requires_grad=True)
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
        l_an = losses.l_anchor(pooled(pos_rows), pooled(tt),
                               anchor_rows, anchor_t)
        loss = w_data * l_data + w_dyn * l_dy + w_anchor * l_an
        loss.backward()
        opt.step()
        if it % 50 == 0:
            print(f"  [m5] it {it}: data {float(l_data):+.4f} dyn {float(l_dy):.4f} "
                  f"anchor {float(l_an):.4f} ({time.time()-t0:.0f}s)")
    d_np = d.detach().numpy()
    pos_x = chx + d_np
    m2 = m2_kalman()
    valid = use & m2["valid"]
    return khz2d.save_method(name, lm["t"], pos_x, m2["y_px"], valid, rate,
                             extra=dict(d=d_np))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="m2")
    ap.add_argument("--S", type=int, default=20)
    ap.add_argument("--rate", type=float, default=1182.0)
    ap.add_argument("--dur", type=float, default=None)
    ap.add_argument("--likelihood", default="physics")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.method == "m0":
        r = m0_chain()
    elif a.method == "m1":
        r = m1_strips(a.S, rebuild=a.rebuild)
    elif a.method == "m2":
        r = m2_kalman(rebuild=a.rebuild)
    elif a.method == "m3":
        r = m3_viterbi(rebuild=a.rebuild)
    elif a.method == "m4":
        r = m4_dpf(a.rate, dur_s=a.dur, likelihood=a.likelihood, rebuild=a.rebuild)
    elif a.method == "m5":
        r = m5_map(rebuild=a.rebuild)
    else:
        raise SystemExit(f"unknown method {a.method}")
    ev = khz2d.evaluate(r["t"], r["x_px"], r["y_px"], r["valid"].astype(bool),
                        float(r["rate"]), a.method, smooth_ms=2)
    print(khz2d.summarize(ev))
