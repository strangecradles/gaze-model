"""Strip-width ladder for people_data_fov SLO captures.

This is a TSLO-style baseline for the SDSLO people data: split each SLO frame
into non-overlapping column strips, register every strip to the previous frame
with 2D NCC, and sweep strip width to expose the rate-vs-precision tradeoff.

Default target is Ashton3. Outputs are written under the subject cache as
``strip_ladder_s<S>[_dN].npz`` plus a CSV/Markdown report in ``results/``.
Canonical PF caches are not touched.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Callable, Iterable

import cv2
import numpy as np

import data
import dynamics
import filter as flt
import khz2d
import people_fov_pf as pf

DEFAULT_WIDTHS = (64, 32, 16, 15, 8, 4, 2, 1)
DEFAULT_OUT_PREFIX = os.path.join("results", "people_strip_ladder_Ashton3")
DEFAULT_METHODS = ("raw",)
PF_N_PARTICLES = 300
PF_INIT_SPREAD_PX = 15.0
PF_BETA = flt.BETA
PF_ESS_FRAC = flt.ESS_FRAC
PF_ROUGHEN_PERP = flt.ROUGHEN_PERP
PF_ROUGHEN_ALONG = flt.ROUGHEN_ALONG
PF_HYPOTHESIS_CLUSTER_ROWS = flt.HYPOTHESIS_CLUSTER_ROWS


def _tag_float(x: float) -> str:
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def parse_strip_widths(text: str) -> list[int]:
    widths = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        width = int(part)
        if width <= 0:
            raise ValueError("strip widths must be positive")
        if width not in widths:
            widths.append(width)
    if not widths:
        raise ValueError("at least one strip width is required")
    return widths


def strip_rate_hz(frame_cols: int, fps: float, strip_width: int) -> float:
    if strip_width <= 0:
        raise ValueError("strip_width must be positive")
    return float((int(frame_cols) // int(strip_width)) * float(fps))


def parse_methods(text: str) -> list[str]:
    methods = []
    for part in text.replace(";", ",").split(","):
        method = part.strip().lower()
        if not method:
            continue
        if method not in {"raw", "resolver", "pf"}:
            raise ValueError(f"unknown strip method {method!r}")
        if method not in methods:
            methods.append(method)
    if not methods:
        raise ValueError("at least one method is required")
    return methods


def strip_cache_path(
    sub: pf.Subject,
    strip_width: int,
    dur_s: float | None = None,
    method: str = "raw",
) -> str:
    prefix = "strip_ladder" if method == "raw" else f"strip_ladder_{method}"
    tag = f"{prefix}_s{int(strip_width)}"
    if dur_s is not None:
        tag += f"_d{_tag_float(dur_s)}"
    return os.path.join(sub.cache_dir, f"{tag}.npz")


def resolver_variant_name(
    *,
    top_k: int = 5,
    lag_ms: float = flt.HYPOTHESIS_LAG_MS,
    transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS,
    obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT,
    velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
    acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
    slew_gate: bool = False,
    slew_max_deg_s: float = flt.SLEW_GATE_MAX_DEG_S,
    blend_immediate: bool = False,
    blend_delta_rows: float = flt.HYPOTHESIS_BLEND_DELTA_ROWS,
    blend_alpha: float = flt.HYPOTHESIS_BLEND_ALPHA,
) -> str:
    tag = "resolver"
    if int(top_k) != 5:
        tag += f"_k{int(top_k)}"
    if float(lag_ms) != flt.HYPOTHESIS_LAG_MS:
        tag += f"_lag{_tag_float(lag_ms)}"
    if float(transition_sigma_rows) != flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS:
        tag += f"_ts{_tag_float(transition_sigma_rows)}"
    if float(obs_weight) != flt.HYPOTHESIS_OBS_WEIGHT:
        tag += f"_ow{_tag_float(obs_weight)}"
    if float(velocity_cost) != flt.HYPOTHESIS_VEL_COST:
        tag += f"_vc{_tag_float(velocity_cost)}"
    if float(acceleration_cost) != flt.HYPOTHESIS_ACCEL_COST:
        tag += f"_ac{_tag_float(acceleration_cost)}"
    if bool(slew_gate):
        tag += f"_sg{_tag_float(slew_max_deg_s)}"
    if bool(blend_immediate):
        tag += f"_bi_d{_tag_float(blend_delta_rows)}_a{_tag_float(blend_alpha)}"
    return tag


def pf_variant_name(
    *,
    n_particles: int = PF_N_PARTICLES,
    beta: float = PF_BETA,
    ess_frac: float = PF_ESS_FRAC,
    roughen_perp: float = PF_ROUGHEN_PERP,
    roughen_along: float = PF_ROUGHEN_ALONG,
    init_spread_px: float = PF_INIT_SPREAD_PX,
    top_k: int = 5,
    lag_ms: float = flt.HYPOTHESIS_LAG_MS,
    transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS,
    obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT,
    velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
    acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
    slew_gate: bool = False,
    slew_max_deg_s: float = flt.SLEW_GATE_MAX_DEG_S,
    blend_immediate: bool = False,
    blend_delta_rows: float = flt.HYPOTHESIS_BLEND_DELTA_ROWS,
    blend_alpha: float = flt.HYPOTHESIS_BLEND_ALPHA,
) -> str:
    tag = "pf"
    if int(n_particles) != PF_N_PARTICLES:
        tag += f"_n{int(n_particles)}"
    if float(beta) != PF_BETA:
        tag += f"_b{_tag_float(beta)}"
    if float(ess_frac) != PF_ESS_FRAC:
        tag += f"_ess{_tag_float(ess_frac)}"
    if float(roughen_perp) != PF_ROUGHEN_PERP:
        tag += f"_rp{_tag_float(roughen_perp)}"
    if float(roughen_along) != PF_ROUGHEN_ALONG:
        tag += f"_ra{_tag_float(roughen_along)}"
    if float(init_spread_px) != PF_INIT_SPREAD_PX:
        tag += f"_init{_tag_float(init_spread_px)}"
    res = resolver_variant_name(
        top_k=top_k, lag_ms=lag_ms,
        transition_sigma_rows=transition_sigma_rows,
        obs_weight=obs_weight, velocity_cost=velocity_cost,
        acceleration_cost=acceleration_cost, slew_gate=slew_gate,
        slew_max_deg_s=slew_max_deg_s, blend_immediate=blend_immediate,
        blend_delta_rows=blend_delta_rows, blend_alpha=blend_alpha)
    if res != "resolver":
        tag += "_" + res.replace("resolver_", "")
    return tag


def _nz(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def strip_valid_mask(q: np.ndarray, con: np.ndarray, quality_thr: float,
                     contrast_frac: float) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    con = np.asarray(con, dtype=np.float64)
    med = float(np.nanmedian(con)) if con.size else float("nan")
    if not np.isfinite(med) or med <= 0.0:
        return np.zeros_like(q, dtype=bool)
    return (
        np.isfinite(q)
        & np.isfinite(con)
        & (q > float(quality_thr))
        & (con > float(contrast_frac) * med)
    )


def _load_npz(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def sample_response_bilinear(
    response: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    *,
    fill: float = -1.0,
) -> np.ndarray:
    """Bilinear sample a matchTemplate response at floating index coordinates."""
    r = np.asarray(response, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    out = np.full(np.broadcast_shapes(y.shape, x.shape), float(fill), dtype=np.float64)
    yy = np.broadcast_to(y, out.shape)
    xx = np.broadcast_to(x, out.shape)
    valid = (
        np.isfinite(yy) & np.isfinite(xx)
        & (yy >= 0.0) & (xx >= 0.0)
        & (yy <= r.shape[0] - 1) & (xx <= r.shape[1] - 1)
    )
    if not valid.any():
        return out
    yv = yy[valid]
    xv = xx[valid]
    y0 = np.floor(yv).astype(np.int64)
    x0 = np.floor(xv).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, r.shape[0] - 1)
    x1 = np.clip(x0 + 1, 0, r.shape[1] - 1)
    wy = yv - y0
    wx = xv - x0
    out[valid] = (
        (1.0 - wy) * (1.0 - wx) * r[y0, x0]
        + wy * (1.0 - wx) * r[y1, x0]
        + (1.0 - wy) * wx * r[y0, x1]
        + wy * wx * r[y1, x1]
    )
    return out


def topk_response_peaks(
    response: np.ndarray,
    top_k: int,
    *,
    pad: int,
    suppress_radius: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return top-K local NCC peaks as (dy, dx, score)."""
    r = np.asarray(response, dtype=np.float64)
    work = r.copy()
    dy, dx, q = [], [], []
    for _ in range(max(1, int(top_k))):
        if not np.isfinite(work).any():
            break
        flat = int(np.nanargmax(work))
        y0, x0 = np.unravel_index(flat, work.shape)
        score = float(work[y0, x0])
        if not np.isfinite(score):
            break
        yy = khz2d._parab(r[:, x0], int(y0)) if 0 <= x0 < r.shape[1] else float(y0)
        xx = khz2d._parab(r[y0, :], int(x0)) if 0 <= y0 < r.shape[0] else float(x0)
        dy.append(float(yy) - float(pad))
        dx.append(float(xx) - float(pad))
        q.append(score)
        y1 = max(0, y0 - suppress_radius)
        y2 = min(work.shape[0], y0 + suppress_radius + 1)
        x1 = max(0, x0 - suppress_radius)
        x2 = min(work.shape[1], x0 + suppress_radius + 1)
        work[y1:y2, x1:x2] = -np.inf
    if not q:
        return (
            np.asarray([0.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            np.asarray([-np.inf], dtype=np.float64),
        )
    return (
        np.asarray(dy, dtype=np.float64),
        np.asarray(dx, dtype=np.float64),
        np.asarray(q, dtype=np.float64),
    )


def _clip_run(run: dict, dur_s: float | None) -> dict:
    if dur_s is None or "t" not in run or len(run["t"]) == 0:
        return run
    t = np.asarray(run["t"], dtype=np.float64)
    keep = t <= float(t[0]) + float(dur_s)
    out = {}
    for key, value in run.items():
        arr = np.asarray(value)
        out[key] = arr[keep] if arr.shape[:1] == t.shape[:1] else value
    return out


def raw_step_metrics(run: dict) -> dict[str, float]:
    valid = np.asarray(run["valid"], dtype=bool)
    x = np.asarray(run["x_px"], dtype=np.float64)
    rate = float(run["rate"])
    step_ok = valid[1:] & valid[:-1] & np.isfinite(x[1:]) & np.isfinite(x[:-1])
    step = np.abs(np.diff(x)[step_ok])
    if step.size == 0:
        return dict(
            raw_jump_ge3_frac=float("nan"),
            raw_step_p99_px=float("nan"),
            raw_step_p999_px=float("nan"),
            raw_speed_p999_px_s=float("nan"),
            raw_speed_p999_arcmin_s=float("nan"),
        )
    speed = step * rate
    return dict(
        raw_jump_ge3_frac=float(np.mean(step >= 3.0)),
        raw_step_p99_px=float(np.percentile(step, 99.0)),
        raw_step_p999_px=float(np.percentile(step, 99.9)),
        raw_speed_p999_px_s=float(np.percentile(speed, 99.9)),
        raw_speed_p999_arcmin_s=float(np.percentile(speed * pf.ARC_PER_PX, 99.9)),
    )


def collect_metrics(run: dict, refs: dict) -> dict[str, float]:
    t = np.asarray(run["t"], dtype=np.float64)
    x = np.asarray(run["x_px"], dtype=np.float64)
    y = np.asarray(run["y_px"], dtype=np.float64)
    valid = np.asarray(run["valid"], dtype=bool)
    rate = float(run["rate"])
    ev = pf.evaluate(t, x, y, valid, rate, refs)
    out = dict(
        rate=rate,
        n_samples=float(len(t)),
        valid_frac=float(ev["valid_frac"]),
        r_dot_x=float(ev["r_dot_x"]),
        r_dot_y=float(ev["r_dot_y"]),
        r_trk_x=float(ev["r_trk_x"]),
        r_trk_y=float(ev["r_trk_y"]),
        rms_x=float(ev["rms_x"]),
        rms_y=float(ev["rms_y"]),
        prec_x=float(ev["prec_x"]),
        prec_y=float(ev["prec_y"]),
        j30=float(pf.frame_jitter_30fps(ev["cal_x"], rate, valid)),
    )
    out.update(raw_step_metrics(run))
    return out


def run_strip_tracker(
    sub: pf.Subject,
    strip_width: int,
    *,
    pad: int = 80,
    dur_s: float | None = None,
    rebuild: bool = False,
    quality_thr: float = 0.35,
    contrast_frac: float = pf.CONTRAST_FRAC,
    cache_path: str | None = None,
) -> dict:
    """Run one non-overlapping strip tracker for a people_data_fov subject."""
    out_path = cache_path or strip_cache_path(sub, strip_width, dur_s)
    if os.path.exists(out_path) and not rebuild:
        return _load_npz(out_path)

    cap = cv2.VideoCapture(sub.slo_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    DY, DX, Q, CON, T = [], [], [], [], []
    cum_y = 0.0
    cum_x = 0.0
    prev = None
    frame_idx = 0
    nstrip = 0
    frame_cols = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if dur_s is not None and fps > 0 and frame_idx / fps > float(dur_s):
            break
        raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cur = _nz(data._deband(raw))
        if prev is not None:
            _height, width = cur.shape
            if strip_width > width:
                raise ValueError(f"strip width {strip_width} exceeds frame width {width}")
            frame_cols = width
            nstrip = width // strip_width
            refp = cv2.copyMakeBorder(prev, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=0)
            fy = np.empty(nstrip, dtype=np.float64)
            fx = np.empty(nstrip, dtype=np.float64)
            fq = np.empty(nstrip, dtype=np.float64)
            fc = np.empty(nstrip, dtype=np.float64)
            for s in range(nstrip):
                col0 = s * strip_width
                strip = cur[:, col0:col0 + strip_width]
                region = refp[:, col0:col0 + strip_width + 2 * pad]
                r = cv2.matchTemplate(region, strip, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                fy[s] = loc[1] - pad
                fx[s] = loc[0] - pad
                fq[s] = mx
                fc[s] = raw[:, col0:col0 + strip_width].std()
            med_y = float(np.median(fy))
            med_x = float(np.median(fx))
            cum_y += med_y
            cum_x += med_x
            DY.extend(cum_y + (fy - med_y))
            DX.extend(cum_x + (fx - med_x))
            Q.extend(fq)
            CON.extend(fc)
            centers = np.arange(nstrip, dtype=np.float64) * strip_width + strip_width / 2.0
            T.extend((frame_idx + centers / width) / fps)
        prev = cur
        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{sub.name} strips S={strip_width}] frame {frame_idx} ({elapsed:.0f}s)",
                flush=True,
            )
    cap.release()

    t = np.asarray(T, dtype=np.float64)
    x = np.asarray(DX, dtype=np.float64)
    y = np.asarray(DY, dtype=np.float64)
    q = np.asarray(Q, dtype=np.float64)
    con = np.asarray(CON, dtype=np.float64)
    valid = strip_valid_mask(q, con, quality_thr, contrast_frac)
    rate = strip_rate_hz(frame_cols, fps, strip_width) if frame_cols else float("nan")
    out = dict(
        t=t, x_px=x, y_px=y, valid=valid, rate=np.float64(rate),
        q=q, con=con, strip_width=np.int64(strip_width), pad=np.int64(pad),
        nstrip=np.int64(nstrip), frame_cols=np.int64(frame_cols),
        fps=np.float64(fps), n_frames=np.int64(frame_idx),
        quality_thr=np.float64(quality_thr),
        contrast_frac=np.float64(contrast_frac),
        subject=np.asarray(sub.name),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    return out


def _strip_step_posterior(
    hyp_x: np.ndarray,
    hyp_y: np.ndarray,
    hyp_q: np.ndarray,
    *,
    prev_x: float,
    prev_y: float,
    dt: float,
) -> flt.StepPosterior:
    hyp_x = np.asarray(hyp_x, dtype=np.float64)
    hyp_y = np.asarray(hyp_y, dtype=np.float64)
    hyp_q = np.asarray(hyp_q, dtype=np.float64)
    if hyp_x.size == 0:
        hyp_x = np.asarray([prev_x if np.isfinite(prev_x) else 0.0])
        hyp_y = np.asarray([prev_y if np.isfinite(prev_y) else 0.0])
        hyp_q = np.asarray([0.0])
    logp = np.where(np.isfinite(hyp_q), hyp_q, -1e6)
    j = int(np.argmax(logp))
    if np.isfinite(prev_x) and dt > 0:
        vx = (hyp_x - prev_x) / dt
        vy = (hyp_y - prev_y) / dt
    else:
        vx = np.zeros_like(hyp_x)
        vy = np.zeros_like(hyp_y)
    ww = np.exp(logp - float(np.max(logp)))
    ww = ww / max(float(ww.sum()), 1e-12)
    return flt.StepPosterior(
        est_perp=float(hyp_x[j]),
        est_along=float(hyp_y[j]),
        est_v_perp=float(vx[j]),
        est_v_along=float(vy[j]),
        ess=float(1.0 / np.sum(ww ** 2)),
        mode_posterior=(1.0, 0.0),
        resampled=False,
        reseeded=False,
        max_ncc=float(hyp_q[j]) if np.isfinite(hyp_q[j]) else 0.0,
        along_sigma_eff=0.0,
        hyp_perp=hyp_x,
        hyp_along=hyp_y,
        hyp_v_perp=vx,
        hyp_v_along=vy,
        hyp_logp=logp,
        pos_perp=hyp_x.copy(),
        pos_along=hyp_y.copy(),
        weight=ww,
    )


def run_strip_resolver(
    sub: pf.Subject,
    strip_width: int,
    *,
    pad: int = 80,
    dur_s: float | None = None,
    rebuild: bool = False,
    quality_thr: float = 0.35,
    contrast_frac: float = pf.CONTRAST_FRAC,
    cache_path: str | None = None,
    top_k: int = 5,
    lag_ms: float = flt.HYPOTHESIS_LAG_MS,
    transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS,
    obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT,
    velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
    velocity_sigma_deg_s: float = flt.HYPOTHESIS_VEL_SIGMA_DEG_S,
    acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
    acceleration_sigma_deg_s2: float = flt.HYPOTHESIS_ACCEL_SIGMA_DEG_S2,
    slew_gate: bool = False,
    slew_max_deg_s: float = flt.SLEW_GATE_MAX_DEG_S,
    blend_immediate: bool = False,
    blend_delta_rows: float = flt.HYPOTHESIS_BLEND_DELTA_ROWS,
    blend_alpha: float = flt.HYPOTHESIS_BLEND_ALPHA,
) -> dict:
    """Run the strip ladder with top-K NCC peaks resolved by fixed-lag dynamics."""
    out_path = cache_path or strip_cache_path(sub, strip_width, dur_s, method="resolver")
    if os.path.exists(out_path) and not rebuild:
        return _load_npz(out_path)

    cap = cv2.VideoCapture(sub.slo_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    T, XRAW, YRAW, Q, CON = [], [], [], [], []
    XRES, YRES = [], []
    RSLV, HCOUNT, HIDX, HRANK, HGAP, HMARGIN = [], [], [], [], [], []
    cum_y = 0.0
    cum_x = 0.0
    prev = None
    prev_est_x = float("nan")
    prev_est_y = float("nan")
    prev_t = float("nan")
    resolver: flt.FixedLagHypothesisResolver | None = None
    frame_idx = 0
    nstrip = 0
    frame_cols = 0
    t0 = time.time()

    def apply_resolved(est: flt.FixedLagEstimate) -> None:
        XRES[est.index] = est.est_perp
        YRES[est.index] = est.est_along
        RSLV[est.index] = True
        HIDX[est.index] = est.hyp_index
        HRANK[est.index] = est.hyp_rank
        HGAP[est.index] = est.hyp_logp_gap
        HMARGIN[est.index] = est.hyp_logp_margin

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if dur_s is not None and fps > 0 and frame_idx / fps > float(dur_s):
            break
        raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cur = _nz(data._deband(raw))
        if prev is not None:
            _height, width = cur.shape
            if strip_width > width:
                raise ValueError(f"strip width {strip_width} exceeds frame width {width}")
            frame_cols = width
            nstrip = width // strip_width
            rate = strip_rate_hz(width, fps, strip_width)
            if resolver is None:
                resolver = flt.FixedLagHypothesisResolver(
                    rate,
                    lag_ms=lag_ms,
                    transition_sigma_rows=transition_sigma_rows,
                    obs_weight=obs_weight,
                    velocity_cost=velocity_cost,
                    velocity_sigma_deg_s=velocity_sigma_deg_s,
                    acceleration_cost=acceleration_cost,
                    acceleration_sigma_deg_s2=acceleration_sigma_deg_s2,
                    slew_gate=slew_gate,
                    slew_max_deg_s=slew_max_deg_s,
                    blend_immediate=blend_immediate,
                    blend_delta_rows=blend_delta_rows,
                    blend_alpha=blend_alpha,
                )
            refp = cv2.copyMakeBorder(prev, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=0)
            hyp_dy: list[np.ndarray] = []
            hyp_dx: list[np.ndarray] = []
            hyp_q: list[np.ndarray] = []
            best_y = np.empty(nstrip, dtype=np.float64)
            best_x = np.empty(nstrip, dtype=np.float64)
            best_q = np.empty(nstrip, dtype=np.float64)
            con = np.empty(nstrip, dtype=np.float64)
            for s in range(nstrip):
                col0 = s * strip_width
                strip = cur[:, col0:col0 + strip_width]
                region = refp[:, col0:col0 + strip_width + 2 * pad]
                r = cv2.matchTemplate(region, strip, cv2.TM_CCOEFF_NORMED)
                dy, dx, qq = topk_response_peaks(r, top_k, pad=pad)
                hyp_dy.append(dy)
                hyp_dx.append(dx)
                hyp_q.append(qq)
                best_y[s] = dy[0]
                best_x[s] = dx[0]
                best_q[s] = qq[0]
                con[s] = raw[:, col0:col0 + strip_width].std()
            med_y = float(np.median(best_y))
            med_x = float(np.median(best_x))
            cum_y += med_y
            cum_x += med_x
            centers = np.arange(nstrip, dtype=np.float64) * strip_width + strip_width / 2.0
            for s in range(nstrip):
                t = float((frame_idx + centers[s] / width) / fps)
                abs_x = cum_x + (hyp_dx[s] - med_x)
                abs_y = cum_y + (hyp_dy[s] - med_y)
                raw_x = float(abs_x[0])
                raw_y = float(abs_y[0])
                dt = (1.0 / rate) if not np.isfinite(prev_t) else max(t - prev_t, 1e-9)
                post = _strip_step_posterior(
                    abs_x,
                    abs_y,
                    hyp_q[s],
                    prev_x=prev_est_x,
                    prev_y=prev_est_y,
                    dt=dt,
                )
                out_i = len(T)
                T.append(t)
                XRAW.append(raw_x)
                YRAW.append(raw_y)
                XRES.append(raw_x)
                YRES.append(raw_y)
                RSLV.append(False)
                HCOUNT.append(len(abs_x))
                HIDX.append(-1)
                HRANK.append(-1)
                HGAP.append(np.nan)
                HMARGIN.append(np.nan)
                Q.append(float(best_q[s]))
                CON.append(float(con[s]))
                est = resolver.push(post, dt_s=dt)
                if est is not None:
                    apply_resolved(est)
                prev_est_x = raw_x
                prev_est_y = raw_y
                prev_t = t
        prev = cur
        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{sub.name} strip resolver S={strip_width}] frame {frame_idx} ({elapsed:.0f}s)",
                flush=True,
            )
    cap.release()
    if resolver is not None:
        for est in resolver.flush():
            apply_resolved(est)

    t = np.asarray(T, dtype=np.float64)
    x = np.asarray(XRES, dtype=np.float64)
    y = np.asarray(YRES, dtype=np.float64)
    q = np.asarray(Q, dtype=np.float64)
    con = np.asarray(CON, dtype=np.float64)
    valid = strip_valid_mask(q, con, quality_thr, contrast_frac)
    rate = strip_rate_hz(frame_cols, fps, strip_width) if frame_cols else float("nan")
    out = dict(
        t=t, x_px=x, y_px=y, valid=valid, rate=np.float64(rate),
        x_px_immediate=np.asarray(XRAW, dtype=np.float64),
        y_px_immediate=np.asarray(YRAW, dtype=np.float64),
        q=q, con=con, strip_width=np.int64(strip_width), pad=np.int64(pad),
        nstrip=np.int64(nstrip), frame_cols=np.int64(frame_cols),
        fps=np.float64(fps), n_frames=np.int64(frame_idx),
        quality_thr=np.float64(quality_thr),
        contrast_frac=np.float64(contrast_frac),
        top_k=np.int64(top_k), fixed_lag_ms=np.float64(lag_ms),
        fixed_lag_lines=np.int64(resolver.lag if resolver is not None else 0),
        fixed_lag_resolved=np.asarray(RSLV, dtype=bool),
        hyp_count=np.asarray(HCOUNT, dtype=np.int16),
        fixed_lag_hyp_index=np.asarray(HIDX, dtype=np.int16),
        fixed_lag_hyp_rank=np.asarray(HRANK, dtype=np.int16),
        fixed_lag_hyp_logp_gap=np.asarray(HGAP, dtype=np.float64),
        fixed_lag_hyp_logp_margin=np.asarray(HMARGIN, dtype=np.float64),
        hypothesis_transition_sigma_rows=np.float64(transition_sigma_rows),
        hypothesis_obs_weight=np.float64(obs_weight),
        hypothesis_velocity_cost=np.float64(velocity_cost),
        hypothesis_acceleration_cost=np.float64(acceleration_cost),
        slew_gate=np.bool_(slew_gate),
        slew_max_deg_s=np.float64(slew_max_deg_s if slew_gate else 0.0),
        hypothesis_blend_immediate=np.bool_(blend_immediate),
        hypothesis_blend_delta_rows=np.float64(blend_delta_rows if blend_immediate else 0.0),
        hypothesis_blend_alpha=np.float64(blend_alpha if blend_immediate else 0.0),
        subject=np.asarray(sub.name),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    return out


def _roughen_particle_state(
    st: dynamics.ParticleState,
    idx: np.ndarray,
    rng: np.random.Generator,
    roughen_perp: float,
    roughen_along: float,
) -> dynamics.ParticleState:
    for field in flt._STATE_FIELDS:
        setattr(st, field, getattr(st, field)[idx])
    n = st.n
    st.pos_perp = st.pos_perp + rng.normal(0.0, float(roughen_perp), n)
    st.pos_along = st.pos_along + rng.normal(0.0, float(roughen_along), n)
    st.weight = np.full(n, 1.0 / n, dtype=np.float64)
    return st


def _strip_pf_posterior(
    st: dynamics.ParticleState,
    weights: np.ndarray,
    ncc: np.ndarray,
    *,
    ess: float,
    resampled: bool,
    top_k: int,
    cluster_rows: float,
) -> tuple[flt.StepPosterior, list[np.ndarray]]:
    masks, hyp = flt._hypothesis_clusters(st, weights, top_k, cluster_rows)
    imap = int(np.argmax(weights))
    near = np.abs(st.pos_perp - st.pos_perp[imap]) < 0.5 * data.UNITS.alias_spacing_rows
    wn = weights * near
    if wn.sum() <= 0.0:
        wn = weights
    wn = wn / max(float(wn.sum()), 1e-12)
    est_perp = float(np.sum(wn * st.pos_perp))
    est_along = float(np.sum(wn * st.pos_along))
    est_v_perp = float(np.sum(wn * st.vel_perp))
    est_v_along = float(np.sum(wn * st.vel_along))
    mode_post = dynamics.mode_posterior(st)
    post = flt.StepPosterior(
        est_perp=est_perp,
        est_along=est_along,
        est_v_perp=est_v_perp,
        est_v_along=est_v_along,
        ess=float(ess),
        mode_posterior=mode_post,
        resampled=bool(resampled),
        reseeded=False,
        max_ncc=float(np.nanmax(ncc)) if np.isfinite(ncc).any() else 0.0,
        along_sigma_eff=0.0,
        hyp_perp=hyp["perp"],
        hyp_along=hyp["along"],
        hyp_v_perp=hyp["v_perp"],
        hyp_v_along=hyp["v_along"],
        hyp_logp=hyp["logp"],
        pos_perp=st.pos_perp.copy(),
        pos_along=st.pos_along.copy(),
        weight=weights.copy(),
    )
    return post, masks


def run_strip_pf(
    sub: pf.Subject,
    strip_width: int,
    *,
    pad: int = 80,
    dur_s: float | None = None,
    rebuild: bool = False,
    quality_thr: float = 0.35,
    contrast_frac: float = pf.CONTRAST_FRAC,
    cache_path: str | None = None,
    seed: int = 0,
    n_particles: int = PF_N_PARTICLES,
    init_spread_px: float = PF_INIT_SPREAD_PX,
    beta: float = PF_BETA,
    ess_frac: float = PF_ESS_FRAC,
    roughen_perp: float = PF_ROUGHEN_PERP,
    roughen_along: float = PF_ROUGHEN_ALONG,
    top_k: int = 5,
    cluster_rows: float = PF_HYPOTHESIS_CLUSTER_ROWS,
    lag_ms: float = flt.HYPOTHESIS_LAG_MS,
    transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS,
    obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT,
    velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
    velocity_sigma_deg_s: float = flt.HYPOTHESIS_VEL_SIGMA_DEG_S,
    acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
    acceleration_sigma_deg_s2: float = flt.HYPOTHESIS_ACCEL_SIGMA_DEG_S2,
    slew_gate: bool = False,
    slew_max_deg_s: float = flt.SLEW_GATE_MAX_DEG_S,
    blend_immediate: bool = False,
    blend_delta_rows: float = flt.HYPOTHESIS_BLEND_DELTA_ROWS,
    blend_alpha: float = flt.HYPOTHESIS_BLEND_ALPHA,
    sample_keep: Callable[[int, int, int], bool] | None = None,
    effective_rate: float | None = None,
) -> dict:
    """Run a true IMM particle filter over strip NCC response surfaces."""
    out_path = cache_path or strip_cache_path(sub, strip_width, dur_s, method="pf")
    if os.path.exists(out_path) and not rebuild:
        return _load_npz(out_path)

    cap = cv2.VideoCapture(sub.slo_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    T, XRAW, YRAW, XPF, YPF, Q, CON = [], [], [], [], [], [], []
    ESS, MAXNCC, PSACC, RESAMP, RSLV, HCOUNT = [], [], [], [], [], []
    HIDX, HRANK, HGAP, HMARGIN = [], [], [], []
    cum_y = 0.0
    cum_x = 0.0
    prev = None
    prev_t = float("nan")
    resolver: flt.FixedLagHypothesisResolver | None = None
    st: dynamics.ParticleState | None = None
    rng = np.random.default_rng(seed)
    frame_idx = 0
    nstrip = 0
    frame_cols = 0
    t0 = time.time()

    def apply_resolved(est: flt.FixedLagEstimate) -> None:
        XPF[est.index] = est.est_perp
        YPF[est.index] = est.est_along
        RSLV[est.index] = True
        HIDX[est.index] = est.hyp_index
        HRANK[est.index] = est.hyp_rank
        HGAP[est.index] = est.hyp_logp_gap
        HMARGIN[est.index] = est.hyp_logp_margin

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if dur_s is not None and fps > 0 and frame_idx / fps > float(dur_s):
            break
        raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cur = _nz(data._deband(raw))
        if prev is not None:
            _height, width = cur.shape
            if strip_width > width:
                raise ValueError(f"strip width {strip_width} exceeds frame width {width}")
            frame_cols = width
            nstrip = width // strip_width
            full_rate = strip_rate_hz(width, fps, strip_width)
            global0 = max(0, frame_idx - 1) * nstrip
            keep_idx = [
                s for s in range(nstrip)
                if sample_keep is None or bool(sample_keep(global0 + s, frame_idx, s))
            ]
            if not keep_idx:
                prev = cur
                frame_idx += 1
                continue
            rate = float(effective_rate) if effective_rate is not None else (
                full_rate * len(keep_idx) / max(nstrip, 1)
            )
            if resolver is None:
                resolver = flt.FixedLagHypothesisResolver(
                    rate,
                    lag_ms=lag_ms,
                    transition_sigma_rows=transition_sigma_rows,
                    obs_weight=obs_weight,
                    velocity_cost=velocity_cost,
                    velocity_sigma_deg_s=velocity_sigma_deg_s,
                    acceleration_cost=acceleration_cost,
                    acceleration_sigma_deg_s2=acceleration_sigma_deg_s2,
                    slew_gate=slew_gate,
                    slew_max_deg_s=slew_max_deg_s,
                    blend_immediate=blend_immediate,
                    blend_delta_rows=blend_delta_rows,
                    blend_alpha=blend_alpha,
                )
            refp = cv2.copyMakeBorder(prev, pad, pad, pad, pad,
                                      cv2.BORDER_CONSTANT, value=0)
            responses: list[np.ndarray] = []
            best_y = np.empty(nstrip, dtype=np.float64)
            best_x = np.empty(nstrip, dtype=np.float64)
            best_q = np.empty(nstrip, dtype=np.float64)
            con = np.empty(nstrip, dtype=np.float64)
            for s in range(nstrip):
                col0 = s * strip_width
                strip = cur[:, col0:col0 + strip_width]
                region = refp[:, col0:col0 + strip_width + 2 * pad]
                r = cv2.matchTemplate(region, strip, cv2.TM_CCOEFF_NORMED).astype(np.float64)
                _, mx, _, loc = cv2.minMaxLoc(r.astype(np.float32))
                yy = khz2d._parab(r[:, loc[0]], int(loc[1]))
                xx = khz2d._parab(r[loc[1], :], int(loc[0]))
                best_y[s] = yy - pad
                best_x[s] = xx - pad
                best_q[s] = mx
                con[s] = raw[:, col0:col0 + strip_width].std()
                responses.append(r)
            med_y = float(np.median(best_y[keep_idx]))
            med_x = float(np.median(best_x[keep_idx]))
            cum_y += med_y
            cum_x += med_x
            centers = np.arange(nstrip, dtype=np.float64) * strip_width + strip_width / 2.0
            for s in keep_idx:
                t = float((frame_idx + centers[s] / width) / fps)
                raw_x = float(cum_x + (best_x[s] - med_x))
                raw_y = float(cum_y + (best_y[s] - med_y))
                dt = (1.0 / rate) if not np.isfinite(prev_t) else max(t - prev_t, 1e-9)
                if st is None:
                    st = flt.init_filter(
                        int(n_particles),
                        raw_x,
                        raw_y,
                        float(init_spread_px),
                        float(init_spread_px),
                        rng=rng,
                    )
                else:
                    st = dynamics.predict(st, dt, rng)
                r = responses[s]
                # Convert absolute particle positions back into this strip's local
                # NCC-response coordinates: response[y=dy+pad, x=dx+pad].
                resp_x = pad + med_x + (st.pos_perp - cum_x)
                resp_y = pad + med_y + (st.pos_along - cum_y)
                ncc = sample_response_bilinear(r, resp_y, resp_x, fill=-1.0)
                max_ncc = float(np.nanmax(ncc)) if np.isfinite(ncc).any() else -1.0
                w_obs = np.exp(float(beta) * (ncc - max_ncc))
                weights = st.weight * w_obs
                sw = float(weights.sum())
                if not np.isfinite(sw) or sw <= 0.0:
                    st = flt.init_filter(
                        int(n_particles),
                        raw_x,
                        raw_y,
                        float(init_spread_px),
                        float(init_spread_px),
                        rng=rng,
                    )
                    weights = st.weight.copy()
                else:
                    weights = weights / sw
                    st.weight = weights
                ess = float(1.0 / np.sum(weights ** 2))
                post, masks = _strip_pf_posterior(
                    st, weights, ncc, ess=ess, resampled=False,
                    top_k=top_k, cluster_rows=cluster_rows)
                out_i = len(T)
                T.append(t)
                XRAW.append(raw_x)
                YRAW.append(raw_y)
                XPF.append(post.est_perp)
                YPF.append(post.est_along)
                Q.append(float(best_q[s]))
                CON.append(float(con[s]))
                ESS.append(ess)
                MAXNCC.append(max_ncc)
                PSACC.append(float(post.mode_posterior[1]))
                RESAMP.append(False)
                RSLV.append(False)
                HCOUNT.append(len(post.hyp_perp))
                HIDX.append(-1)
                HRANK.append(-1)
                HGAP.append(np.nan)
                HMARGIN.append(np.nan)
                est = resolver.push(post, dt_s=dt)
                if est is not None:
                    apply_resolved(est)
                if ess < float(ess_frac) * st.n:
                    idx = flt._hypothesis_resample(weights, masks, rng, 6)
                    st = _roughen_particle_state(
                        st, idx, rng, roughen_perp, roughen_along)
                    RESAMP[-1] = True
                prev_t = t
        prev = cur
        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{sub.name} strip PF S={strip_width}] frame {frame_idx} ({elapsed:.0f}s)",
                flush=True,
            )
    cap.release()
    if resolver is not None:
        for est in resolver.flush():
            apply_resolved(est)

    t = np.asarray(T, dtype=np.float64)
    x = np.asarray(XPF, dtype=np.float64)
    y = np.asarray(YPF, dtype=np.float64)
    q = np.asarray(Q, dtype=np.float64)
    con = np.asarray(CON, dtype=np.float64)
    valid = strip_valid_mask(q, con, quality_thr, contrast_frac)
    if effective_rate is not None:
        rate = float(effective_rate)
    elif len(t) > 1 and t[-1] > t[0]:
        rate = float((len(t) - 1) / (t[-1] - t[0]))
    else:
        rate = strip_rate_hz(frame_cols, fps, strip_width) if frame_cols else float("nan")
    out = dict(
        t=t, x_px=x, y_px=y, valid=valid, rate=np.float64(rate),
        x_px_immediate=np.asarray(XRAW, dtype=np.float64),
        y_px_immediate=np.asarray(YRAW, dtype=np.float64),
        q=q, con=con, strip_width=np.int64(strip_width), pad=np.int64(pad),
        nstrip=np.int64(nstrip), frame_cols=np.int64(frame_cols),
        fps=np.float64(fps), n_frames=np.int64(frame_idx),
        quality_thr=np.float64(quality_thr),
        contrast_frac=np.float64(contrast_frac),
        n_particles=np.int64(n_particles),
        pf_beta=np.float64(beta),
        pf_ess_frac=np.float64(ess_frac),
        pf_roughen_perp=np.float64(roughen_perp),
        pf_roughen_along=np.float64(roughen_along),
        pf_init_spread_px=np.float64(init_spread_px),
        ess=np.asarray(ESS, dtype=np.float64),
        max_ncc=np.asarray(MAXNCC, dtype=np.float64),
        p_saccade=np.asarray(PSACC, dtype=np.float64),
        resampled=np.asarray(RESAMP, dtype=bool),
        top_k=np.int64(top_k), fixed_lag_ms=np.float64(lag_ms),
        fixed_lag_lines=np.int64(resolver.lag if resolver is not None else 0),
        fixed_lag_resolved=np.asarray(RSLV, dtype=bool),
        hyp_count=np.asarray(HCOUNT, dtype=np.int16),
        fixed_lag_hyp_index=np.asarray(HIDX, dtype=np.int16),
        fixed_lag_hyp_rank=np.asarray(HRANK, dtype=np.int16),
        fixed_lag_hyp_logp_gap=np.asarray(HGAP, dtype=np.float64),
        fixed_lag_hyp_logp_margin=np.asarray(HMARGIN, dtype=np.float64),
        hypothesis_transition_sigma_rows=np.float64(transition_sigma_rows),
        hypothesis_obs_weight=np.float64(obs_weight),
        hypothesis_velocity_cost=np.float64(velocity_cost),
        hypothesis_acceleration_cost=np.float64(acceleration_cost),
        slew_gate=np.bool_(slew_gate),
        slew_max_deg_s=np.float64(slew_max_deg_s if slew_gate else 0.0),
        hypothesis_blend_immediate=np.bool_(blend_immediate),
        hypothesis_blend_delta_rows=np.float64(blend_delta_rows if blend_immediate else 0.0),
        hypothesis_blend_alpha=np.float64(blend_alpha if blend_immediate else 0.0),
        subject=np.asarray(sub.name),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **out)
    return out


def _delta(candidate: dict, baseline: dict, key: str) -> float:
    a = float(candidate.get(key, float("nan")))
    b = float(baseline.get(key, float("nan")))
    return a - b if np.isfinite(a) and np.isfinite(b) else float("nan")


def make_result_row(sub: pf.Subject, method: str, strip_width: int, run: dict,
                    metrics: dict, cache_path: str,
                    baseline_metrics: dict | None = None,
                    baseline_path: str = "") -> dict:
    method_id = f"strip_s{strip_width}" if method == "raw" else f"strip_{method}_s{strip_width}"
    row = dict(
        subject=sub.name,
        method=method_id,
        strip_method=method,
        strip_width=int(strip_width),
        fps=float(run.get("fps", float("nan"))),
        frame_cols=int(run.get("frame_cols", 0)),
        nstrip=int(run.get("nstrip", 0)),
        n_frames=int(run.get("n_frames", 0)),
        cache_path=cache_path,
        baseline_path=baseline_path,
    )
    row.update(metrics)
    if baseline_metrics is not None:
        for key in ("valid_frac", "r_dot_x", "prec_x", "j30",
                    "raw_jump_ge3_frac", "raw_speed_p999_px_s"):
            row[f"baseline_{key}"] = baseline_metrics.get(key, float("nan"))
            row[f"delta_{key}"] = _delta(metrics, baseline_metrics, key)
    return row


def _format_float(value, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return ""
    return f"{x:.{digits}f}" if np.isfinite(x) else ""


def write_reports(rows: list[dict], out_prefix: str) -> tuple[str, str]:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    csv_path = out_prefix + ".csv"
    md_path = out_prefix + ".md"
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    subject = str(rows[0].get("subject", "people-data")) if rows else "people-data"
    frame_cols = int(rows[0].get("frame_cols", 0)) if rows else 0
    multi_method = len({str(r.get("strip_method", "raw")) for r in rows}) > 1
    lines = [f"# {subject} SLO Strip Ladder", ""]
    lines.append("Non-overlapping SLO-column strips registered to the previous frame.")
    if multi_method:
        lines.append(
            "`raw` is immediate top-1 NCC; `resolver` feeds top-K NCC peaks into "
            "the fixed-lag path resolver; `pf` runs the IMM particle filter on "
            "the strip NCC response surface, then resolves posterior clusters."
        )
    lines.append("")
    if multi_method:
        lines.append("| method | S | rate Hz | valid | r_dot_x | prec_x | j30 | jump>=3 px | p99.9 speed px/s |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:
        lines.append("| S | rate Hz | valid | r_dot_x | prec_x | j30 | jump>=3 px | p99.9 speed px/s |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        cells = []
        if multi_method:
            cells.append(str(row.get("strip_method", row.get("method", ""))))
        cells.extend([
            str(row.get("strip_width", "")),
            _format_float(row.get("rate"), 1),
            _format_float(row.get("valid_frac")),
            _format_float(row.get("r_dot_x")),
            _format_float(row.get("prec_x")),
            _format_float(row.get("j30")),
            _format_float(row.get("raw_jump_ge3_frac")),
            _format_float(row.get("raw_speed_p999_px_s"), 1),
        ])
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if frame_cols:
        lines.append(
            f"Rates are `(frame_cols // S) * fps`; this capture has {frame_cols} columns, "
            "so `S=15/16` is the ~1 kHz baseline zone."
        )
    else:
        lines.append("Rates are `(frame_cols // S) * fps`; `S=15/16` is the ~1 kHz baseline zone.")
    lines.append("The report is a baseline characterization, not a replacement for PF caches.")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return csv_path, md_path


def run_ladder(
    subject: str = "Ashton3",
    widths: Iterable[int] = DEFAULT_WIDTHS,
    *,
    methods: Iterable[str] = DEFAULT_METHODS,
    dur_s: float | None = None,
    pad: int = 80,
    rebuild: bool = False,
    rebuild_inputs: bool = False,
    quality_thr: float = 0.35,
    contrast_frac: float = pf.CONTRAST_FRAC,
    out_prefix: str = DEFAULT_OUT_PREFIX,
    top_k: int = 5,
    lag_ms: float = flt.HYPOTHESIS_LAG_MS,
    transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS,
    obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT,
    velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
    acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
    slew_gate: bool = False,
    slew_max_deg_s: float = flt.SLEW_GATE_MAX_DEG_S,
    blend_immediate: bool = False,
    blend_delta_rows: float = flt.HYPOTHESIS_BLEND_DELTA_ROWS,
    blend_alpha: float = flt.HYPOTHESIS_BLEND_ALPHA,
    n_particles: int = PF_N_PARTICLES,
    pf_beta: float = PF_BETA,
    pf_ess_frac: float = PF_ESS_FRAC,
    pf_roughen_perp: float = PF_ROUGHEN_PERP,
    pf_roughen_along: float = PF_ROUGHEN_ALONG,
    pf_init_spread_px: float = PF_INIT_SPREAD_PX,
    pf_hypothesis_cluster_rows: float = PF_HYPOTHESIS_CLUSTER_ROWS,
) -> list[dict]:
    sub = pf.subject_by_name(subject)
    lm = pf.build_line_measurements(sub, rebuild=rebuild_inputs)
    refs = pf.compute_refs(sub, lm)
    baseline_path = os.path.join(sub.cache_dir, "m4_dpf_physics.npz")
    baseline_metrics = None
    if os.path.exists(baseline_path):
        baseline_metrics = collect_metrics(_clip_run(_load_npz(baseline_path), dur_s), refs)

    methods = list(methods)
    rows = []
    raw_metrics_by_width: dict[int, dict] = {}
    for width in widths:
        for method in methods:
            cache_path = strip_cache_path(sub, int(width), dur_s, method=method)
            if method == "raw":
                run = run_strip_tracker(
                    sub, int(width), pad=pad, dur_s=dur_s, rebuild=rebuild,
                    quality_thr=quality_thr, contrast_frac=contrast_frac,
                    cache_path=cache_path)
            elif method == "resolver":
                resolver_method = resolver_variant_name(
                    top_k=top_k, lag_ms=lag_ms,
                    transition_sigma_rows=transition_sigma_rows,
                    obs_weight=obs_weight, velocity_cost=velocity_cost,
                    acceleration_cost=acceleration_cost, slew_gate=slew_gate,
                    slew_max_deg_s=slew_max_deg_s,
                    blend_immediate=blend_immediate,
                    blend_delta_rows=blend_delta_rows,
                    blend_alpha=blend_alpha)
                cache_path = strip_cache_path(sub, int(width), dur_s,
                                              method=resolver_method)
                run = run_strip_resolver(
                    sub, int(width), pad=pad, dur_s=dur_s, rebuild=rebuild,
                    quality_thr=quality_thr, contrast_frac=contrast_frac,
                    cache_path=cache_path, top_k=top_k, lag_ms=lag_ms,
                    transition_sigma_rows=transition_sigma_rows,
                    obs_weight=obs_weight, velocity_cost=velocity_cost,
                    acceleration_cost=acceleration_cost, slew_gate=slew_gate,
                    slew_max_deg_s=slew_max_deg_s,
                    blend_immediate=blend_immediate,
                    blend_delta_rows=blend_delta_rows,
                    blend_alpha=blend_alpha)
            elif method == "pf":
                pf_method = pf_variant_name(
                    n_particles=n_particles, beta=pf_beta,
                    ess_frac=pf_ess_frac,
                    roughen_perp=pf_roughen_perp,
                    roughen_along=pf_roughen_along,
                    init_spread_px=pf_init_spread_px,
                    top_k=top_k, lag_ms=lag_ms,
                    transition_sigma_rows=transition_sigma_rows,
                    obs_weight=obs_weight, velocity_cost=velocity_cost,
                    acceleration_cost=acceleration_cost, slew_gate=slew_gate,
                    slew_max_deg_s=slew_max_deg_s,
                    blend_immediate=blend_immediate,
                    blend_delta_rows=blend_delta_rows,
                    blend_alpha=blend_alpha)
                cache_path = strip_cache_path(sub, int(width), dur_s,
                                              method=pf_method)
                run = run_strip_pf(
                    sub, int(width), pad=pad, dur_s=dur_s, rebuild=rebuild,
                    quality_thr=quality_thr, contrast_frac=contrast_frac,
                    cache_path=cache_path, n_particles=n_particles,
                    beta=pf_beta, ess_frac=pf_ess_frac,
                    roughen_perp=pf_roughen_perp,
                    roughen_along=pf_roughen_along,
                    init_spread_px=pf_init_spread_px,
                    top_k=top_k, cluster_rows=pf_hypothesis_cluster_rows,
                    lag_ms=lag_ms,
                    transition_sigma_rows=transition_sigma_rows,
                    obs_weight=obs_weight, velocity_cost=velocity_cost,
                    acceleration_cost=acceleration_cost, slew_gate=slew_gate,
                    slew_max_deg_s=slew_max_deg_s,
                    blend_immediate=blend_immediate,
                    blend_delta_rows=blend_delta_rows,
                    blend_alpha=blend_alpha)
            else:  # pragma: no cover - parse_methods guards CLI use
                raise ValueError(f"unknown strip method {method!r}")
            metrics = collect_metrics(run, refs)
            if method == "raw":
                raw_metrics_by_width[int(width)] = metrics
            row = make_result_row(
                sub, method, int(width), run, metrics, cache_path,
                baseline_metrics=baseline_metrics,
                baseline_path=baseline_path if baseline_metrics is not None else "")
            raw_metrics = raw_metrics_by_width.get(int(width))
            if method != "raw" and raw_metrics is not None:
                for key in ("valid_frac", "r_dot_x", "prec_x", "j30",
                            "raw_jump_ge3_frac", "raw_speed_p999_px_s"):
                    row[f"raw_strip_{key}"] = raw_metrics.get(key, float("nan"))
                    row[f"delta_vs_raw_strip_{key}"] = _delta(metrics, raw_metrics, key)
            rows.append(row)
            print(
                f"  [strip ladder {method}] S={int(width):>2} rate={metrics['rate']:.1f} Hz "
                f"valid={metrics['valid_frac']:.3f} r_x={metrics['r_dot_x']:.3f} "
                f"prec_x={metrics['prec_x']:.3f} j30={metrics['j30']:.3f}",
                flush=True,
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Ashton3")
    ap.add_argument("--strip-widths", default=",".join(str(w) for w in DEFAULT_WIDTHS))
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS),
                    help="comma-separated: raw, resolver, pf")
    ap.add_argument("--dur", type=float, default=None, help="optional duration cap in seconds")
    ap.add_argument("--pad", type=int, default=80)
    ap.add_argument("--quality-thr", type=float, default=0.35)
    ap.add_argument("--contrast-frac", type=float, default=pf.CONTRAST_FRAC)
    ap.add_argument("--top-k", type=int, default=5,
                    help="top-K strip NCC peaks for resolver mode")
    ap.add_argument("--lag-ms", type=float, default=flt.HYPOTHESIS_LAG_MS)
    ap.add_argument("--hypothesis-transition-sigma-rows", type=float,
                    default=flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS)
    ap.add_argument("--hypothesis-obs-weight", type=float,
                    default=flt.HYPOTHESIS_OBS_WEIGHT)
    ap.add_argument("--hypothesis-velocity-cost", type=float,
                    default=flt.HYPOTHESIS_VEL_COST)
    ap.add_argument("--hypothesis-acceleration-cost", type=float,
                    default=flt.HYPOTHESIS_ACCEL_COST)
    ap.add_argument("--slew-gate", action="store_true")
    ap.add_argument("--slew-max-deg-s", type=float, default=flt.SLEW_GATE_MAX_DEG_S)
    ap.add_argument("--hypothesis-blend-immediate", action="store_true")
    ap.add_argument("--hypothesis-blend-delta-rows", type=float,
                    default=flt.HYPOTHESIS_BLEND_DELTA_ROWS)
    ap.add_argument("--hypothesis-blend-alpha", type=float,
                    default=flt.HYPOTHESIS_BLEND_ALPHA)
    ap.add_argument("--n-particles", type=int, default=PF_N_PARTICLES,
                    help="particle count for strip PF mode")
    ap.add_argument("--pf-beta", type=float, default=PF_BETA,
                    help="strip PF NCC likelihood sharpness")
    ap.add_argument("--pf-ess-frac", type=float, default=PF_ESS_FRAC)
    ap.add_argument("--pf-roughen-perp", type=float, default=PF_ROUGHEN_PERP)
    ap.add_argument("--pf-roughen-along", type=float, default=PF_ROUGHEN_ALONG)
    ap.add_argument("--pf-init-spread-px", type=float, default=PF_INIT_SPREAD_PX)
    ap.add_argument("--pf-hypothesis-cluster-rows", type=float,
                    default=PF_HYPOTHESIS_CLUSTER_ROWS)
    ap.add_argument("--rebuild", action="store_true", help="rebuild strip caches")
    ap.add_argument("--rebuild-inputs", action="store_true",
                    help="rebuild shared people-data line/chain inputs")
    ap.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    args = ap.parse_args(argv)

    widths = parse_strip_widths(args.strip_widths)
    methods = parse_methods(args.methods)
    out_prefix = args.out_prefix
    if args.person != "Ashton3" and out_prefix == DEFAULT_OUT_PREFIX:
        out_prefix = os.path.join("results", f"people_strip_ladder_{args.person}")
    if args.dur is not None:
        out_prefix += f"_d{_tag_float(args.dur)}"
    rows = run_ladder(
        args.person, widths, methods=methods, dur_s=args.dur, pad=args.pad,
        rebuild=args.rebuild, rebuild_inputs=args.rebuild_inputs,
        quality_thr=args.quality_thr, contrast_frac=args.contrast_frac,
        out_prefix=out_prefix, top_k=args.top_k, lag_ms=args.lag_ms,
        transition_sigma_rows=args.hypothesis_transition_sigma_rows,
        obs_weight=args.hypothesis_obs_weight,
        velocity_cost=args.hypothesis_velocity_cost,
        acceleration_cost=args.hypothesis_acceleration_cost,
        slew_gate=args.slew_gate,
        slew_max_deg_s=args.slew_max_deg_s,
        blend_immediate=args.hypothesis_blend_immediate,
        blend_delta_rows=args.hypothesis_blend_delta_rows,
        blend_alpha=args.hypothesis_blend_alpha,
        n_particles=args.n_particles,
        pf_beta=args.pf_beta,
        pf_ess_frac=args.pf_ess_frac,
        pf_roughen_perp=args.pf_roughen_perp,
        pf_roughen_along=args.pf_roughen_along,
        pf_init_spread_px=args.pf_init_spread_px,
        pf_hypothesis_cluster_rows=args.pf_hypothesis_cluster_rows)
    csv_path, md_path = write_reports(rows, out_prefix)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
