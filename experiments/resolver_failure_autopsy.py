"""Autopsy fixed-lag resolver regressions on people-data captures.

This script is intentionally diagnostic: it does not run the tracker. It uses
existing full-capture caches to find the time windows where a candidate
resolved trajectory worsens the same gates used by along_quality_calib.py.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

import along_quality_calib as aqc
import khz2d
import people_fov_pf as pf


DEFAULT_OUT = os.path.join("results", "resolver_failure_autopsy")
DEFAULT_PAVEL_CANDIDATE = "m4_dpf_physics_aq_constant_sg100_mp_s2_tau6.npz"


@dataclass(frozen=True)
class Trace:
    label: str
    source_cache: str
    run: dict
    cal_x: np.ndarray
    hx: np.ndarray
    hf_x: np.ndarray
    valid: np.ndarray
    j30_t: np.ndarray
    j30_d2: np.ndarray
    j30_ok: np.ndarray


def _fmt(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(x):
        return "nan"
    if abs(x) < 0.01 and x != 0:
        return f"{x:.2e}"
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _md_table(rows: list[dict], cols: list[str]) -> list[str]:
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    return out


def _load(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _subject(name: str) -> pf.Subject:
    matches = [s for s in pf.discover_subjects() if s.name == name]
    if not matches:
        raise SystemExit(f"unknown subject {name!r}")
    return matches[0]


def _cache_path(sub: pf.Subject, cache: str) -> str:
    return cache if os.path.isabs(cache) else os.path.join(sub.cache_dir, cache)


def _immediate_run(run: dict) -> dict:
    if "x_px_immediate" not in run or "y_px_immediate" not in run:
        raise ValueError("candidate cache has no immediate PF trace")
    out = dict(run)
    out["x_px"] = np.asarray(run["x_px_immediate"], dtype=np.float64)
    out["y_px"] = np.asarray(run["y_px_immediate"], dtype=np.float64)
    return out


def _x_components(run: dict, refs: dict, smooth_ms: float = 2.0):
    t = np.asarray(run["t"], dtype=np.float64)
    rate = float(run["rate"])
    valid = np.asarray(run["valid"], dtype=bool)
    vx = np.where(valid, np.asarray(run["x_px"], dtype=np.float64), np.nan)
    if smooth_ms > 0 and rate > 0:
        k = max(1.0, rate * smooth_ms / 1000.0)
        vx = gaussian_filter1d(khz2d.fill_nan(vx), k)
        vx[~valid] = np.nan
    hx = khz2d._drift_removed(vx, rate)
    dot_x = np.interp(t, refs["dot_t"] + refs["off"], refs["dot_x"])
    m = valid & np.isfinite(hx)
    if m.sum() > 20 and np.nanstd(hx[m]) > 0:
        A = np.c_[hx[m], np.ones(m.sum())]
        coef, *_ = np.linalg.lstsq(A, dot_x[m], rcond=None)
        cal_x = coef[0] * hx + coef[1]
    else:
        cal_x = hx * np.nan
    xf = khz2d.fill_nan(hx)
    hf_x = xf - gaussian_filter1d(xf, max(1.0, rate * 25 / 1000.0))
    return t, rate, valid, hx, hf_x, cal_x


def _mapped_j30(t: np.ndarray, cal_x: np.ndarray, valid: np.ndarray, rate: float,
                smooth_ms: float = 2.0):
    xf = gaussian_filter1d(khz2d.fill_nan(cal_x),
                           max(1.0, rate * smooth_ms / 1000.0))
    step = max(1, int(round(rate / 30.0)))
    idx = np.arange(0, len(xf), step, dtype=np.int64)
    x30 = xf[idx]
    m30 = valid[idx] & np.isfinite(x30)
    if len(x30) < 3:
        return np.array([]), np.array([]), np.array([], dtype=bool)
    ok = m30[:-2] & m30[1:-1] & m30[2:]
    d2 = np.abs(x30[2:] - 2.0 * x30[1:-1] + x30[:-2])
    return t[idx[1:-1]], d2, ok


def _make_trace(label: str, cache: str, run: dict, refs: dict) -> Trace:
    t, rate, valid, hx, hf_x, cal_x = _x_components(run, refs)
    jt, jd2, jok = _mapped_j30(t, cal_x, valid, rate)
    return Trace(label, cache, run, cal_x, hx, hf_x, valid, jt, jd2, jok)


def _rolling_precision_windows(base: Trace, cand: Trace, window_s: float,
                               stride_s: float, top_n: int) -> list[dict]:
    t = np.asarray(cand.run["t"], dtype=np.float64)
    rate = float(cand.run["rate"])
    win = max(10, int(round(window_s * rate)))
    stride = max(1, int(round(stride_s * rate)))
    common = base.valid & cand.valid & np.isfinite(base.hf_x) & np.isfinite(cand.hf_x)
    delta_sq = np.where(common, cand.hf_x ** 2 - base.hf_x ** 2, 0.0)
    cand_sq = np.where(common, cand.hf_x ** 2, 0.0)
    base_sq = np.where(common, base.hf_x ** 2, 0.0)
    cnt = np.cumsum(np.r_[0.0, common.astype(np.float64)])
    dcs = np.cumsum(np.r_[0.0, delta_sq])
    ccs = np.cumsum(np.r_[0.0, cand_sq])
    bcs = np.cumsum(np.r_[0.0, base_sq])
    rows = []
    for start in range(0, max(1, len(t) - win + 1), stride):
        end = min(len(t), start + win)
        n = cnt[end] - cnt[start]
        if n < max(50, 0.2 * win):
            continue
        delta_mean = (dcs[end] - dcs[start]) / n
        cand_rms = math.sqrt(max(0.0, (ccs[end] - ccs[start]) / n))
        base_rms = math.sqrt(max(0.0, (bcs[end] - bcs[start]) / n))
        rows.append(dict(
            source="precision",
            start_s=float(t[start]),
            end_s=float(t[end - 1]),
            score=float(delta_mean),
            baseline_prec=base_rms,
            candidate_prec=cand_rms,
            delta_prec=cand_rms - base_rms,
        ))
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]


def _rolling_j30_windows(base: Trace, cand: Trace, window_s: float,
                         stride_s: float, top_n: int) -> list[dict]:
    if cand.j30_t.size == 0 or base.j30_t.size != cand.j30_t.size:
        return []
    dt = float(np.nanmedian(np.diff(cand.j30_t))) if cand.j30_t.size > 1 else 1 / 30.0
    win = max(3, int(round(window_s / max(dt, 1e-6))))
    stride = max(1, int(round(stride_s / max(dt, 1e-6))))
    common = base.j30_ok & cand.j30_ok
    rows = []
    for start in range(0, max(1, len(cand.j30_t) - win + 1), stride):
        end = min(len(cand.j30_t), start + win)
        m = common[start:end]
        if m.sum() < max(3, 0.2 * win):
            continue
        bj = float(np.median(base.j30_d2[start:end][m]))
        cj = float(np.median(cand.j30_d2[start:end][m]))
        rows.append(dict(
            source="j30",
            start_s=float(cand.j30_t[start]),
            end_s=float(cand.j30_t[end - 1]),
            score=cj - bj,
            baseline_j30=bj,
            candidate_j30=cj,
            delta_j30=cj - bj,
        ))
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:top_n]


def _local_j30(trace: Trace, start_s: float, end_s: float) -> float:
    m = (trace.j30_t >= start_s) & (trace.j30_t <= end_s) & trace.j30_ok
    return float(np.median(trace.j30_d2[m])) if m.sum() >= 3 else float("nan")


def _raw_step_stats(run: dict, start_s: float, end_s: float) -> dict:
    t = np.asarray(run["t"], dtype=np.float64)
    x = np.asarray(run["x_px"], dtype=np.float64)
    valid = np.asarray(run["valid"], dtype=bool)
    m = (t >= start_s) & (t <= end_s)
    step_ok = m[1:] & m[:-1] & valid[1:] & valid[:-1] & np.isfinite(x[1:]) & np.isfinite(x[:-1])
    step = np.abs(np.diff(x)[step_ok])
    if not step.size:
        return dict(
            raw_jump_ge3_frac=float("nan"),
            raw_jump_return_ge3_frac=float("nan"),
            raw_step_p999_px=float("nan"),
            raw_speed_p999_px_s=float("nan"),
        )
    signed = np.diff(x)[step_ok]
    ret = float("nan")
    if signed.size >= 2:
        return_jump = (
            (np.abs(signed[:-1]) >= 3.0)
            & (np.abs(signed[1:]) >= 3.0)
            & (signed[:-1] * signed[1:] < 0.0)
        )
        ret = float(np.mean(return_jump))
    rate = float(run["rate"])
    return dict(
        raw_jump_ge3_frac=float(np.mean(step >= 3.0)),
        raw_jump_return_ge3_frac=ret,
        raw_step_p999_px=float(np.percentile(step, 99.9)),
        raw_speed_p999_px_s=float(np.percentile(step * rate, 99.9)),
    )


def _audit_stats(run: dict, start_s: float, end_s: float) -> dict:
    t = np.asarray(run["t"], dtype=np.float64)
    m = (t >= start_s) & (t <= end_s) & np.asarray(run["valid"], dtype=bool)
    out = {}
    if "x_px_immediate" in run:
        dx = np.asarray(run["x_px"], dtype=np.float64) - np.asarray(run["x_px_immediate"], dtype=np.float64)
        vv = m & np.isfinite(dx)
        out["resolved_minus_immediate_rms_px"] = (
            float(np.sqrt(np.nanmean(dx[vv] ** 2))) if np.any(vv) else float("nan"))
        out["resolved_minus_immediate_p95_px"] = (
            float(np.nanpercentile(np.abs(dx[vv]), 95)) if np.any(vv) else float("nan"))
    for key, func in (
        ("hyp_count", np.nanmean),
        ("qv", np.nanmedian),
        ("con", np.nanmedian),
        ("p_saccade", np.nanmedian),
        ("max_ncc", np.nanmedian),
        ("along_sigma_eff", np.nanmedian),
    ):
        if key in run:
            vals = np.asarray(run[key], dtype=np.float64)
            out[key] = float(func(vals[m])) if np.any(m) else float("nan")
    if "fixed_lag_hyp_rank" in run:
        rank = np.asarray(run["fixed_lag_hyp_rank"])
        ok = m & (rank >= 0)
        out["resolver_override_frac"] = float(np.mean(rank[ok] > 0)) if np.any(ok) else float("nan")
        out["resolver_rank_p95"] = float(np.percentile(rank[ok], 95)) if np.any(ok) else float("nan")
    if "fixed_lag_blended_immediate" in run:
        blended = np.asarray(run["fixed_lag_blended_immediate"], dtype=bool)
        ok = m & np.asarray(run["valid"], dtype=bool)
        out["resolver_blend_frac"] = float(np.mean(blended[ok])) if np.any(ok) else float("nan")
    if "fixed_lag_hyp_logp_gap" in run:
        gap = np.asarray(run["fixed_lag_hyp_logp_gap"], dtype=np.float64)
        ok = m & np.isfinite(gap)
        out["resolver_logp_gap_median"] = float(np.median(gap[ok])) if np.any(ok) else float("nan")
        out["resolver_logp_gap_p95"] = float(np.percentile(gap[ok], 95)) if np.any(ok) else float("nan")
    if "fixed_lag_local_best_x_px" in run:
        dx_best = (np.asarray(run["x_px"], dtype=np.float64)
                   - np.asarray(run["fixed_lag_local_best_x_px"], dtype=np.float64))
        ok = m & np.isfinite(dx_best)
        out["resolved_minus_local_best_rms_px"] = (
            float(np.sqrt(np.nanmean(dx_best[ok] ** 2))) if np.any(ok) else float("nan"))
    return out


def _window_rows(windows: list[dict], traces: list[Trace]) -> list[dict]:
    rows = []
    for wi, win in enumerate(windows, 1):
        for tr in traces:
            start = float(win["start_s"])
            end = float(win["end_s"])
            m = ((np.asarray(tr.run["t"], dtype=np.float64) >= start)
                 & (np.asarray(tr.run["t"], dtype=np.float64) <= end)
                 & tr.valid & np.isfinite(tr.hf_x))
            row = dict(
                window_id=wi,
                source=win["source"],
                window_score=win["score"],
                start_s=start,
                end_s=end,
                trace=tr.label,
                cache=tr.source_cache,
                local_prec=float(np.sqrt(np.nanmean(tr.hf_x[m] ** 2))) if np.any(m) else float("nan"),
                local_j30=_local_j30(tr, start, end),
            )
            row.update(_raw_step_stats(tr.run, start, end))
            row.update(_audit_stats(tr.run, start, end))
            rows.append(row)
    return rows


def _write_reports(out_prefix: str, summary: list[dict], windows: list[dict],
                   rows: list[dict]) -> tuple[str, str]:
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    csv_path = f"{out_prefix}.csv"
    md_path = f"{out_prefix}.md"
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# Resolver Failure Autopsy",
        "",
        "This report ranks local windows where the resolved candidate worsens "
        "the same precision and 30 fps jitter signals used by calibration.",
        "",
        "## Global Summary",
        "",
    ]
    md.extend(_md_table(summary, [
        "trace", "pass_hard_gates", "raw_jump_reduction",
        "raw_speed_p999_reduction", "raw_jump_return_reduction",
        "delta_prec_x", "delta_j30",
        "prec_x", "j30",
    ]))
    md.extend(["", "## Ranked Windows", ""])
    md.extend(_md_table(windows, [
        "source", "start_s", "end_s", "score",
        "baseline_prec", "candidate_prec", "delta_prec",
        "baseline_j30", "candidate_j30", "delta_j30",
    ]))
    md.extend(["", "## Window Trace Comparison", ""])
    md.extend(_md_table(rows, [
        "window_id", "source", "start_s", "end_s", "trace",
        "local_prec", "local_j30", "raw_jump_ge3_frac",
        "raw_jump_return_ge3_frac", "raw_step_p999_px",
        "raw_speed_p999_px_s", "resolved_minus_immediate_rms_px",
        "resolver_blend_frac", "resolver_override_frac",
        "resolver_logp_gap_median",
        "qv", "con", "p_saccade", "max_ncc",
    ]))
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    return csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="Pavel")
    ap.add_argument("--baseline-cache", default="m4_dpf_physics.npz")
    ap.add_argument("--candidate-cache", default=DEFAULT_PAVEL_CANDIDATE)
    ap.add_argument("--comparison-caches", default="",
                    help="comma-separated additional resolved caches to include")
    ap.add_argument("--precision-window-s", type=float, default=1.0)
    ap.add_argument("--j30-window-s", type=float, default=2.0)
    ap.add_argument("--stride-s", type=float, default=0.25)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--out-prefix", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    sub = _subject(args.subject)
    lm = pf.build_line_measurements(sub)
    refs = pf.compute_refs(sub, lm)

    baseline_path = _cache_path(sub, args.baseline_cache)
    candidate_path = _cache_path(sub, args.candidate_cache)
    baseline_run = _load(baseline_path)
    candidate_run = _load(candidate_path)

    baseline = _make_trace("baseline", args.baseline_cache, baseline_run, refs)
    resolved = _make_trace("resolved", args.candidate_cache, candidate_run, refs)
    traces = [baseline, resolved]
    if "x_px_immediate" in candidate_run:
        traces.append(_make_trace("immediate", args.candidate_cache + "::immediate",
                                  _immediate_run(candidate_run), refs))
    for cache in [c.strip() for c in args.comparison_caches.split(",") if c.strip()]:
        run = _load(_cache_path(sub, cache))
        traces.append(_make_trace(cache.replace(".npz", ""), cache, run, refs))

    base_metrics = aqc.collect_metrics(baseline_run, refs)
    summary = []
    for tr in traces:
        metrics = aqc.collect_metrics(tr.run, refs)
        comp = aqc.compare_to_baseline(metrics, base_metrics)
        row = dict(trace=tr.label, **metrics, **comp)
        summary.append(row)

    precision_windows = _rolling_precision_windows(
        baseline, resolved, args.precision_window_s, args.stride_s, args.top_n)
    j30_windows = _rolling_j30_windows(
        baseline, resolved, args.j30_window_s, args.stride_s, args.top_n)
    windows = precision_windows + j30_windows
    windows.sort(key=lambda r: (r["source"], -float(r["score"])))
    rows = _window_rows(windows, traces)
    paths = _write_reports(args.out_prefix, summary, windows, rows)
    print(f"Wrote {paths[0]} and {paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
