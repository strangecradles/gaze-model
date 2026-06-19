"""Reality checks for strip-PF retinal motion on people_data_fov captures.

This is intentionally ground-truth-free. It asks whether high-rate strip-PF
motion is reproducible across independent-ish observations and whether the
artifact metrics scale like real motion or differentiated measurement noise.
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

import people_fov_pf as pf
import people_strip_ladder as ladder

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
ARC = pf.ARC_PER_PX


def _load_npz(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _tag_float(x: float) -> str:
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def _rms(x: np.ndarray) -> float:
    x = _finite(x)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def _pct(x: np.ndarray, q: float) -> float:
    x = _finite(x)
    return float(np.percentile(x, q)) if x.size else float("nan")


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return float("nan")
    aa = a[m] - float(np.mean(a[m]))
    bb = b[m] - float(np.mean(b[m]))
    d = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / d) if d > 0 else float("nan")


def _prep(run: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(run["t"], dtype=np.float64)
    x = np.asarray(run["x_px"], dtype=np.float64) * ARC
    v = np.asarray(run["valid"], dtype=bool) & np.isfinite(t) & np.isfinite(x)
    return t[v], x[v], v


def _interp_pair(run_a: dict, run_b: dict, hz: float | None = None) -> dict:
    ta, xa, _ = _prep(run_a)
    tb, xb, _ = _prep(run_b)
    if len(ta) < 100 or len(tb) < 100:
        return {}
    t0 = max(float(ta.min()), float(tb.min()))
    t1 = min(float(ta.max()), float(tb.max()))
    if t1 <= t0:
        return {}
    ra = float(run_a["rate"])
    rb = float(run_b["rate"])
    grid_hz = float(hz) if hz is not None else min(ra, rb, 1000.0)
    grid = np.arange(t0, t1, 1.0 / grid_hz)
    if len(grid) < 100:
        return {}
    ya = np.interp(grid, ta, xa)
    yb = np.interp(grid, tb, xb)
    return dict(grid=grid, a=ya, b=yb, rate=grid_hz)


def pair_agreement(run_a: dict, run_b: dict, label: str) -> dict:
    p = _interp_pair(run_a, run_b)
    if not p:
        return dict(kind="pair", label=label, n=0)
    diff = (p["a"] - p["b"]) / np.sqrt(2.0)
    k25 = max(1.0, p["rate"] * 25.0 / 1000.0)
    k50 = max(1.0, p["rate"] * 50.0 / 1000.0)
    hf = diff - gaussian_filter1d(diff, k25)
    slow = gaussian_filter1d(diff, k50)
    a_slow = gaussian_filter1d(p["a"], k50)
    b_slow = gaussian_filter1d(p["b"], k50)
    return dict(
        kind="pair",
        label=label,
        n=int(len(diff)),
        grid_hz=float(p["rate"]),
        rms_all_arcmin=_rms(diff),
        rms_hf25_arcmin=_rms(hf),
        rms_slow50_arcmin=_rms(slow),
        corr_slow50=_corr(a_slow, b_slow),
    )


def step_metrics(run: dict) -> dict:
    valid = np.asarray(run["valid"], dtype=bool)
    x = np.asarray(run["x_px"], dtype=np.float64)
    rate = float(run["rate"])
    ok = valid[1:] & valid[:-1] & np.isfinite(x[1:]) & np.isfinite(x[:-1])
    step = np.abs(np.diff(x)[ok])
    if not step.size:
        return dict(jump_ge3_frac=float("nan"), step_p999_px=float("nan"),
                    speed_p999_px_s=float("nan"), speed_p999_arcmin_s=float("nan"))
    speed = step * rate
    return dict(
        jump_ge3_frac=float(np.mean(step >= 3.0)),
        step_p999_px=_pct(step, 99.9),
        speed_p999_px_s=_pct(speed, 99.9),
        speed_p999_arcmin_s=_pct(speed * ARC, 99.9),
    )


def evidence_metrics(run: dict, method: str, width: int) -> dict:
    out = dict(kind="evidence", method=method, strip_width=width,
               rate=float(run["rate"]), n=int(len(run["t"])),
               valid_frac=float(np.mean(np.asarray(run["valid"], dtype=bool))))
    out.update(step_metrics(run))
    if "max_ncc" in run:
        max_ncc = np.asarray(run["max_ncc"], dtype=np.float64)
        out.update(max_ncc_p10=_pct(max_ncc, 10), max_ncc_med=_pct(max_ncc, 50))
    if "ess" in run:
        ess = np.asarray(run["ess"], dtype=np.float64)
        n_particles = float(np.asarray(run.get("n_particles", np.nan)))
        out.update(ess_frac_med=_pct(ess / n_particles, 50) if n_particles else float("nan"))
    if "resampled" in run:
        out["resampled_frac"] = float(np.mean(np.asarray(run["resampled"], dtype=bool)))
    if "hyp_count" in run:
        hc = np.asarray(run["hyp_count"], dtype=np.float64)
        out["multi_hyp_frac"] = float(np.mean(hc > 1))
        out["hyp_count_med"] = _pct(hc, 50)
    if "fixed_lag_hyp_logp_margin" in run:
        margin = np.asarray(run["fixed_lag_hyp_logp_margin"], dtype=np.float64)
        out["hyp_margin_p10"] = _pct(margin[np.isfinite(margin)], 10)
        out["hyp_margin_med"] = _pct(margin[np.isfinite(margin)], 50)
    if "x_px_immediate" in run:
        x = np.asarray(run["x_px"], dtype=np.float64)
        xi = np.asarray(run["x_px_immediate"], dtype=np.float64)
        v = np.asarray(run["valid"], dtype=bool) & np.isfinite(x) & np.isfinite(xi)
        out["rms_vs_immediate_arcmin"] = _rms((x[v] - xi[v]) * ARC)
    return out


def _method_cache(sub: pf.Subject, method: str, width: int, dur_s: float) -> str:
    if method == "raw":
        return ladder.strip_cache_path(sub, width, dur_s, method="raw")
    if method == "pf":
        return ladder.strip_cache_path(sub, width, dur_s, method=ladder.pf_variant_name())
    if method == "resolver":
        return ladder.strip_cache_path(sub, width, dur_s, method=ladder.resolver_variant_name())
    raise ValueError(method)


def _split_keep(mode: str, parity: int, frame_cols: int, strip_width: int):
    nstrip = frame_cols // strip_width

    def keep(global_i: int, _frame_idx: int, _strip_idx: int) -> bool:
        if mode == "evenodd":
            return (global_i % 2) == parity
        if mode == "frameblock":
            return ((global_i // max(nstrip, 1)) % 2) == parity
        raise ValueError(mode)

    return keep


def run_pf_split(sub: pf.Subject, width: int, dur_s: float, mode: str,
                 parity: int, rebuild: bool = False) -> dict:
    frame_cols = 808
    full_rate = frame_cols // width * 18.603074141048825
    eff_rate = full_rate / 2.0
    tag = f"strip_ladder_pf_split_{mode}{parity}_s{width}_d{_tag_float(dur_s)}.npz"
    cache_path = os.path.join(sub.cache_dir, tag)
    return ladder.run_strip_pf(
        sub,
        width,
        dur_s=dur_s,
        rebuild=rebuild,
        cache_path=cache_path,
        sample_keep=_split_keep(mode, parity, frame_cols, width),
        effective_rate=eff_rate,
    )


def rate_scaling_rows(runs_by_method: dict[str, dict[int, dict]]) -> list[dict]:
    rows = []
    for method, by_width in runs_by_method.items():
        rates, speeds = [], []
        for width, run in sorted(by_width.items()):
            m = step_metrics(run)
            rate = float(run["rate"])
            speed = float(m["speed_p999_px_s"])
            rows.append(dict(kind="rate_point", method=method, strip_width=width,
                             rate=rate, speed_p999_px_s=speed,
                             jump_ge3_frac=m["jump_ge3_frac"]))
            if np.isfinite(rate) and rate > 0 and np.isfinite(speed) and speed > 0:
                rates.append(rate)
                speeds.append(speed)
        if len(rates) >= 3:
            slope = float(np.polyfit(np.log(rates), np.log(speeds), 1)[0])
        else:
            slope = float("nan")
        rows.append(dict(kind="rate_slope", method=method,
                         loglog_speed_rate_slope=slope, n=len(rates)))
    return rows


def write_outputs(rows: list[dict], out_prefix: str) -> tuple[str, str]:
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    csv_path = out_prefix + ".csv"
    md_path = out_prefix + ".md"
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = ["# Strip-PF Reality Checks", ""]
    lines.append("Metrics are ground-truth-free checks of reproducibility and data support. `r_dot_x` is intentionally not used.")
    slopes = [r for r in rows if r.get("kind") == "rate_slope"]
    if slopes:
        lines.append("")
        lines.append("## Rate Scaling")
        lines.append("| method | log-log p99.9 speed vs rate slope | n |")
        lines.append("|---|---:|---:|")
        for r in slopes:
            lines.append(f"| {r.get('method','')} | {float(r.get('loglog_speed_rate_slope', np.nan)):.3f} | {r.get('n','')} |")
        lines.append("")
        lines.append("Slope near 1 means fixed-size measurement jumps are being differentiated; flatter is better.")
    pairs = [r for r in rows if r.get("kind") == "pair"]
    if pairs:
        lines.append("")
        lines.append("## Repeatability")
        lines.append("| label | n | all RMS ' | HF25 RMS ' | slow50 RMS ' | slow corr |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in pairs:
            lines.append(
                f"| {r.get('label','')} | {r.get('n','')} | "
                f"{float(r.get('rms_all_arcmin', np.nan)):.3f} | "
                f"{float(r.get('rms_hf25_arcmin', np.nan)):.3f} | "
                f"{float(r.get('rms_slow50_arcmin', np.nan)):.3f} | "
                f"{float(r.get('corr_slow50', np.nan)):.3f} |"
            )
    evs = [r for r in rows if r.get("kind") == "evidence"]
    if evs:
        lines.append("")
        lines.append("## Evidence")
        lines.append("| method | S | max NCC med | ESS frac med | multi hyp | RMS vs immediate ' | jump>=3 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in evs:
            if r.get("method") != "pf":
                continue
            lines.append(
                f"| {r.get('method','')} | {r.get('strip_width','')} | "
                f"{float(r.get('max_ncc_med', np.nan)):.3f} | "
                f"{float(r.get('ess_frac_med', np.nan)):.3f} | "
                f"{float(r.get('multi_hyp_frac', np.nan)):.3f} | "
                f"{float(r.get('rms_vs_immediate_arcmin', np.nan)):.3f} | "
                f"{float(r.get('jump_ge3_frac', np.nan)):.3f} |"
            )
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Ashton3")
    ap.add_argument("--dur", type=float, default=5.0)
    ap.add_argument("--widths", default="64,32,16,15,8,4,2,1")
    ap.add_argument("--methods", default="raw,pf")
    ap.add_argument("--run-splits", action="store_true")
    ap.add_argument("--split-widths", default="15,1")
    ap.add_argument("--split-dur", type=float, default=2.0)
    ap.add_argument("--rebuild-splits", action="store_true")
    ap.add_argument("--out-prefix", default=os.path.join("results", "strip_pf_reality_checks_Ashton3"))
    args = ap.parse_args(argv)

    sub = pf.subject_by_name(args.person)
    widths = ladder.parse_strip_widths(args.widths)
    methods = ladder.parse_methods(args.methods)
    runs_by_method: dict[str, dict[int, dict]] = {m: {} for m in methods}
    rows: list[dict] = []

    for method in methods:
        for width in widths:
            path = _method_cache(sub, method, width, args.dur)
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing {path}; run people_strip_ladder first")
            run = _load_npz(path)
            runs_by_method[method][width] = run
            rows.append(evidence_metrics(run, method, width))

    rows.extend(rate_scaling_rows(runs_by_method))

    for method, by_width in runs_by_method.items():
        if 15 in by_width and 1 in by_width:
            rows.append(pair_agreement(by_width[1], by_width[15], f"{method}:S1_vs_S15"))
        if 16 in by_width and 1 in by_width:
            rows.append(pair_agreement(by_width[1], by_width[16], f"{method}:S1_vs_S16"))
        if 2 in by_width and 1 in by_width:
            rows.append(pair_agreement(by_width[1], by_width[2], f"{method}:S1_vs_S2"))

    if args.run_splits:
        split_widths = ladder.parse_strip_widths(args.split_widths)
        for width in split_widths:
            for mode in ("evenodd", "frameblock"):
                a = run_pf_split(sub, width, args.split_dur, mode, 0, args.rebuild_splits)
                b = run_pf_split(sub, width, args.split_dur, mode, 1, args.rebuild_splits)
                rows.append(pair_agreement(a, b, f"pf_split:{mode}:S{width}:d{_tag_float(args.split_dur)}"))

    out_prefix = args.out_prefix
    if args.person != "Ashton3" and out_prefix.endswith("Ashton3"):
        out_prefix = out_prefix[:-len("Ashton3")] + args.person
    out_prefix += f"_d{_tag_float(args.dur)}"
    csv_path, md_path = write_outputs(rows, out_prefix)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
