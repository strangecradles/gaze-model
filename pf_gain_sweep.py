"""pf_gain_sweep.py — BETA × roughen_perp sweep on people_fov Igor (15 s first pass).

Grid: BETA ∈ {3,5,8,20} × roughen_perp ∈ {0.05,0.15,0.5} plus lock-gated default.
Metrics: HF precision, frame jitter @30 fps, r vs dot, PSD bands.

Usage:
  python pf_gain_sweep.py                  # 15 s sweep on Igor
  python pf_gain_sweep.py --person Kathy --dur 15
  python pf_gain_sweep.py --full           # full capture on top configs
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import people_fov_pf as pf

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SWEEP_DIR = os.path.join(HERE, "cache", "people_fov", "_sweep")

BETAS = (3, 5, 8, 20)
ROUGHS = (0.05, 0.15, 0.5)
BASE_R_DOT = {"Igor": 0.80, "Ashton3": 0.94, "Kathy": 0.96}


def _cache_name(beta: float | None, rp: float | None, lg: bool, dur_s: float | None) -> str:
    if lg:
        tag = "m4_lg"
    else:
        tag = f"m4_b{int(beta)}_rp{rp:g}"
    if dur_s is not None:
        tag += f"_d{int(dur_s)}"
    return tag + ".npz"


def _configs():
    cfgs = []
    for b in BETAS:
        for rp in ROUGHS:
            cfgs.append((f"b{b}_rp{rp:g}", dict(beta=float(b), roughen_perp=float(rp))))
    cfgs.append(("lock_gated", dict(lock_gated_gain=True)))
    return cfgs


def _resample_stats(run: dict, valid: np.ndarray) -> tuple[float, float]:
    """Resample fraction overall and during steady pursuit lock.

    If ~1.0 during lock, ESS->1 every step and temporal averaging never engaged —
    the "retain memory" lever was not actually pulled by lowering BETA.
    """
    res = run.get("resampled")
    if res is None or len(res) != len(valid) or valid.sum() == 0:
        return float("nan"), float("nan")
    res = np.asarray(res, bool)
    rs_frac = float(np.mean(res[valid]))
    # steady lock: high NCC + low saccade posterior (filter.LOCK_*_THR)
    ncc = run.get("max_ncc"); psa = run.get("p_saccade")
    if ncc is None or psa is None:
        return rs_frac, float("nan")
    lock = valid & (np.asarray(ncc) >= 0.35) & ((1.0 - np.asarray(psa)) >= 0.9)
    rs_lock = float(np.mean(res[lock])) if lock.sum() > 0 else float("nan")
    return rs_frac, rs_lock


def _score_row(row: dict) -> float:
    """Lower is better: HF precision primary, jitter secondary."""
    prec = row.get("prec_x", np.nan)
    j30 = row.get("jitter_30", np.nan)
    if not np.isfinite(prec):
        return 1e9
    return prec + 0.1 * (j30 if np.isfinite(j30) else 0.0)


def run_sweep(person: str, dur_s: float | None, rebuild: bool) -> list[dict]:
    sub = pf.subject_by_name(person)
    ch = pf.build_chain(sub)
    lm = pf.build_line_measurements(sub)
    refs = pf.compute_refs(sub, lm)
    os.makedirs(os.path.join(SWEEP_DIR, person), exist_ok=True)
    base_r = BASE_R_DOT.get(person, 0.75)
    rows = []
    for name, kw in _configs():
        cpath = os.path.join(SWEEP_DIR, person, _cache_name(
            kw.get("beta"), kw.get("roughen_perp"), kw.get("lock_gated_gain", False), dur_s))
        t0 = time.time()
        print(f"\n[{person}] {name} dur={dur_s} -> {os.path.basename(cpath)}")
        run = pf.run_m4(sub, lm, ch, cache_path=cpath, dur_s=dur_s,
                        rebuild=rebuild, **kw)
        valid = run["valid"].astype(bool)
        rate = float(run["rate"])
        ev = pf.evaluate(run["t"], run["x_px"], run["y_px"], valid, rate, refs)
        hx = ev["cal_x"]
        p_low, p_hf = pf.psd_band_ratio(hx, rate, valid)
        rs_frac, rs_lock = _resample_stats(run, valid)
        row = dict(
            person=person, config=name, dur_s=dur_s or "full",
            beta=kw.get("beta", "lg"),
            roughen_perp=kw.get("roughen_perp", "lg"),
            lock_gated=kw.get("lock_gated_gain", False),
            prec_x=ev["prec_x"], prec_y=ev.get("prec_y", np.nan),
            r_dot_x=ev["r_dot_x"], r_dot_y=ev["r_dot_y"],
            r_trk_x=ev["r_trk_x"], valid_frac=ev["valid_frac"],
            jitter_30=pf.frame_jitter_30fps(hx, rate, valid),
            psd_pursuit=p_low, psd_hf=p_hf,
            resample_frac=rs_frac, resample_frac_lock=rs_lock,
            elapsed_s=time.time() - t0,
        )
        ok = row["r_dot_x"] >= base_r - 0.02
        row["pass_guardrail"] = ok
        print(f"  prec_x={row['prec_x']:.2f}'  j30={row['jitter_30']:.2f}'  "
              f"r_dot={row['r_dot_x']:.2f}  resample={rs_frac:.2f}(lock {rs_lock:.2f})  "
              f"guardrail={'OK' if ok else 'FAIL'}  ({row['elapsed_s']:.0f}s)")
        rows.append(row)
    return rows


def write_outputs(rows: list[dict], tag: str) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    csv_path = os.path.join(RESULTS, f"pf_dejitter_sweep_{tag}.csv")
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    ok_rows = [r for r in rows if r["pass_guardrail"]]
    if not ok_rows:
        ok_rows = rows
    best = min(ok_rows, key=_score_row)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    grid_rows = [r for r in rows if not r["lock_gated"]]
    for rp in ROUGHS:
        sub = [r for r in grid_rows if r["roughen_perp"] == rp]
        sub.sort(key=lambda r: r["beta"])
        if not sub:
            continue
        axes[0].plot([r["beta"] for r in sub], [r["prec_x"] for r in sub],
                     "o-", label=f"rp={rp}")
    lg = [r for r in rows if r["lock_gated"]]
    if lg:
        axes[0].axhline(lg[0]["prec_x"], color="C3", ls="--", label="lock-gated")
    axes[0].set_xlabel("BETA"); axes[0].set_ylabel("HF precision x (′)")
    axes[0].set_title(f"De-jitter sweep ({tag})"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for rp in ROUGHS:
        sub = [r for r in grid_rows if r["roughen_perp"] == rp]
        sub.sort(key=lambda r: r["beta"])
        if not sub:
            continue
        axes[1].plot([r["beta"] for r in sub], [r["jitter_30"] for r in sub],
                     "o-", label=f"rp={rp}")
    if lg:
        axes[1].axhline(lg[0]["jitter_30"], color="C3", ls="--", label="lock-gated")
    axes[1].set_xlabel("BETA"); axes[1].set_ylabel("Frame jitter @30 fps (′)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(RESULTS, f"pf_dejitter_sweep_{tag}.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Best (guardrailed): {best['config']} prec_x={best['prec_x']:.2f}' "
          f"j30={best['jitter_30']:.2f}' r_dot={best['r_dot_x']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Igor")
    ap.add_argument("--dur", type=float, default=15.0, help="duration cap (s); 0=full")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--full", action="store_true", help="full-length on top-3 configs")
    args = ap.parse_args()
    dur = None if args.dur <= 0 else args.dur
    tag = f"{args.person}_d{int(dur)}" if dur else f"{args.person}_full"
    rows = run_sweep(args.person, dur, args.rebuild)
    write_outputs(rows, tag)

    if args.full and dur is not None:
        ok_rows = [r for r in rows if r["pass_guardrail"]]
        top = sorted(ok_rows or rows, key=_score_row)[:3]
        print(f"\n=== Full-length rerun: top {len(top)} configs ===")
        full_rows = []
        for r in top:
            name = r["config"]
            kw = dict(lock_gated_gain=True) if r["lock_gated"] else dict(
                beta=float(r["beta"]), roughen_perp=float(r["roughen_perp"]))
            sub = pf.subject_by_name(args.person)
            ch = pf.build_chain(sub)
            lm = pf.build_line_measurements(sub)
            refs = pf.compute_refs(sub, lm)
            cpath = os.path.join(SWEEP_DIR, args.person, _cache_name(
                kw.get("beta"), kw.get("roughen_perp"), kw.get("lock_gated_gain", False), None))
            run = pf.run_m4(sub, lm, ch, cache_path=cpath, dur_s=None, rebuild=True, **kw)
            valid = run["valid"].astype(bool)
            rate = float(run["rate"])
            ev = pf.evaluate(run["t"], run["x_px"], run["y_px"], valid, rate, refs)
            hx = ev["cal_x"]
            p_low, p_hf = pf.psd_band_ratio(hx, rate, valid)
            full_rows.append(dict(
                person=args.person, config=name + "_full", dur_s="full",
                beta=kw.get("beta", "lg"), roughen_perp=kw.get("roughen_perp", "lg"),
                lock_gated=kw.get("lock_gated_gain", False),
                prec_x=ev["prec_x"], prec_y=ev.get("prec_y", np.nan),
                r_dot_x=ev["r_dot_x"], r_dot_y=ev["r_dot_y"],
                r_trk_x=ev["r_trk_x"], valid_frac=ev["valid_frac"],
                jitter_30=pf.frame_jitter_30fps(hx, rate, valid),
                psd_pursuit=p_low, psd_hf=p_hf, elapsed_s=0, pass_guardrail=True,
            ))
        write_outputs(full_rows, f"{args.person}_full_top3")


if __name__ == "__main__":
    main()
