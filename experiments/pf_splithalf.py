"""pf_splithalf.py — split-half repeatability test for the PF gaze estimator.

The decisive, ground-truth-free test for the ~3' fixation-jitter floor. Estimate
gaze twice from disjoint halves of the lines, interpolate to a common grid, and
measure RMS[(A-B)/sqrt(2)] -- the estimator's own measurement-noise floor.

  * What is COMMON to A and B but varies at HF is real eye motion (plus error
    correlated across the lines each half keeps).
  * What DIFFERS between A and B is independent per-line measurement noise.

Two split modes (per the critique):
  A1 even/odd lines    -- adjacent; measures estimator + adjacent-correlated noise.
  A2 interleaved blocks -- K-line blocks alternate A,B; exposes slowly-varying
                           atlas/decoder registration error common to A1's halves.

Verdict (HF band, 25 ms high-pass, vs the full-run single-trace HF "floor"):
  split_hf << floor            -> floor is mostly REAL eye motion (halves agree).
  split_hf ~= floor            -> floor is MEASUREMENT noise (halves disagree).
  A1 split_hf small, A2 large   -> registration error dominates.

Usage:
  python3 pf_splithalf.py --person Igor --dur 15
  python3 pf_splithalf.py --person Ashton3 --dur 0      # full capture
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
from scipy.ndimage import gaussian_filter1d

import people_fov_pf as pf

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SPLIT_DIR = os.path.join(HERE, "cache", "people_fov", "_split")
ARC = pf.ARC_PER_PX                      # arcmin per raster pixel (60/126)

# Operating point whose floor we are explaining (recommended in pf_dejitter.md).
OP_NAME = "b3_rp0.15"
OP_KW = dict(beta=3.0, roughen_perp=0.15)
TAG = ""                                 # variant suffix for Phase-D observation-model runs

BLOCK_K = 808                            # one frame of lines (A2 block size)


def _keep(mode: str, parity: int, K: int = BLOCK_K):
    if mode == "evenodd":
        return lambda gi: (gi % 2) == parity
    if mode == "block":
        return lambda gi: ((gi // K) % 2) == parity
    raise ValueError(mode)


def _run_half(sub, lm, ch, mode, parity, dur_s, rebuild):
    tag = f"m4_split_{mode}_{parity}{TAG}" + (f"_d{int(dur_s)}" if dur_s else "")
    cpath = os.path.join(SPLIT_DIR, sub.name, tag + ".npz")
    os.makedirs(os.path.dirname(cpath), exist_ok=True)
    # Independent RNG per half: the only thing shared between A and B must be the
    # real eye motion + measurement error, never the particle-filter noise.
    return pf.run_m4(sub, lm, ch, cache_path=cpath, dur_s=dur_s, rebuild=rebuild,
                     seed=11 + parity, line_keep=_keep(mode, parity), **OP_KW)


def _near(tv: np.ndarray, grid: np.ndarray, gap: float) -> np.ndarray:
    """True where a sorted-time series tv has a sample within `gap` of each grid pt."""
    if len(tv) == 0:
        return np.zeros(len(grid), bool)
    idx = np.clip(np.searchsorted(tv, grid), 1, len(tv) - 1)
    d = np.minimum(grid - tv[idx - 1], tv[idx] - grid)
    return d < gap


def split_metric(runA: dict, runB: dict, rate: float, band: str = "hf",
                 ms: float = 25.0, K: int = BLOCK_K) -> dict:
    """RMS[(A-B)/sqrt2] on a common grid.

    band="hf"   -> even/odd halves (~85 us apart): 25 ms high-pass disagreement.
                   Real HF motion is common (cancels); independent per-line noise remains.
    band="slow" -> frame-parity halves (~33 ms blocks): 50 ms low-pass disagreement.
                   Slow drift is common; differing per-frame registration offset remains.
    """
    nan = dict(rms_all=np.nan, rms_hf=np.nan, rms_slow=np.nan, n=0, Rc=np.nan, band=band)

    def prep(r):
        t = np.asarray(r["t"], float)
        x = np.asarray(r["x_px"], float) * ARC
        v = r["valid"].astype(bool) & np.isfinite(x)
        return t[v], x[v]
    tA, xA = prep(runA)
    tB, xB = prep(runB)
    if len(tA) < 100 or len(tB) < 100:
        return nan
    t0 = max(tA.min(), tB.min()); t1 = min(tA.max(), tB.max())
    if not (t1 > t0):
        return nan

    if band == "hf":
        Rc = rate / 2.0                               # each half ~ half the line rate
        gap = 0.001                                   # ~1 ms: even/odd lines are ~85 us apart
        smooth_ms = ms
    else:                                             # slow registration probe
        Rc = 480.0
        gap = 1.5 * K / rate                          # ~1.5 frame periods
        smooth_ms = 50.0
    grid = np.arange(t0, t1, 1.0 / Rc)
    if len(grid) < 100:
        return nan
    ga = np.interp(grid, tA, xA); gb = np.interp(grid, tB, xB)
    gv = _near(tA, grid, gap) & _near(tB, grid, gap)
    if gv.sum() < 100:
        return {**nan, "n": int(gv.sum()), "Rc": float(Rc)}
    diff = (ga - gb) / np.sqrt(2.0)
    k = max(1.0, Rc * smooth_ms / 1000.0)
    if band == "hf":
        comp = diff - gaussian_filter1d(diff, k)      # high-pass
        rms_hf = float(np.sqrt(np.mean(comp[gv] ** 2))); rms_slow = np.nan
    else:
        comp = gaussian_filter1d(diff, k)             # low-pass
        rms_slow = float(np.sqrt(np.mean(comp[gv] ** 2))); rms_hf = np.nan
    return dict(
        rms_all=float(np.sqrt(np.mean(diff[gv] ** 2))),
        rms_hf=rms_hf, rms_slow=rms_slow,
        n=int(gv.sum()), Rc=float(Rc), band=band,
        grid=grid, diff=diff, comp=comp, gv=gv, ga=ga, gb=gb,
    )


def _single_hf(run: dict, ms: float = 25.0) -> float:
    x = np.asarray(run["x_px"], float) * ARC
    return pf.hf_precision_arcmin(x, float(run["rate"]), run["valid"].astype(bool), ms=ms)


def _half_eval(run: dict, refs: dict) -> dict:
    ev = pf.evaluate(run["t"], run["x_px"], run["y_px"], run["valid"].astype(bool),
                     float(run["rate"]), refs)
    return dict(prec_x=ev["prec_x"], r_dot_x=ev["r_dot_x"],
                valid_frac=ev["valid_frac"], n_valid=int(run["valid"].sum()))


def run_subject(person: str, dur_s: float | None, rebuild: bool) -> dict:
    sub = pf.subject_by_name(person)
    ch = pf.build_chain(sub)
    lm = pf.build_line_measurements(sub)
    refs = pf.compute_refs(sub, lm)

    # Full-run reference at the same operating point (the "floor" we explain).
    full_cpath = os.path.join(SPLIT_DIR, sub.name,
                              f"m4_full_{OP_NAME}{TAG}" + (f"_d{int(dur_s)}" if dur_s else "") + ".npz")
    os.makedirs(os.path.dirname(full_cpath), exist_ok=True)
    full = pf.run_m4(sub, lm, ch, cache_path=full_cpath, dur_s=dur_s,
                     rebuild=rebuild, **OP_KW)
    floor_hf = _single_hf(full)
    full_eval = _half_eval(full, refs)
    rate = float(full["rate"])

    rows = []
    plot = {}
    for mode, band in (("evenodd", "hf"), ("block", "slow")):
        t0 = time.time()
        A = _run_half(sub, lm, ch, mode, 0, dur_s, rebuild)
        B = _run_half(sub, lm, ch, mode, 1, dur_s, rebuild)
        m = split_metric(A, B, rate, band=band)
        ea, eb = _half_eval(A, refs), _half_eval(B, refs)
        primary = m["rms_hf"] if band == "hf" else m["rms_slow"]
        row = dict(
            person=person, mode=mode, band=band, op=OP_NAME, dur_s=dur_s or "full",
            floor_hf=floor_hf, split_primary=primary, split_rms_all=m["rms_all"],
            split_rms_hf=m["rms_hf"], split_rms_slow=m["rms_slow"],
            ratio=(primary / floor_hf if floor_hf else np.nan),
            n_common=m["n"], Rc=m["Rc"],
            full_prec_x=full_eval["prec_x"], full_r_dot_x=full_eval["r_dot_x"],
            A_prec_x=ea["prec_x"], B_prec_x=eb["prec_x"],
            A_r_dot_x=ea["r_dot_x"], B_r_dot_x=eb["r_dot_x"],
            A_valid_frac=ea["valid_frac"], B_valid_frac=eb["valid_frac"],
            elapsed_s=time.time() - t0,
        )
        rows.append(row)
        plot[mode] = m
        print(f"[{person}/{mode}:{band}] floor_hf={floor_hf:.3f}'  "
              f"split_{band}={primary:.3f}'  ratio={row['ratio']:.2f}  "
              f"all={m['rms_all']:.3f}'  n={m['n']}  ({row['elapsed_s']:.0f}s)")
    return dict(rows=rows, plot=plot, floor_hf=floor_hf,
                full_eval=full_eval, person=person, dur_s=dur_s)


def verdict(rows: list[dict]) -> str:
    by = {r["mode"]: r for r in rows}
    eo = by.get("evenodd", {}); bl = by.get("block", {})
    floor = eo.get("floor_hf", np.nan)
    hf = eo.get("split_rms_hf", np.nan)
    slow = bl.get("split_rms_slow", np.nan)
    if not np.isfinite(hf) or not floor:
        return "INCONCLUSIVE (no valid common samples)"
    r_hf = hf / floor
    reg = np.isfinite(slow) and slow > 0.5 * floor          # registration a real contributor
    if r_hf >= 0.5:
        s = (f"MEASUREMENT-LIMITED: even/odd halves disagree at HF (split_hf={hf:.2f}' vs "
             f"floor={floor:.2f}', ratio={r_hf:.2f}) -> the HF floor is per-line measurement "
             "noise, NOT real eye motion. -> Phase D-measurement (observation model).")
        if reg:
            s += (f" Frame-parity slow disagreement is also large (slow={slow:.2f}') -> "
                  "atlas/decoder REGISTRATION error is a second contributor.")
        return s
    if reg:
        return (f"REGISTRATION-LIMITED: even/odd agree at HF (split_hf={hf:.2f}' << "
                f"floor={floor:.2f}') but frame-parity halves disagree in the slow band "
                f"(slow={slow:.2f}') -> slowly-varying atlas/decoder registration error "
                "dominates. -> Phase D-measurement (registration).")
    return (f"MOTION-LIMITED: halves agree at HF (split_hf={hf:.2f}' << floor={floor:.2f}') and "
            f"frame-parity slow disagreement is small (slow={slow:.2f}') -> the ~{floor:.1f}' "
            "floor is mostly REAL fixational eye motion. -> Phase D-motion.")


def write_outputs(res: dict) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    person = res["person"]
    rows = res["rows"]
    csv_path = os.path.join(RESULTS, f"pf_splithalf_{person}{TAG}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode in zip(axes, ("evenodd", "block")):
        m = res["plot"].get(mode, {})
        if "grid" in m:
            g = m["grid"]; n = min(len(g), int(m["Rc"] * 0.5))  # ~0.5 s window
            sl = slice(0, n)
            ax.plot(g[sl], m["ga"][sl], lw=0.6, label="A")
            ax.plot(g[sl], m["gb"][sl], lw=0.6, label="B")
            ax.plot(g[sl], (m["ga"][sl] - m["gb"][sl]), lw=0.6, color="k",
                    label="A-B")
            prim = m["rms_hf"] if m.get("band") == "hf" else m["rms_slow"]
            ax.set_title(f"{mode} ({m.get('band')}): split={prim:.2f}'  "
                         f"floor={res['floor_hf']:.2f}'")
        ax.set_xlabel("t (s)"); ax.set_ylabel("perp (arcmin)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"Split-half repeatability — {person} ({OP_NAME})")
    fig.tight_layout()
    png = os.path.join(RESULTS, f"pf_splithalf_{person}{TAG}.png")
    fig.savefig(png, dpi=150); plt.close(fig)
    print(f"Wrote {png}")


def append_md(all_res: list[dict]) -> None:
    md = os.path.join(RESULTS, f"pf_splithalf{TAG}.md")
    lines = ["# Split-half repeatability — the decisive 3' floor test\n"]
    lines.append("`RMS[(A-B)/sqrt2]` on disjoint line halves, interpolated to a common "
                 "half-rate grid. `floor_hf` = full-run single-trace HF precision "
                 f"(25 ms high-pass) at operating point `{OP_NAME}`. All in arcmin "
                 f"(scale {ARC:.4f}'/px).\n")
    lines.append("| subj | mode | band | floor_hf | split | ratio | split_all | "
                 "full r·dot | A/B r·dot | A/B prec | n |")
    lines.append("|------|------|------|---------:|------:|------:|----------:|"
                 "-----------:|----------:|---------:|---:|")
    for res in all_res:
        for r in res["rows"]:
            lines.append(
                f"| {r['person']} | {r['mode']} | {r['band']} | {r['floor_hf']:.3f} | "
                f"{r['split_primary']:.3f} | {r['ratio']:.2f} | {r['split_rms_all']:.3f} | "
                f"{r['full_r_dot_x']:.3f} | {r['A_r_dot_x']:.2f}/{r['B_r_dot_x']:.2f} | "
                f"{r['A_prec_x']:.2f}/{r['B_prec_x']:.2f} | {r['n_common']} |")
    lines.append("")
    for res in all_res:
        lines.append(f"**{res['person']} verdict:** {verdict(res['rows'])}")
    lines.append("")
    with open(md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {md}")


def main():
    global TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default="Igor", help="comma-separated subject list")
    ap.add_argument("--dur", type=float, default=15.0, help="duration cap (s); 0=full")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--hp-sigma", type=float, default=None,
                    help="Phase-D: fine-band high-pass sigma override (default 6)")
    ap.add_argument("--likelihood", default=None, choices=[None, "physics", "learned"],
                    help="Phase-D: observation likelihood override")
    ap.add_argument("--tag", default=None, help="output/cache suffix for Phase-D variants")
    args = ap.parse_args()
    dur = None if args.dur <= 0 else args.dur
    if args.hp_sigma is not None:
        OP_KW["hp_sigma"] = args.hp_sigma
    if args.likelihood is not None:
        OP_KW["likelihood"] = args.likelihood
    if args.tag:
        TAG = "_" + args.tag.lstrip("_")
    all_res = []
    for person in [p.strip() for p in args.person.split(",") if p.strip()]:
        res = run_subject(person, dur, args.rebuild)
        write_outputs(res)
        print(f"\nVERDICT [{person}]: {verdict(res['rows'])}")
        all_res.append(res)
    append_md(all_res)


if __name__ == "__main__":
    main()
