"""dewarp_ab.py — Task 4: decisive A/B of the atlas dewarp on rdx_scatter.

OFF = committed line cache (atlas = chain-aligned previous frame).
ON  = lines_dewarp.npz (atlas additionally intra-frame dewarped, horizontal path).

Metric identical to Task 1: rdx_scatter = std(rdx - 10ms_smooth)[mask] * ARC_PER_PX.
Comparison is PAIRED on a COMMON FOV mask (OFF-mask AND ON-mask) so the same lines
are scored under both. Bootstrap CI is over FRAMES (resample frames w/ replacement).

  fraction_owned = (scatter_off - scatter_on) / scatter_off

Emit results/dewarp_ab.json. (Run dewarp_diag-style build with dewarp_atlas=True first;
this script only reads caches and will error if lines_dewarp.npz is missing.)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

import khz2d
import people_fov_pf as pf
from dewarp_diag import residual_px

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX


def scatter_from_resid(resid: np.ndarray, mask: np.ndarray) -> float:
    return float(np.nanstd(resid[mask]) * ARC)


def per_frame_resid(lm: dict, resid: np.ndarray, mask: np.ndarray):
    """Group masked squared-residuals by frame for a frame-level bootstrap."""
    fr = lm["frame"]
    order = np.argsort(fr[mask], kind="stable")
    f_sorted = fr[mask][order]
    r_sorted = resid[mask][order]
    uf, starts = np.unique(f_sorted, return_index=True)
    groups = np.split(r_sorted, starts[1:])
    return uf, groups


def boot_scatter(groups_off, groups_on, n_boot=2000, seed=0):
    """Bootstrap CI over frames for scatter_off, scatter_on, fraction_owned."""
    rng = np.random.default_rng(seed)
    nf = len(groups_off)
    # precompute per-frame sum(x^2) and count for fast std (mean assumed ~0; we use
    # the pooled std about 0 of the residual, which equals std since residual~0-mean)
    soff = np.array([np.sum(g ** 2) for g in groups_off])
    non = np.array([g.size for g in groups_off])
    son = np.array([np.sum(g ** 2) for g in groups_on])
    nn = np.array([g.size for g in groups_on])
    off_b = np.empty(n_boot); on_b = np.empty(n_boot); frac_b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, nf, nf)
        o = np.sqrt(soff[idx].sum() / max(non[idx].sum(), 1)) * ARC
        n = np.sqrt(son[idx].sum() / max(nn[idx].sum(), 1)) * ARC
        off_b[i] = o; on_b[i] = n
        frac_b[i] = (o - n) / o if o > 0 else 0.0
    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(off_ci=ci(off_b), on_ci=ci(on_b), frac_ci=ci(frac_b),
                frac_mean=float(frac_b.mean()), n_boot=n_boot)


def run_one(person: str) -> dict:
    sub = pf.subject_by_name(person)
    lm_off = pf.build_line_measurements(sub)                       # committed cache
    lm_on = pf.build_line_measurements(sub, dewarp_atlas=True)     # lines_dewarp.npz
    assert lm_off["rdx"].shape == lm_on["rdx"].shape, "OFF/ON line grids differ"

    m_off = pf.fov_mask(lm_off)
    m_on = pf.fov_mask(lm_on)
    m = m_off & m_on                                               # common, paired
    r_off = residual_px(lm_off, m)
    r_on = residual_px(lm_on, m)

    s_off = scatter_from_resid(r_off, m)
    s_on = scatter_from_resid(r_on, m)
    frac = (s_off - s_on) / s_off if s_off > 0 else 0.0
    frac_vs_433 = (4.33 - s_on) / 4.33

    uf_off, g_off = per_frame_resid(lm_off, r_off, m)
    uf_on, g_on = per_frame_resid(lm_on, r_on, m)
    boot = boot_scatter(g_off, g_on)

    out = dict(
        person=person, capture=sub.stem,
        scatter_off_arcmin=s_off, scatter_on_arcmin=s_on,
        fraction_owned=frac, fraction_owned_vs_4p33=frac_vs_433,
        n_lines_common=int(m.sum()), n_frames=int(len(g_off)),
        bootstrap=boot,
        note=("paired common-mask; fraction_owned=(off-on)/off; "
              "bootstrap CI over frames; gaze source = chain within-frame velocity ramp"),
    )
    print(f"[AB {person}] OFF={s_off:.3f}'  ON={s_on:.3f}'  "
          f"fraction_owned={frac:+.3f}  CI={boot['frac_ci']}  (n={int(m.sum())})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="+", default=["Igor"])
    args = ap.parse_args()
    results = {}
    for p in args.people:
        try:
            results[p] = run_one(p)
        except FileNotFoundError as e:
            print(f"[AB {p}] SKIP — missing dewarp cache: {e}")
        except Exception as e:  # noqa
            print(f"[AB {p}] ERROR: {e}")
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "dewarp_ab.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("-> results/dewarp_ab.json")


if __name__ == "__main__":
    main()
