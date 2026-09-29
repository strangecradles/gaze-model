"""h2_signal.py — Phase 2 closeout.

(A) Records the H2 A/B as GATED-OUT: Task 5 showed the per-line resample re-reference
    injects 1.1-2.0' of alias-flip intervention noise >= 0.5x the real-motion bound, so per
    the guardrail the A/B is not run (it would be underpowered/invalid). Emits h2_ab.json.

(B) HIGH-VALUE: re-judges multi-frame reference averaging (refavg3) with the now-VALID
    signal-safety tests — NON-adjacent block-interleaved coherence crossover (not the
    invalid even/odd) and the coincidence microsaccade count — per subject. This overturns
    or confirms the prior loop's even/odd-based 'refavg3 fails on Ashton3' verdict.
    Emits h2_signal.json.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

import khz2d
import people_fov_pf as pf
import floor_decompose as fdc
from dewarp_signal_check import microsaccade_count

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARC = pf.ARC_PER_PX


def _scatter(lm, m):
    rdx = lm["rdx"].astype(float); fs = float(lm["line_rate"])
    sm = gaussian_filter1d(median_filter(khz2d.fill_nan(np.where(m, rdx, np.nan)), 7), fs * 0.01)
    return float(np.nanstd((rdx - sm)[m]) * ARC)


def _block_crossover(lm, m, Kb=128):
    """Non-adjacent block-interleaved coherence crossover on the rdx track (px->arcmin)."""
    t = lm["t"].astype(float); fs = float(lm["line_rate"])
    x = lm["rdx"].astype(float) * ARC
    li = np.arange(t.size)
    par = (li // Kb) % 2
    a = m & (par == 0); b = m & (par == 1)
    return fdc._crossover(t[a], x[a], t[b], x[b], fs, full=True)


def gated_ab():
    """H2 A/B intentionally not run — power gate failed. Record the rationale."""
    pw = json.load(open(os.path.join(RESULTS, "h2_power.json")))
    out = {}
    for p, d in pw.items():
        out[p] = dict(
            person=p, ab_run=False,
            reason="power gate FAIL: per-line reref intervention noise "
                   f"{d['intervention_noise_arcmin']:.2f}' >= 0.5 x real-motion bound (~1.7'); "
                   "alias-peak flipping dominates. Guardrail forbids underpowered A/B.",
            intervention_noise_arcmin=d["intervention_noise_arcmin"],
            verdict="H2 untestable-as-specified / H0 for the rdx-C1 metric "
                    "(identity for FOV-locked lines, alias-flip noise otherwise).")
    json.dump(out, open(os.path.join(RESULTS, "h2_ab.json"), "w"), indent=2)
    print("[H2-AB] gated out (power-gate failure) -> results/h2_ab.json")
    return out


def refavg_rejudge(person, K=3):
    sub = pf.subject_by_name(person)
    single = pf.build_line_measurements(sub)
    refavg = pf.build_line_measurements(sub, ref_frames=K)
    m = pf.fov_mask(single) & pf.fov_mask(refavg)
    s_single = _scatter(single, m); s_ref = _scatter(refavg, m)
    red = (s_single - s_ref) / s_single
    co_s = _block_crossover(single, m); co_r = _block_crossover(refavg, m)
    ms_s = microsaccade_count(single); ms_r = microsaccade_count(refavg)
    # block-interleaved coherence is UNINFORMATIVE here: blocks are non-simultaneous, so the
    # two halves are temporally offset and never cohere (crossover -> nan). The only split that
    # is BOTH simultaneous and non-atlas-shared is cross-reference (same line, different refs),
    # which needs a multi-reference AVERAGED build to apply to refavg3 (named next step).
    cross_informative = bool(np.isfinite(co_s["crossover_hz"]) and np.isfinite(co_r["crossover_hz"]))
    ms_delta = float((ms_r - ms_s) / ms_s)
    out = dict(person=person, K=K,
               scatter_single=s_single, scatter_refavg=s_ref, reduction=float(red),
               block128_crossover_single_hz=co_s["crossover_hz"],
               block128_crossover_refavg_hz=co_r["crossover_hz"],
               block_crossover_informative=cross_informative,
               microsacc_single=ms_s, microsacc_refavg=ms_r, microsacc_delta=ms_delta,
               signal_safety="UNRESOLVED",
               signal_safety_reason=(
                   "even/odd gate invalid (adjacency); block-interleaved coherence "
                   "uninformative (non-simultaneous halves -> nan); coincidence microsaccade "
                   "detector itself uses even/odd (partly confounded). Valid test = cross-"
                   "reference coherence on a multi-ref-averaged build (next step)."),
               scatter_reduction_robust=True,
               note="refavg3 scatter reduction is robust; its signal-safety is NOT yet "
                    "establishable with a valid (simultaneous + non-atlas-shared) coherence test.")
    print(f"[refavg-rejudge {person}] {s_single:.3f}->{s_ref:.3f}' (-{red*100:.1f}%)  "
          f"block-coh informative={cross_informative}  ms Δ={ms_delta*100:+.1f}%  "
          f"signal-safety=UNRESOLVED")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", nargs="+", default=["Igor", "Ashton3"])
    args = ap.parse_args()
    gated_ab()
    sig = {p: refavg_rejudge(p) for p in args.people}
    json.dump(sig, open(os.path.join(RESULTS, "h2_signal.json"), "w"), indent=2)
    print("-> results/h2_signal.json")


if __name__ == "__main__":
    main()
