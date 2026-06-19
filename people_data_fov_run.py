"""people_data_fov_run.py — particle-filter gaze tracking for people_data_fov/.

Builds per-subject chain + line caches, runs M4 physics PF (optionally
lock-gated de-jitter params), and can delegate animation to people_data_fov_anim.

Usage:
  python people_data_fov_run.py
  python people_data_fov_run.py --person Ashton3
  python people_data_fov_run.py --lock-gated --cache-tag m4_dpf_physics_lg
  python people_data_fov_run.py --dur 15   # partial run for testing
  python people_data_fov_run.py --anim --anim-style raw  # diagnostic/raw display
"""
from __future__ import annotations

import argparse
import os

from along_quality import AlongQualityModel
import filter as flt
import people_fov_pf as pf


def process_subject(sub: pf.Subject, *, rebuild: bool, dur_s: float | None,
                    cache_tag: str, lock_gated: bool, beta: float | None,
                    roughen_perp: float | None, anim: bool,
                    anim_style: str, sdslo_upgrade: bool,
                    aq_model: str | None, aq_sigma_min: float,
                    aq_sigma_max: float, aq_gamma: float,
                    top_k: int, slew_gate: bool,
                    slew_max_deg_s: float,
                    hypothesis_velocity_cost: float,
                    hypothesis_velocity_sigma_deg_s: float,
                    hypothesis_acceleration_cost: float,
                    hypothesis_acceleration_sigma_deg_s2: float,
                    motion_prior: bool,
                    motion_prior_sigma_rows: float,
                    motion_prior_tau_ms: float,
                    motion_prior_ncc_thr: float) -> None:
    print(f"\n=== {sub.name} ({sub.stem}) ===")
    ch = pf.build_chain(sub, rebuild=rebuild)
    lm = pf.build_line_measurements(sub, rebuild=rebuild)
    refs = pf.compute_refs(sub, lm)
    print(f"  clock OFF={refs['off']:.2f}s (chain-vs-dot r={refs['off_r']:+.2f})")

    kw = {}
    if lock_gated:
        kw["lock_gated_gain"] = True
    if beta is not None:
        kw["beta"] = beta
    if roughen_perp is not None:
        kw["roughen_perp"] = roughen_perp
    if sdslo_upgrade:
        kw["quality_scaled_along"] = True
        kw["along_sigma_max"] = 18.0
        kw["multi_hypothesis"] = True
    if aq_model is not None:
        if aq_model == "constant":
            model = AlongQualityModel.constant()
        elif aq_model == "qv-power":
            model = AlongQualityModel.fit_qv_power(
                lm["qv"], aq_sigma_min, aq_sigma_max, aq_gamma)
        else:
            raise ValueError(f"unknown AQ model {aq_model!r}")
        kw["quality_scaled_along"] = True
        kw["along_quality_model"] = model
        kw["multi_hypothesis"] = True
        kw["hypothesis_top_k"] = int(top_k)
    if slew_gate:
        kw["slew_gate"] = True
        kw["slew_max_deg_s"] = float(slew_max_deg_s)
    if hypothesis_velocity_cost != flt.HYPOTHESIS_VEL_COST:
        kw["hypothesis_velocity_cost"] = float(hypothesis_velocity_cost)
        kw["hypothesis_velocity_sigma_deg_s"] = float(hypothesis_velocity_sigma_deg_s)
    if hypothesis_acceleration_cost != flt.HYPOTHESIS_ACCEL_COST:
        kw["hypothesis_acceleration_cost"] = float(hypothesis_acceleration_cost)
        kw["hypothesis_acceleration_sigma_deg_s2"] = float(hypothesis_acceleration_sigma_deg_s2)
    if motion_prior:
        kw["mosaic_prior"] = True
        kw["mosaic_prior_sigma_rows"] = float(motion_prior_sigma_rows)
        kw["mosaic_prior_track_tau_s"] = float(motion_prior_tau_ms) / 1000.0
        kw["mosaic_prior_ncc_thr"] = float(motion_prior_ncc_thr)

    cache_path = os.path.join(sub.cache_dir, f"{cache_tag}.npz")
    run = pf.run_m4(sub, lm, ch, cache_path=cache_path, dur_s=dur_s,
                    rebuild=rebuild, **kw)
    valid = run["valid"].astype(bool)
    rate = float(run["rate"])
    ev = pf.evaluate(run["t"], run["x_px"], run["y_px"], valid, rate, refs)
    hx = ev["cal_x"]
    print(f"  DPF @ {rate:.0f} Hz, {len(run['t'])} samples, "
          f"in-FOV {valid.mean()*100:.0f}% -> {cache_path}")
    print(f"  r-vs-dot x={ev['r_dot_x']:.2f} y={ev['r_dot_y']:.2f}; "
          f"prec_x={ev['prec_x']:.2f}'  j30={pf.frame_jitter_30fps(hx, rate, valid):.2f}'")

    if anim:
        import people_data_fov_anim as anim
        anim.render(anim.prepare_traces(sub.name, anim_style, cache_tag=cache_tag))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--dur", type=float, default=None, help="cap duration (s)")
    ap.add_argument("--lock-gated", action="store_true")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--roughen-perp", type=float, default=None)
    ap.add_argument("--sdslo-upgrade", action="store_true",
                    help="enable quality-scaled along + multi-hypothesis PF")
    ap.add_argument("--aq-model", choices=("constant", "qv-power"), default=None,
                    help="explicit calibrated along-quality model for PF sigma")
    ap.add_argument("--aq-sigma-min", type=float, default=2.0)
    ap.add_argument("--aq-sigma-max", type=float, default=10.0)
    ap.add_argument("--aq-gamma", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=5,
                    help="top-K line-match hypotheses for AQ calibrated runs")
    ap.add_argument("--slew-gate", action="store_true",
                    help="enable physiological pursuit-slew gate in fixed-lag resolver")
    ap.add_argument("--slew-max-deg-s", type=float, default=flt.SLEW_GATE_MAX_DEG_S)
    ap.add_argument("--hypothesis-velocity-cost", type=float, default=flt.HYPOTHESIS_VEL_COST)
    ap.add_argument("--hypothesis-velocity-sigma-deg-s", type=float,
                    default=flt.HYPOTHESIS_VEL_SIGMA_DEG_S)
    ap.add_argument("--hypothesis-acceleration-cost", type=float,
                    default=flt.HYPOTHESIS_ACCEL_COST)
    ap.add_argument("--hypothesis-acceleration-sigma-deg-s2", type=float,
                    default=flt.HYPOTHESIS_ACCEL_SIGMA_DEG_S2)
    ap.add_argument("--motion-prior", action="store_true",
                    help="enable default-off in-PF SDSLO EMA motion prior")
    ap.add_argument("--motion-prior-sigma-rows", type=float, default=2.0)
    ap.add_argument("--motion-prior-tau-ms", type=float, default=3.0)
    ap.add_argument("--motion-prior-ncc-thr", type=float, default=0.2)
    ap.add_argument("--cache-tag", default="m4_dpf_physics")
    ap.add_argument("--anim", action="store_true")
    ap.add_argument("--anim-style", choices=("oculomotor", "raw"), default="oculomotor")
    args = ap.parse_args()

    subs = pf.discover_subjects()
    if args.person:
        subs = [s for s in subs if s.name == args.person]
        if not subs:
            raise SystemExit(f"unknown person {args.person!r}")
    cache_tag = args.cache_tag
    if args.sdslo_upgrade and cache_tag == "m4_dpf_physics":
        cache_tag = "m4_dpf_physics_sdslo"
    if args.aq_model is not None and args.cache_tag == "m4_dpf_physics":
        if args.aq_model == "constant":
            tag_model = AlongQualityModel.constant()
        else:
            tag_model = AlongQualityModel.qv_power(
                args.aq_sigma_min, args.aq_sigma_max, args.aq_gamma, 0.0, 1.0)
        cache_tag = f"m4_dpf_physics_aq_{tag_model.config_tag()}"
    if args.slew_gate and args.cache_tag == "m4_dpf_physics":
        cache_tag += f"_sg{args.slew_max_deg_s:g}".replace(".", "p")
    if args.hypothesis_velocity_cost != flt.HYPOTHESIS_VEL_COST and args.cache_tag == "m4_dpf_physics":
        cache_tag += f"_vc{args.hypothesis_velocity_cost:g}".replace(".", "p")
    if args.hypothesis_acceleration_cost != flt.HYPOTHESIS_ACCEL_COST and args.cache_tag == "m4_dpf_physics":
        cache_tag += f"_ac{args.hypothesis_acceleration_cost:g}".replace(".", "p")
    if args.motion_prior and args.cache_tag == "m4_dpf_physics":
        sigma = f"{args.motion_prior_sigma_rows:g}".replace(".", "p")
        tau = f"{args.motion_prior_tau_ms:g}".replace(".", "p")
        cache_tag += f"_mp_s{sigma}_tau{tau}"
    print(f"Processing {len(subs)} subject(s): {[s.name for s in subs]}")
    for sub in subs:
        process_subject(sub, rebuild=args.rebuild, dur_s=args.dur,
                        cache_tag=cache_tag, lock_gated=args.lock_gated,
                        beta=args.beta, roughen_perp=args.roughen_perp,
                        anim=args.anim, anim_style=args.anim_style,
                        sdslo_upgrade=args.sdslo_upgrade,
                        aq_model=args.aq_model,
                        aq_sigma_min=args.aq_sigma_min,
                        aq_sigma_max=args.aq_sigma_max,
                        aq_gamma=args.aq_gamma,
                        top_k=args.top_k,
                        slew_gate=args.slew_gate,
                        slew_max_deg_s=args.slew_max_deg_s,
                        hypothesis_velocity_cost=args.hypothesis_velocity_cost,
                        hypothesis_velocity_sigma_deg_s=args.hypothesis_velocity_sigma_deg_s,
                        hypothesis_acceleration_cost=args.hypothesis_acceleration_cost,
                        hypothesis_acceleration_sigma_deg_s2=args.hypothesis_acceleration_sigma_deg_s2,
                        motion_prior=args.motion_prior,
                        motion_prior_sigma_rows=args.motion_prior_sigma_rows,
                        motion_prior_tau_ms=args.motion_prior_tau_ms,
                        motion_prior_ncc_thr=args.motion_prior_ncc_thr)


if __name__ == "__main__":
    main()
