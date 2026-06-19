"""Calibrate SDSLO along-quality models on longer people-data captures.

This harness keeps canonical people-data caches untouched. Candidate runs are
written as ``m4_dpf_physics_aq_<model>[_dN].npz`` and compared against each
subject's existing ``m4_dpf_physics.npz`` baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np

from along_quality import AlongQualityModel
import filter as flt
import people_fov_pf as pf

RESULT_PREFIX = os.path.join("results", "along_quality_calibration")
POLICY_PREFIX = os.path.join("results", "along_quality_policy")
SIGMA_MIN_GRID = (2.0, 3.0, 4.0)
SIGMA_MAX_GRID = (6.0, 10.0, 14.0, 18.0)
GAMMA_GRID = (0.5, 1.0, 2.0, 3.0)
KEY_METRICS = (
    "valid_frac", "r_dot_x", "r_dot_y", "prec_x", "j30",
    "raw_jump_ge3_frac", "raw_step_p999_px", "raw_speed_p999_px_s",
    "raw_speed_p999_arcmin_s", "raw_jump_return_ge3_frac",
    "oculo_r_dot_x", "oculo_prec_x", "oculo_j30",
)


@dataclass(frozen=True)
class CandidateConfig:
    sigma_min: float
    sigma_max: float
    gamma: float
    kind: str = "qv_power"

    @property
    def tag(self) -> str:
        if self.kind == "constant":
            return "constant"
        model = AlongQualityModel.qv_power(
            self.sigma_min, self.sigma_max, self.gamma, 0.0, 1.0)
        return model.config_tag()

    def fit_model(self, qv) -> AlongQualityModel:
        if self.kind == "constant":
            return AlongQualityModel.constant()
        return AlongQualityModel.fit_qv_power(
            qv, self.sigma_min, self.sigma_max, self.gamma)


@dataclass(frozen=True)
class RunVariant:
    cfg: CandidateConfig
    lag_ms: float = flt.HYPOTHESIS_LAG_MS
    transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS
    obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT
    blend_immediate: bool = flt.HYPOTHESIS_BLEND_IMMEDIATE
    blend_delta_rows: float = flt.HYPOTHESIS_BLEND_DELTA_ROWS
    blend_alpha: float = flt.HYPOTHESIS_BLEND_ALPHA
    blend_saccade_p: float = flt.HYPOTHESIS_BLEND_SACCADE_P
    slew_gate: bool = False
    slew_max_deg_s: float = flt.SLEW_GATE_MAX_DEG_S
    velocity_cost: float = flt.HYPOTHESIS_VEL_COST
    velocity_sigma_deg_s: float = flt.HYPOTHESIS_VEL_SIGMA_DEG_S
    acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST
    acceleration_sigma_deg_s2: float = flt.HYPOTHESIS_ACCEL_SIGMA_DEG_S2
    motion_prior: bool = False
    motion_prior_sigma_rows: float = 2.0
    motion_prior_tau_s: float = 0.003
    motion_prior_ncc_thr: float = 0.2

    @property
    def variant_id(self) -> str:
        tag = self.cfg.tag
        if self.lag_ms != flt.HYPOTHESIS_LAG_MS:
            tag += f"_lag{_tag_float(self.lag_ms)}"
        if self.transition_sigma_rows != flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS:
            tag += f"_ts{_tag_float(self.transition_sigma_rows)}"
        if self.obs_weight != flt.HYPOTHESIS_OBS_WEIGHT:
            tag += f"_ow{_tag_float(self.obs_weight)}"
        if self.blend_immediate:
            tag += (
                f"_bi_d{_tag_float(self.blend_delta_rows)}"
                f"_a{_tag_float(self.blend_alpha)}"
            )
            if self.blend_saccade_p != flt.HYPOTHESIS_BLEND_SACCADE_P:
                tag += f"_sp{_tag_float(self.blend_saccade_p)}"
        if self.slew_gate:
            tag += f"_sg{_tag_float(self.slew_max_deg_s)}"
        if self.velocity_cost != flt.HYPOTHESIS_VEL_COST:
            tag += f"_vc{_tag_float(self.velocity_cost)}"
        if self.acceleration_cost != flt.HYPOTHESIS_ACCEL_COST:
            tag += f"_ac{_tag_float(self.acceleration_cost)}"
        if self.motion_prior:
            tag += _motion_prior_tag(self.motion_prior_sigma_rows,
                                     self.motion_prior_tau_s)
        return tag

    @property
    def cache_tag(self) -> str:
        return cache_tag_for_config(
            self.cfg, slew_gate=self.slew_gate,
            lag_ms=self.lag_ms,
            transition_sigma_rows=self.transition_sigma_rows,
            obs_weight=self.obs_weight,
            blend_immediate=self.blend_immediate,
            blend_delta_rows=self.blend_delta_rows,
            blend_alpha=self.blend_alpha,
            blend_saccade_p=self.blend_saccade_p,
            slew_max_deg_s=self.slew_max_deg_s,
            velocity_cost=self.velocity_cost,
            acceleration_cost=self.acceleration_cost,
            motion_prior=self.motion_prior,
            motion_prior_sigma_rows=self.motion_prior_sigma_rows,
            motion_prior_tau_s=self.motion_prior_tau_s)


def candidate_grid(max_configs: int | None = None) -> list[CandidateConfig]:
    out = [
        CandidateConfig(sigma_min=smin, sigma_max=smax, gamma=gamma)
        for smin in SIGMA_MIN_GRID
        for smax in SIGMA_MAX_GRID
        for gamma in GAMMA_GRID
        if smax >= smin
    ]
    return out[:max_configs] if max_configs is not None else out


def config_lookup() -> dict[str, CandidateConfig]:
    out = {c.tag: c for c in candidate_grid()}
    out["constant"] = CandidateConfig(2.0, 2.0, 1.0, kind="constant")
    return out


def _tag_float(x: float) -> str:
    return f"{float(x):g}".replace("-", "m").replace(".", "p")


def _motion_prior_tag(sigma_rows: float, tau_s: float) -> str:
    tau_ms = float(tau_s) * 1000.0
    return f"_mp_s{_tag_float(sigma_rows)}_tau{_tag_float(tau_ms)}"


def cache_tag_for_config(cfg: CandidateConfig, dur_s: float | None = None,
                         lag_ms: float = flt.HYPOTHESIS_LAG_MS,
                         transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS,
                         obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT,
                         blend_immediate: bool = flt.HYPOTHESIS_BLEND_IMMEDIATE,
                         blend_delta_rows: float = flt.HYPOTHESIS_BLEND_DELTA_ROWS,
                         blend_alpha: float = flt.HYPOTHESIS_BLEND_ALPHA,
                         blend_saccade_p: float = flt.HYPOTHESIS_BLEND_SACCADE_P,
                         slew_gate: bool = False,
                         slew_max_deg_s: float | None = None,
                         velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
                         acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
                         motion_prior: bool = False,
                         motion_prior_sigma_rows: float = 2.0,
                         motion_prior_tau_s: float = 0.003) -> str:
    tag = f"m4_dpf_physics_aq_{cfg.tag}"
    if float(lag_ms) != flt.HYPOTHESIS_LAG_MS:
        tag += f"_lag{_tag_float(lag_ms)}"
    if float(transition_sigma_rows) != flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS:
        tag += f"_ts{_tag_float(transition_sigma_rows)}"
    if float(obs_weight) != flt.HYPOTHESIS_OBS_WEIGHT:
        tag += f"_ow{_tag_float(obs_weight)}"
    if bool(blend_immediate):
        tag += f"_bi_d{_tag_float(blend_delta_rows)}_a{_tag_float(blend_alpha)}"
        if float(blend_saccade_p) != flt.HYPOTHESIS_BLEND_SACCADE_P:
            tag += f"_sp{_tag_float(blend_saccade_p)}"
    if slew_gate:
        max_deg = flt.SLEW_GATE_MAX_DEG_S if slew_max_deg_s is None else float(slew_max_deg_s)
        tag += f"_sg{_tag_float(max_deg)}"
    if float(velocity_cost) != flt.HYPOTHESIS_VEL_COST:
        tag += f"_vc{_tag_float(velocity_cost)}"
    if float(acceleration_cost) != flt.HYPOTHESIS_ACCEL_COST:
        tag += f"_ac{_tag_float(acceleration_cost)}"
    if motion_prior:
        tag += _motion_prior_tag(motion_prior_sigma_rows, motion_prior_tau_s)
    if dur_s is not None:
        tag += f"_d{int(dur_s)}"
    return tag


def variant_id_for_config(cfg: CandidateConfig, slew_gate: bool = False,
                          lag_ms: float = flt.HYPOTHESIS_LAG_MS,
                          transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS,
                          obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT,
                          blend_immediate: bool = flt.HYPOTHESIS_BLEND_IMMEDIATE,
                          blend_delta_rows: float = flt.HYPOTHESIS_BLEND_DELTA_ROWS,
                          blend_alpha: float = flt.HYPOTHESIS_BLEND_ALPHA,
                          blend_saccade_p: float = flt.HYPOTHESIS_BLEND_SACCADE_P,
                          slew_max_deg_s: float | None = None,
                          velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
                          acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
                          motion_prior: bool = False,
                          motion_prior_sigma_rows: float = 2.0,
                          motion_prior_tau_s: float = 0.003) -> str:
    tag = cfg.tag
    if float(lag_ms) != flt.HYPOTHESIS_LAG_MS:
        tag += f"_lag{_tag_float(lag_ms)}"
    if float(transition_sigma_rows) != flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS:
        tag += f"_ts{_tag_float(transition_sigma_rows)}"
    if float(obs_weight) != flt.HYPOTHESIS_OBS_WEIGHT:
        tag += f"_ow{_tag_float(obs_weight)}"
    if bool(blend_immediate):
        tag += f"_bi_d{_tag_float(blend_delta_rows)}_a{_tag_float(blend_alpha)}"
        if float(blend_saccade_p) != flt.HYPOTHESIS_BLEND_SACCADE_P:
            tag += f"_sp{_tag_float(blend_saccade_p)}"
    if slew_gate:
        max_deg = flt.SLEW_GATE_MAX_DEG_S if slew_max_deg_s is None else float(slew_max_deg_s)
        tag += f"_sg{_tag_float(max_deg)}"
    if float(velocity_cost) != flt.HYPOTHESIS_VEL_COST:
        tag += f"_vc{_tag_float(velocity_cost)}"
    if float(acceleration_cost) != flt.HYPOTHESIS_ACCEL_COST:
        tag += f"_ac{_tag_float(acceleration_cost)}"
    if motion_prior:
        tag += _motion_prior_tag(motion_prior_sigma_rows, motion_prior_tau_s)
    return tag


def _load_npz(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _clip_run(run: dict, dur_s: float | None) -> dict:
    if dur_s is None or "t" not in run or len(run["t"]) == 0:
        return run
    t = np.asarray(run["t"], dtype=np.float64)
    keep = t <= float(t[0]) + float(dur_s)
    out = {}
    for k, v in run.items():
        a = np.asarray(v)
        out[k] = a[keep] if a.shape[:1] == t.shape[:1] else v
    return out


def _percentile(x, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def raw_step_metrics(run: dict) -> dict[str, float]:
    valid = np.asarray(run["valid"], dtype=bool)
    x = np.asarray(run["x_px"], dtype=np.float64)
    rate = float(run["rate"])
    step_ok = valid[1:] & valid[:-1] & np.isfinite(x[1:]) & np.isfinite(x[:-1])
    step = np.abs(np.diff(x)[step_ok])
    if not step.size:
        return dict(
            raw_jump_ge3_frac=float("nan"),
            raw_step_p999_px=float("nan"),
            raw_speed_p999_px_s=float("nan"),
            raw_speed_p999_arcmin_s=float("nan"),
            raw_jump_return_ge3_frac=float("nan"),
        )
    ret = float("nan")
    if step.size >= 2:
        signed = np.diff(x)[step_ok]
        return_jump = (
            (np.abs(signed[:-1]) >= 3.0)
            & (np.abs(signed[1:]) >= 3.0)
            & (signed[:-1] * signed[1:] < 0.0)
        )
        ret = float(np.mean(return_jump))
    speed = step * rate
    return dict(
        raw_jump_ge3_frac=float(np.mean(step >= 3.0)),
        raw_step_p999_px=_percentile(step, 99.9),
        raw_speed_p999_px_s=_percentile(speed, 99.9),
        raw_speed_p999_arcmin_s=_percentile(speed * pf.ARC_PER_PX, 99.9),
        raw_jump_return_ge3_frac=ret,
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
    try:
        import oculo_smooth as osm
        ox, oy = osm.oculomotor_trajectory_2d(x, y, rate, valid)
        ov = valid & np.isfinite(ox) & np.isfinite(oy)
        oe = pf.evaluate(t, ox, oy, ov, rate, refs, smooth_ms=0.0)
        out.update(
            oculo_valid_frac=float(oe["valid_frac"]),
            oculo_r_dot_x=float(oe["r_dot_x"]),
            oculo_r_dot_y=float(oe["r_dot_y"]),
            oculo_prec_x=float(oe["prec_x"]),
            oculo_prec_y=float(oe["prec_y"]),
            oculo_j30=float(pf.frame_jitter_30fps(oe["cal_x"], rate, ov,
                                                  smooth_ms=0.0)),
        )
    except Exception as exc:  # pragma: no cover - defensive report path
        out.update(
            oculo_valid_frac=float("nan"),
            oculo_r_dot_x=float("nan"),
            oculo_r_dot_y=float("nan"),
            oculo_prec_x=float("nan"),
            oculo_prec_y=float("nan"),
            oculo_j30=float("nan"),
            oculo_error=type(exc).__name__,
        )
    return out


def _finite_float(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _frac_reduction(base: float, cand: float) -> float:
    base = _finite_float(base)
    cand = _finite_float(cand)
    if not np.isfinite(base) or not np.isfinite(cand):
        return float("nan")
    if abs(base) <= 1e-12:
        return 0.0 if cand <= base + 1e-12 else -1.0
    return float((base - cand) / abs(base))


def _gate_ge(cand: float, base: float, margin: float) -> bool:
    base = _finite_float(base)
    cand = _finite_float(cand)
    return (not np.isfinite(base)) or (np.isfinite(cand) and cand >= base - margin)


def _gate_le(cand: float, base: float, margin: float) -> bool:
    base = _finite_float(base)
    cand = _finite_float(cand)
    return (not np.isfinite(base)) or (np.isfinite(cand) and cand <= base + margin)


def compare_to_baseline(cand: dict, base: dict) -> dict[str, float | bool]:
    gates = dict(
        gate_r_dot_x=_gate_ge(cand.get("r_dot_x"), base.get("r_dot_x"), 0.02),
        gate_valid_frac=_gate_ge(cand.get("valid_frac"), base.get("valid_frac"), 0.03),
        gate_prec_x=_gate_le(cand.get("prec_x"), base.get("prec_x"), 0.10),
        gate_j30=_gate_le(cand.get("j30"), base.get("j30"), 0.05),
    )
    jump_red = _frac_reduction(base.get("raw_jump_ge3_frac"),
                               cand.get("raw_jump_ge3_frac"))
    speed_red = _frac_reduction(base.get("raw_speed_p999_px_s"),
                                cand.get("raw_speed_p999_px_s"))
    return_red = _frac_reduction(base.get("raw_jump_return_ge3_frac"),
                                 cand.get("raw_jump_return_ge3_frac"))
    oculo_j30_red = _frac_reduction(base.get("oculo_j30"), cand.get("oculo_j30"))
    score_parts = [jump_red, speed_red, 0.25 * oculo_j30_red]
    score = float(np.nansum(score_parts))
    if not all(gates.values()):
        score -= 10.0
    out: dict[str, float | bool] = dict(gates)
    out.update(
        pass_hard_gates=all(gates.values()),
        raw_jump_reduction=jump_red,
        raw_speed_p999_reduction=speed_red,
        raw_jump_return_reduction=return_red,
        oculo_j30_reduction=oculo_j30_red,
        score=score,
        delta_valid_frac=_finite_float(cand.get("valid_frac")) - _finite_float(base.get("valid_frac")),
        delta_r_dot_x=_finite_float(cand.get("r_dot_x")) - _finite_float(base.get("r_dot_x")),
        delta_prec_x=_finite_float(cand.get("prec_x")) - _finite_float(base.get("prec_x")),
        delta_j30=_finite_float(cand.get("j30")) - _finite_float(base.get("j30")),
    )
    return out


def make_result_row(stage: str, subject: str, cfg: CandidateConfig,
                    model: AlongQualityModel, cand_metrics: dict,
                    base_metrics: dict, cache_path: str,
                    baseline_path: str, dur_s: float | None,
                    variant_id: str | None = None) -> dict:
    row = dict(
        stage=stage,
        subject=subject,
        config_id=cfg.tag,
        variant_id=variant_id or cfg.tag,
        aq_model=cfg.kind,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
        gamma=cfg.gamma,
        q_p10=model.q_p10,
        q_p90=model.q_p90,
        dur_s="" if dur_s is None else float(dur_s),
        cache_path=cache_path,
        baseline_path=baseline_path,
    )
    for k, v in cand_metrics.items():
        row[k] = v
    for k in KEY_METRICS:
        row[f"baseline_{k}"] = base_metrics.get(k, float("nan"))
    row.update(compare_to_baseline(cand_metrics, base_metrics))
    return row


def run_subject_config(sub: pf.Subject, cfg: CandidateConfig, *,
                       dur_s: float | None, stage: str, rebuild: bool,
                       rebuild_inputs: bool, top_k: int, n_particles: int,
                       lag_ms: float = flt.HYPOTHESIS_LAG_MS,
                       hypothesis_transition_sigma_rows: float = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS,
                       hypothesis_obs_weight: float = flt.HYPOTHESIS_OBS_WEIGHT,
                       hypothesis_blend_immediate: bool = flt.HYPOTHESIS_BLEND_IMMEDIATE,
                       hypothesis_blend_delta_rows: float = flt.HYPOTHESIS_BLEND_DELTA_ROWS,
                       hypothesis_blend_alpha: float = flt.HYPOTHESIS_BLEND_ALPHA,
                       hypothesis_blend_saccade_p: float = flt.HYPOTHESIS_BLEND_SACCADE_P,
                       slew_gate: bool = False,
                       slew_max_deg_s: float = flt.SLEW_GATE_MAX_DEG_S,
                       slew_gate_cost: float = flt.SLEW_GATE_COST,
                       slew_gate_saccade_p: float = flt.SLEW_GATE_SACCADE_P,
                       hypothesis_velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
                       hypothesis_velocity_sigma_deg_s: float = flt.HYPOTHESIS_VEL_SIGMA_DEG_S,
                       hypothesis_acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
                       hypothesis_acceleration_sigma_deg_s2: float = flt.HYPOTHESIS_ACCEL_SIGMA_DEG_S2,
                       motion_prior: bool = False,
                       motion_prior_sigma_rows: float = 2.0,
                       motion_prior_tau_s: float = 0.003,
                       motion_prior_ncc_thr: float = 0.2) -> dict:
    ch = pf.build_chain(sub, rebuild=rebuild_inputs)
    lm = pf.build_line_measurements(sub, rebuild=rebuild_inputs)
    refs = pf.compute_refs(sub, lm)
    baseline_path = os.path.join(sub.cache_dir, "m4_dpf_physics.npz")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"{baseline_path} is required; build the canonical baseline first")
    base_run = _clip_run(_load_npz(baseline_path), dur_s)
    base_metrics = collect_metrics(base_run, refs)

    model = cfg.fit_model(lm["qv"])
    cache_tag = cache_tag_for_config(cfg, dur_s, slew_gate=slew_gate,
                                     lag_ms=lag_ms,
                                     transition_sigma_rows=hypothesis_transition_sigma_rows,
                                     obs_weight=hypothesis_obs_weight,
                                     blend_immediate=hypothesis_blend_immediate,
                                     blend_delta_rows=hypothesis_blend_delta_rows,
                                     blend_alpha=hypothesis_blend_alpha,
                                     blend_saccade_p=hypothesis_blend_saccade_p,
                                     slew_max_deg_s=slew_max_deg_s,
                                     velocity_cost=hypothesis_velocity_cost,
                                     acceleration_cost=hypothesis_acceleration_cost,
                                     motion_prior=motion_prior,
                                     motion_prior_sigma_rows=motion_prior_sigma_rows,
                                     motion_prior_tau_s=motion_prior_tau_s)
    cache_path = os.path.join(sub.cache_dir, f"{cache_tag}.npz")
    cand_run = pf.run_m4(
        sub, lm, ch, cache_path=cache_path, dur_s=dur_s, rebuild=rebuild,
        n_particles=n_particles, quality_scaled_along=True,
        along_quality_model=model, multi_hypothesis=True,
        hypothesis_top_k=top_k, slew_gate=slew_gate,
        lag_ms=lag_ms,
        hypothesis_transition_sigma_rows=hypothesis_transition_sigma_rows,
        hypothesis_obs_weight=hypothesis_obs_weight,
        hypothesis_blend_immediate=hypothesis_blend_immediate,
        hypothesis_blend_delta_rows=hypothesis_blend_delta_rows,
        hypothesis_blend_alpha=hypothesis_blend_alpha,
        hypothesis_blend_saccade_p=hypothesis_blend_saccade_p,
        slew_max_deg_s=slew_max_deg_s, slew_gate_cost=slew_gate_cost,
        slew_gate_saccade_p=slew_gate_saccade_p,
        hypothesis_velocity_cost=hypothesis_velocity_cost,
        hypothesis_velocity_sigma_deg_s=hypothesis_velocity_sigma_deg_s,
        hypothesis_acceleration_cost=hypothesis_acceleration_cost,
        hypothesis_acceleration_sigma_deg_s2=hypothesis_acceleration_sigma_deg_s2,
        mosaic_prior=motion_prior,
        mosaic_prior_sigma_rows=motion_prior_sigma_rows,
        mosaic_prior_ncc_thr=motion_prior_ncc_thr,
        mosaic_prior_track_tau_s=motion_prior_tau_s)
    cand_metrics = collect_metrics(cand_run, refs)
    row = make_result_row(
        stage, sub.name, cfg, model, cand_metrics,
        base_metrics, cache_path, baseline_path, dur_s,
        variant_id=variant_id_for_config(
            cfg, slew_gate=slew_gate, slew_max_deg_s=slew_max_deg_s,
            lag_ms=lag_ms,
            transition_sigma_rows=hypothesis_transition_sigma_rows,
            obs_weight=hypothesis_obs_weight,
            blend_immediate=hypothesis_blend_immediate,
            blend_delta_rows=hypothesis_blend_delta_rows,
            blend_alpha=hypothesis_blend_alpha,
            blend_saccade_p=hypothesis_blend_saccade_p,
            velocity_cost=hypothesis_velocity_cost,
            acceleration_cost=hypothesis_acceleration_cost,
            motion_prior=motion_prior,
            motion_prior_sigma_rows=motion_prior_sigma_rows,
            motion_prior_tau_s=motion_prior_tau_s))
    row.update(
        slew_gate=bool(slew_gate),
        lag_ms=float(lag_ms),
        hypothesis_transition_sigma_rows=float(hypothesis_transition_sigma_rows),
        hypothesis_obs_weight=float(hypothesis_obs_weight),
        hypothesis_blend_immediate=bool(hypothesis_blend_immediate),
        hypothesis_blend_delta_rows=float(hypothesis_blend_delta_rows) if hypothesis_blend_immediate else "",
        hypothesis_blend_alpha=float(hypothesis_blend_alpha) if hypothesis_blend_immediate else "",
        hypothesis_blend_saccade_p=float(hypothesis_blend_saccade_p) if hypothesis_blend_immediate else "",
        slew_max_deg_s=float(slew_max_deg_s) if slew_gate else "",
        slew_gate_cost=float(slew_gate_cost) if slew_gate else "",
        slew_gate_saccade_p=float(slew_gate_saccade_p) if slew_gate else "",
        hypothesis_velocity_cost=float(hypothesis_velocity_cost),
        hypothesis_velocity_sigma_deg_s=float(hypothesis_velocity_sigma_deg_s),
        hypothesis_acceleration_cost=float(hypothesis_acceleration_cost),
        hypothesis_acceleration_sigma_deg_s2=float(hypothesis_acceleration_sigma_deg_s2),
        motion_prior=bool(motion_prior),
        motion_prior_sigma_rows=float(motion_prior_sigma_rows) if motion_prior else "",
        motion_prior_tau_ms=float(motion_prior_tau_s) * 1000.0 if motion_prior else "",
        motion_prior_ncc_thr=float(motion_prior_ncc_thr) if motion_prior else "",
    )
    return row


def _as_float(row: dict, key: str) -> float:
    return _finite_float(row.get(key))


def _median(rows: list[dict], key: str) -> float:
    vals = np.asarray([_as_float(r, key) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def summarize_configs(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        variant_id = str(row.get("variant_id") or row.get("config_id", ""))
        groups.setdefault((str(row.get("stage", "")), variant_id), []).append(row)
    out = []
    for (stage, variant_id), rr in sorted(groups.items()):
        cfg_id = str(rr[0].get("config_id", variant_id))
        out.append(dict(
            stage=stage,
            config_id=cfg_id,
            variant_id=variant_id,
            lag_ms=rr[0].get("lag_ms", ""),
            hypothesis_transition_sigma_rows=rr[0].get("hypothesis_transition_sigma_rows", ""),
            hypothesis_obs_weight=rr[0].get("hypothesis_obs_weight", ""),
            hypothesis_blend_immediate=rr[0].get("hypothesis_blend_immediate", ""),
            hypothesis_blend_delta_rows=rr[0].get("hypothesis_blend_delta_rows", ""),
            hypothesis_blend_alpha=rr[0].get("hypothesis_blend_alpha", ""),
            hypothesis_blend_saccade_p=rr[0].get("hypothesis_blend_saccade_p", ""),
            slew_gate=rr[0].get("slew_gate", ""),
            slew_max_deg_s=rr[0].get("slew_max_deg_s", ""),
            hypothesis_velocity_cost=rr[0].get("hypothesis_velocity_cost", ""),
            hypothesis_velocity_sigma_deg_s=rr[0].get("hypothesis_velocity_sigma_deg_s", ""),
            hypothesis_acceleration_cost=rr[0].get("hypothesis_acceleration_cost", ""),
            hypothesis_acceleration_sigma_deg_s2=rr[0].get("hypothesis_acceleration_sigma_deg_s2", ""),
            motion_prior=rr[0].get("motion_prior", ""),
            motion_prior_sigma_rows=rr[0].get("motion_prior_sigma_rows", ""),
            motion_prior_tau_ms=rr[0].get("motion_prior_tau_ms", ""),
            motion_prior_ncc_thr=rr[0].get("motion_prior_ncc_thr", ""),
            n_subjects=len({r.get("subject") for r in rr}),
            pass_all=all(str(r.get("pass_hard_gates")) in {"True", "true", "1"} for r in rr),
            median_score=_median(rr, "score"),
            median_raw_jump_reduction=_median(rr, "raw_jump_reduction"),
            median_raw_speed_p999_reduction=_median(rr, "raw_speed_p999_reduction"),
            median_raw_jump_return_reduction=_median(rr, "raw_jump_return_reduction"),
            median_delta_prec_x=_median(rr, "delta_prec_x"),
            median_delta_j30=_median(rr, "delta_j30"),
        ))
    out.sort(key=lambda r: (
        str(r["stage"]),
        0 if r["pass_all"] else 1,
        -_finite_float(r["median_score"]),
        -_finite_float(r["median_raw_jump_reduction"]),
    ))
    return out


def rank_variant_ids(rows: list[dict], top_k: int) -> list[str]:
    return [str(r["variant_id"]) for r in summarize_configs(rows)[:top_k]]


def rank_config_ids(rows: list[dict], top_k: int) -> list[str]:
    return [str(r["config_id"]) for r in summarize_configs(rows)[:top_k]]


def leave_one_subject_out(rows: list[dict]) -> list[dict]:
    subjects = sorted({str(r.get("subject")) for r in rows})
    variants = sorted({str(r.get("variant_id") or r.get("config_id")) for r in rows})
    if len(subjects) < 2 or len(variants) < 2:
        return []
    out = []
    for holdout in subjects:
        train = [r for r in rows if str(r.get("subject")) != holdout]
        best = rank_variant_ids(train, 1)
        if not best:
            continue
        held = [r for r in rows
                if str(r.get("subject")) == holdout
                and str(r.get("variant_id") or r.get("config_id")) == best[0]]
        if not held:
            continue
        h = held[0]
        out.append(dict(
            holdout=holdout,
            selected_config=best[0],
            pass_hard_gates=h.get("pass_hard_gates"),
            raw_jump_reduction=h.get("raw_jump_reduction"),
            raw_speed_p999_reduction=h.get("raw_speed_p999_reduction"),
            delta_prec_x=h.get("delta_prec_x"),
            delta_j30=h.get("delta_j30"),
        ))
    return out


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
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
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    return lines


def write_reports(rows: list[dict], out_prefix: str = RESULT_PREFIX) -> tuple[str, str]:
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    csv_path = f"{out_prefix}.csv"
    md_path = f"{out_prefix}.md"
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    summaries = summarize_configs(rows)
    loso = leave_one_subject_out(rows)
    md = [
        "# Along-Quality Calibration",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Hard gates per subject: r_dot_x >= baseline - 0.02; "
        "valid_frac >= baseline - 0.03; prec_x <= baseline + 0.10; "
        "j30 <= baseline + 0.05.",
        "",
        "Primary improvement targets: median raw >=3 px jump fraction and "
        "median p99.9 raw line-step speed should each fall by at least 10%.",
        "",
        "## Config Summary",
        "",
    ]
    md.extend(_md_table(
        summaries,
        ["stage", "variant_id", "config_id", "slew_gate", "slew_max_deg_s",
         "lag_ms", "hypothesis_transition_sigma_rows",
         "hypothesis_obs_weight",
         "hypothesis_blend_immediate", "hypothesis_blend_delta_rows",
         "hypothesis_blend_alpha",
         "hypothesis_velocity_cost", "hypothesis_acceleration_cost",
         "motion_prior", "motion_prior_sigma_rows", "motion_prior_tau_ms",
         "motion_prior_ncc_thr",
         "n_subjects", "pass_all", "median_score",
         "median_raw_jump_reduction", "median_raw_speed_p999_reduction",
         "median_raw_jump_return_reduction",
         "median_delta_prec_x", "median_delta_j30"],
    ))
    md.extend(["", "## Subject Rows", ""])
    md.extend(_md_table(
        rows,
        ["stage", "subject", "variant_id", "config_id", "slew_gate", "slew_max_deg_s",
         "lag_ms", "hypothesis_transition_sigma_rows",
         "hypothesis_obs_weight",
         "hypothesis_blend_immediate", "hypothesis_blend_delta_rows",
         "hypothesis_blend_alpha",
         "hypothesis_velocity_cost", "hypothesis_acceleration_cost",
         "motion_prior", "motion_prior_sigma_rows", "motion_prior_tau_ms",
         "motion_prior_ncc_thr",
         "pass_hard_gates",
         "raw_jump_reduction", "raw_speed_p999_reduction",
         "raw_jump_return_reduction",
         "delta_r_dot_x", "delta_valid_frac", "delta_prec_x", "delta_j30",
         "oculo_j30_reduction"],
    ))
    if loso:
        md.extend(["", "## Leave-One-Subject-Out", ""])
        md.extend(_md_table(
            loso,
            ["holdout", "selected_config", "pass_hard_gates",
             "raw_jump_reduction", "raw_speed_p999_reduction",
             "delta_prec_x", "delta_j30"],
        ))
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    return csv_path, md_path


def _read_csv_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def select_finalist_configs(path: str, top_k: int) -> list[CandidateConfig]:
    rows = _read_csv_rows(path)
    wanted = rank_config_ids(rows, top_k)
    lookup = config_lookup()
    missing = [c for c in wanted if c not in lookup]
    if missing:
        raise ValueError(f"unknown config id(s) in finalist CSV: {missing}")
    return [lookup[c] for c in wanted]


def _row_bool(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def variant_from_row(row: dict) -> RunVariant:
    """Reconstruct a runnable variant from one calibration CSV row."""
    lookup = config_lookup()
    cfg_id = str(row.get("config_id", ""))
    if cfg_id not in lookup:
        raise ValueError(f"unknown config id in row: {cfg_id!r}")
    lag_ms = _finite_float(row.get("lag_ms"))
    if not np.isfinite(lag_ms):
        lag_ms = flt.HYPOTHESIS_LAG_MS
    transition_sigma = _finite_float(row.get("hypothesis_transition_sigma_rows"))
    if not np.isfinite(transition_sigma):
        transition_sigma = flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS
    obs_weight = _finite_float(row.get("hypothesis_obs_weight"))
    if not np.isfinite(obs_weight):
        obs_weight = flt.HYPOTHESIS_OBS_WEIGHT
    blend_immediate = _row_bool(row.get("hypothesis_blend_immediate", False))
    blend_delta = _finite_float(row.get("hypothesis_blend_delta_rows"))
    if not np.isfinite(blend_delta):
        blend_delta = flt.HYPOTHESIS_BLEND_DELTA_ROWS
    blend_alpha = _finite_float(row.get("hypothesis_blend_alpha"))
    if not np.isfinite(blend_alpha):
        blend_alpha = flt.HYPOTHESIS_BLEND_ALPHA
    blend_saccade = _finite_float(row.get("hypothesis_blend_saccade_p"))
    if not np.isfinite(blend_saccade):
        blend_saccade = flt.HYPOTHESIS_BLEND_SACCADE_P
    slew_gate = _row_bool(row.get("slew_gate", False))
    slew_max = _finite_float(row.get("slew_max_deg_s"))
    if not np.isfinite(slew_max):
        slew_max = flt.SLEW_GATE_MAX_DEG_S
    vel_cost = _finite_float(row.get("hypothesis_velocity_cost"))
    if not np.isfinite(vel_cost):
        vel_cost = flt.HYPOTHESIS_VEL_COST
    vel_sigma = _finite_float(row.get("hypothesis_velocity_sigma_deg_s"))
    if not np.isfinite(vel_sigma):
        vel_sigma = flt.HYPOTHESIS_VEL_SIGMA_DEG_S
    accel_cost = _finite_float(row.get("hypothesis_acceleration_cost"))
    if not np.isfinite(accel_cost):
        accel_cost = flt.HYPOTHESIS_ACCEL_COST
    accel_sigma = _finite_float(row.get("hypothesis_acceleration_sigma_deg_s2"))
    if not np.isfinite(accel_sigma):
        accel_sigma = flt.HYPOTHESIS_ACCEL_SIGMA_DEG_S2
    motion_prior = _row_bool(row.get("motion_prior", False))
    motion_sigma = _finite_float(row.get("motion_prior_sigma_rows"))
    if not np.isfinite(motion_sigma):
        motion_sigma = 2.0
    motion_tau_ms = _finite_float(row.get("motion_prior_tau_ms"))
    if not np.isfinite(motion_tau_ms):
        motion_tau_ms = 3.0
    motion_ncc = _finite_float(row.get("motion_prior_ncc_thr"))
    if not np.isfinite(motion_ncc):
        motion_ncc = 0.2
    return RunVariant(
        lookup[cfg_id],
        lag_ms=float(lag_ms),
        transition_sigma_rows=float(transition_sigma),
        obs_weight=float(obs_weight),
        blend_immediate=blend_immediate,
        blend_delta_rows=float(blend_delta),
        blend_alpha=float(blend_alpha),
        blend_saccade_p=float(blend_saccade),
        slew_gate=slew_gate,
        slew_max_deg_s=float(slew_max),
        velocity_cost=float(vel_cost),
        velocity_sigma_deg_s=float(vel_sigma),
        acceleration_cost=float(accel_cost),
        acceleration_sigma_deg_s2=float(accel_sigma),
        motion_prior=motion_prior,
        motion_prior_sigma_rows=float(motion_sigma),
        motion_prior_tau_s=float(motion_tau_ms) / 1000.0,
        motion_prior_ncc_thr=float(motion_ncc),
    )


def select_finalist_variants(path: str, top_k: int) -> list[RunVariant]:
    rows = _read_csv_rows(path)
    summaries = summarize_configs(rows)[:top_k]
    out: list[RunVariant] = []
    for row in summaries:
        out.append(variant_from_row(row))
    return out


def _policy_score_tuple(row: dict) -> tuple:
    """Higher is better for selecting one variant per subject."""
    return (
        1 if _row_bool(row.get("pass_hard_gates", False)) else 0,
        _finite_float(row.get("score")),
        _finite_float(row.get("raw_jump_reduction")),
        _finite_float(row.get("raw_speed_p999_reduction")),
        -abs(_finite_float(row.get("delta_j30"))),
        -abs(_finite_float(row.get("delta_prec_x"))),
    )


def select_subject_policy_rows(rows: list[dict]) -> list[dict]:
    """Select the best calibration row independently for each subject."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        subject = str(row.get("subject", "")).strip()
        if subject:
            groups.setdefault(subject, []).append(row)
    selected = []
    for subject in sorted(groups):
        selected.append(max(groups[subject], key=_policy_score_tuple))
    return selected


def _policy_variant_dict(variant: RunVariant) -> dict:
    return dict(
        config_id=variant.cfg.tag,
        variant_id=variant.variant_id,
        lag_ms=variant.lag_ms,
        transition_sigma_rows=variant.transition_sigma_rows,
        obs_weight=variant.obs_weight,
        blend_immediate=variant.blend_immediate,
        blend_delta_rows=variant.blend_delta_rows,
        blend_alpha=variant.blend_alpha,
        blend_saccade_p=variant.blend_saccade_p,
        slew_gate=variant.slew_gate,
        slew_max_deg_s=variant.slew_max_deg_s,
        velocity_cost=variant.velocity_cost,
        velocity_sigma_deg_s=variant.velocity_sigma_deg_s,
        acceleration_cost=variant.acceleration_cost,
        acceleration_sigma_deg_s2=variant.acceleration_sigma_deg_s2,
        motion_prior=variant.motion_prior,
        motion_prior_sigma_rows=variant.motion_prior_sigma_rows,
        motion_prior_tau_s=variant.motion_prior_tau_s,
        motion_prior_ncc_thr=variant.motion_prior_ncc_thr,
    )


def write_subject_policy(rows: list[dict], out_prefix: str = POLICY_PREFIX,
                         source_paths: list[str] | None = None) -> tuple[str, str]:
    """Write an auditable per-subject calibration policy JSON and Markdown."""
    selected = select_subject_policy_rows(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    json_path = f"{out_prefix}.json"
    md_path = f"{out_prefix}.md"
    subjects = {}
    for row in selected:
        variant = variant_from_row(row)
        subjects[str(row["subject"])] = dict(
            variant=_policy_variant_dict(variant),
            source_row={k: row.get(k, "") for k in (
                "stage", "subject", "variant_id", "config_id", "cache_path",
                "lag_ms", "hypothesis_transition_sigma_rows",
                "hypothesis_obs_weight",
                "hypothesis_blend_immediate", "hypothesis_blend_delta_rows",
                "hypothesis_blend_alpha", "hypothesis_blend_saccade_p",
                "pass_hard_gates", "raw_jump_reduction",
                "raw_speed_p999_reduction", "raw_jump_return_reduction",
                "delta_r_dot_x",
                "delta_valid_frac", "delta_prec_x", "delta_j30",
                "oculo_j30_reduction",
            )},
        )
    payload = dict(
        generated=datetime.now().isoformat(timespec="seconds"),
        source_paths=source_paths or [],
        selection=(
            "Per subject: prefer hard-gate pass, then score, raw jump reduction, "
            "p99.9 speed reduction, small |delta_j30|, small |delta_prec_x|."
        ),
        subjects=subjects,
    )
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    md = [
        "# Along-Quality Subject Policy",
        "",
        f"Generated: {payload['generated']}",
        "",
        payload["selection"],
        "",
        "## Selected Variants",
        "",
    ]
    md.extend(_md_table(
        [{**subjects[s]["variant"], **subjects[s]["source_row"]}
         for s in sorted(subjects)],
        ["subject", "variant_id", "pass_hard_gates", "raw_jump_reduction",
         "raw_speed_p999_reduction", "raw_jump_return_reduction",
         "delta_r_dot_x", "delta_valid_frac",
         "delta_prec_x", "delta_j30", "cache_path"],
    ))
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    return json_path, md_path


def load_subject_policy(path: str) -> dict[str, RunVariant]:
    with open(path) as f:
        payload = json.load(f)
    lookup = config_lookup()
    out: dict[str, RunVariant] = {}
    for subject, spec in payload.get("subjects", {}).items():
        v = spec.get("variant", spec)
        cfg_id = str(v.get("config_id", ""))
        if cfg_id not in lookup:
            raise ValueError(f"unknown config id in policy for {subject}: {cfg_id!r}")
        out[str(subject)] = RunVariant(
            lookup[cfg_id],
            lag_ms=float(v.get("lag_ms", flt.HYPOTHESIS_LAG_MS)),
            transition_sigma_rows=float(v.get(
                "transition_sigma_rows", flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS)),
            obs_weight=float(v.get("obs_weight", flt.HYPOTHESIS_OBS_WEIGHT)),
            blend_immediate=bool(v.get("blend_immediate", flt.HYPOTHESIS_BLEND_IMMEDIATE)),
            blend_delta_rows=float(v.get(
                "blend_delta_rows", flt.HYPOTHESIS_BLEND_DELTA_ROWS)),
            blend_alpha=float(v.get("blend_alpha", flt.HYPOTHESIS_BLEND_ALPHA)),
            blend_saccade_p=float(v.get(
                "blend_saccade_p", flt.HYPOTHESIS_BLEND_SACCADE_P)),
            slew_gate=bool(v.get("slew_gate", False)),
            slew_max_deg_s=float(v.get("slew_max_deg_s", flt.SLEW_GATE_MAX_DEG_S)),
            velocity_cost=float(v.get("velocity_cost", flt.HYPOTHESIS_VEL_COST)),
            velocity_sigma_deg_s=float(v.get("velocity_sigma_deg_s",
                                             flt.HYPOTHESIS_VEL_SIGMA_DEG_S)),
            acceleration_cost=float(v.get("acceleration_cost",
                                          flt.HYPOTHESIS_ACCEL_COST)),
            acceleration_sigma_deg_s2=float(v.get(
                "acceleration_sigma_deg_s2", flt.HYPOTHESIS_ACCEL_SIGMA_DEG_S2)),
            motion_prior=bool(v.get("motion_prior", False)),
            motion_prior_sigma_rows=float(v.get("motion_prior_sigma_rows", 2.0)),
            motion_prior_tau_s=float(v.get("motion_prior_tau_s", 0.003)),
            motion_prior_ncc_thr=float(v.get("motion_prior_ncc_thr", 0.2)),
        )
    return out


def _parse_subjects(arg: str | None) -> list[pf.Subject]:
    subs = pf.discover_subjects()
    if not arg:
        return subs
    wanted = {x.strip() for x in arg.split(",") if x.strip()}
    out = [s for s in subs if s.name in wanted]
    missing = sorted(wanted - {s.name for s in out})
    if missing:
        raise SystemExit(f"unknown subject(s): {missing}")
    return out


def _parse_configs(arg: str | None, max_configs: int | None) -> list[CandidateConfig]:
    if not arg:
        return candidate_grid(max_configs)
    lookup = config_lookup()
    out = []
    for item in arg.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in lookup:
            raise SystemExit(f"unknown config {key!r}")
        out.append(lookup[key])
    return out[:max_configs] if max_configs is not None else out


def _parse_slew_grid(slew_gate: bool, slew_max_deg_s: float,
                     slew_max_grid: str | None) -> list[tuple[bool, float]]:
    if not slew_max_grid:
        return [(bool(slew_gate), float(slew_max_deg_s))]
    out = []
    for item in slew_max_grid.split(","):
        item = item.strip()
        if item:
            out.append((True, float(item)))
    if not out:
        raise SystemExit("--slew-max-grid did not contain any numeric thresholds")
    return out


def _parse_float_grid(default: float, grid: str | None) -> list[float]:
    if not grid:
        return [float(default)]
    out = []
    for item in grid.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise SystemExit("resolver grid did not contain any numeric values")
    return out


def _parse_motion_prior_grid(enabled: bool, sigma_rows: float, tau_ms: float,
                             grid: str | None) -> list[tuple[bool, float, float]]:
    if not grid:
        return [(bool(enabled), float(sigma_rows), float(tau_ms) / 1000.0)]
    out = []
    for item in grid.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            sigma_s, tau_s = item.split(":", 1)
        else:
            sigma_s, tau_s = item, str(tau_ms)
        out.append((True, float(sigma_s), float(tau_s) / 1000.0))
    if not out:
        raise SystemExit("--motion-prior-grid did not contain any entries")
    return out


def _make_variants(configs: list[CandidateConfig],
                   slew_specs: list[tuple[bool, float]],
                   lag_specs: list[float] | None = None,
                   transition_sigma_specs: list[float] | None = None,
                   obs_weight_specs: list[float] | None = None,
                   blend_immediate: bool = flt.HYPOTHESIS_BLEND_IMMEDIATE,
                   blend_delta_rows: float = flt.HYPOTHESIS_BLEND_DELTA_ROWS,
                   blend_alpha: float = flt.HYPOTHESIS_BLEND_ALPHA,
                   blend_saccade_p: float = flt.HYPOTHESIS_BLEND_SACCADE_P,
                   velocity_cost: float = flt.HYPOTHESIS_VEL_COST,
                   velocity_sigma_deg_s: float = flt.HYPOTHESIS_VEL_SIGMA_DEG_S,
                   acceleration_cost: float = flt.HYPOTHESIS_ACCEL_COST,
                   acceleration_sigma_deg_s2: float = flt.HYPOTHESIS_ACCEL_SIGMA_DEG_S2,
                   motion_specs: list[tuple[bool, float, float]] | None = None,
                   motion_prior_ncc_thr: float = 0.2) -> list[RunVariant]:
    if motion_specs is None:
        motion_specs = [(False, 2.0, 0.003)]
    if lag_specs is None:
        lag_specs = [flt.HYPOTHESIS_LAG_MS]
    if transition_sigma_specs is None:
        transition_sigma_specs = [flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS]
    if obs_weight_specs is None:
        obs_weight_specs = [flt.HYPOTHESIS_OBS_WEIGHT]
    return [
        RunVariant(cfg, lag_ms=lag_ms,
                   transition_sigma_rows=transition_sigma,
                   obs_weight=obs_weight,
                   blend_immediate=blend_immediate,
                   blend_delta_rows=blend_delta_rows,
                   blend_alpha=blend_alpha,
                   blend_saccade_p=blend_saccade_p,
                   slew_gate=sg, slew_max_deg_s=smax,
                   velocity_cost=velocity_cost,
                   velocity_sigma_deg_s=velocity_sigma_deg_s,
                   acceleration_cost=acceleration_cost,
                   acceleration_sigma_deg_s2=acceleration_sigma_deg_s2,
                   motion_prior=mp,
                   motion_prior_sigma_rows=msig,
                   motion_prior_tau_s=mtau,
                   motion_prior_ncc_thr=motion_prior_ncc_thr)
        for cfg in configs
        for lag_ms in lag_specs
        for transition_sigma in transition_sigma_specs
        for obs_weight in obs_weight_specs
        for sg, smax in slew_specs
        for mp, msig, mtau in motion_specs
    ]


def _dry_rows(subjects: Iterable[str], variants: list[RunVariant],
              stage: str, dur_s: float | None) -> list[dict]:
    rows = []
    for si, subject in enumerate(subjects):
        base = dict(
            valid_frac=0.94, r_dot_x=0.93, r_dot_y=0.80,
            prec_x=2.5, j30=0.9, raw_jump_ge3_frac=0.02,
            raw_step_p999_px=40.0, raw_speed_p999_px_s=6e5,
            raw_speed_p999_arcmin_s=6e5 * pf.ARC_PER_PX,
            raw_jump_return_ge3_frac=0.01,
            oculo_r_dot_x=0.93, oculo_prec_x=2.3, oculo_j30=0.6,
        )
        for ci, variant in enumerate(variants):
            cfg = variant.cfg
            factor = 1.0 - 0.02 * (ci + 1)
            cand = dict(base)
            cand["raw_jump_ge3_frac"] = base["raw_jump_ge3_frac"] * factor
            cand["raw_speed_p999_px_s"] = base["raw_speed_p999_px_s"] * factor
            cand["raw_speed_p999_arcmin_s"] = base["raw_speed_p999_arcmin_s"] * factor
            cand["raw_jump_return_ge3_frac"] = base["raw_jump_return_ge3_frac"] * factor
            cand["prec_x"] = base["prec_x"] + 0.01 * si
            model = (AlongQualityModel.constant() if cfg.kind == "constant"
                     else AlongQualityModel.qv_power(
                         cfg.sigma_min, cfg.sigma_max, cfg.gamma, 0.2, 0.8))
            row = make_result_row(
                stage, subject, cfg, model, cand, base,
                f"dry/{cache_tag_for_config(cfg, dur_s, lag_ms=variant.lag_ms, transition_sigma_rows=variant.transition_sigma_rows, obs_weight=variant.obs_weight, blend_immediate=variant.blend_immediate, blend_delta_rows=variant.blend_delta_rows, blend_alpha=variant.blend_alpha, blend_saccade_p=variant.blend_saccade_p, slew_gate=variant.slew_gate, slew_max_deg_s=variant.slew_max_deg_s, velocity_cost=variant.velocity_cost, acceleration_cost=variant.acceleration_cost, motion_prior=variant.motion_prior, motion_prior_sigma_rows=variant.motion_prior_sigma_rows, motion_prior_tau_s=variant.motion_prior_tau_s)}.npz",
                "dry/m4_dpf_physics.npz", dur_s,
                variant_id=variant.variant_id)
            row.update(lag_ms=float(variant.lag_ms),
                       hypothesis_transition_sigma_rows=float(variant.transition_sigma_rows),
                       hypothesis_obs_weight=float(variant.obs_weight),
                       hypothesis_blend_immediate=bool(variant.blend_immediate),
                       hypothesis_blend_delta_rows=float(variant.blend_delta_rows) if variant.blend_immediate else "",
                       hypothesis_blend_alpha=float(variant.blend_alpha) if variant.blend_immediate else "",
                       hypothesis_blend_saccade_p=float(variant.blend_saccade_p) if variant.blend_immediate else "",
                       slew_gate=bool(variant.slew_gate),
                       slew_max_deg_s=float(variant.slew_max_deg_s) if variant.slew_gate else "",
                       hypothesis_velocity_cost=float(variant.velocity_cost),
                       hypothesis_velocity_sigma_deg_s=float(variant.velocity_sigma_deg_s),
                       hypothesis_acceleration_cost=float(variant.acceleration_cost),
                       hypothesis_acceleration_sigma_deg_s2=float(variant.acceleration_sigma_deg_s2),
                       motion_prior=bool(variant.motion_prior),
                       motion_prior_sigma_rows=float(variant.motion_prior_sigma_rows) if variant.motion_prior else "",
                       motion_prior_tau_ms=float(variant.motion_prior_tau_s) * 1000.0 if variant.motion_prior else "",
                       motion_prior_ncc_thr=float(variant.motion_prior_ncc_thr) if variant.motion_prior else "")
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("prune", "full"), default="prune")
    ap.add_argument("--dur", type=float, default=None,
                    help="duration cap in seconds; default is 20 for prune, full for full")
    ap.add_argument("--subjects", default=None,
                    help="comma-separated subject names; default is all people-data captures")
    ap.add_argument("--configs", default=None,
                    help="comma-separated config ids; default is the full grid")
    ap.add_argument("--finalists-from", default=f"{RESULT_PREFIX}.csv")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-configs", type=int, default=None)
    ap.add_argument("--n-particles", type=int, default=300)
    ap.add_argument("--lag-ms", type=float, default=flt.HYPOTHESIS_LAG_MS)
    ap.add_argument("--lag-ms-grid", default=None,
                    help="comma-separated fixed-lag resolver lags in ms")
    ap.add_argument("--hypothesis-transition-sigma-rows", type=float,
                    default=flt.HYPOTHESIS_TRANSITION_SIGMA_ROWS)
    ap.add_argument("--hypothesis-transition-sigma-grid", default=None,
                    help="comma-separated fixed-lag transition sigma values in rows")
    ap.add_argument("--hypothesis-obs-weight", type=float,
                    default=flt.HYPOTHESIS_OBS_WEIGHT)
    ap.add_argument("--hypothesis-obs-weight-grid", default=None,
                    help="comma-separated fixed-lag observation weights")
    ap.add_argument("--hypothesis-blend-immediate", action="store_true",
                    help="blend fixed-lag commits toward immediate PF when they diverge")
    ap.add_argument("--hypothesis-blend-delta-rows", type=float,
                    default=flt.HYPOTHESIS_BLEND_DELTA_ROWS)
    ap.add_argument("--hypothesis-blend-alpha", type=float,
                    default=flt.HYPOTHESIS_BLEND_ALPHA)
    ap.add_argument("--hypothesis-blend-saccade-p", type=float,
                    default=flt.HYPOTHESIS_BLEND_SACCADE_P)
    ap.add_argument("--slew-gate", action="store_true",
                    help="enable physiological pursuit-slew gate in fixed-lag resolver")
    ap.add_argument("--slew-max-deg-s", type=float, default=flt.SLEW_GATE_MAX_DEG_S)
    ap.add_argument("--slew-max-grid", default=None,
                    help="comma-separated slew thresholds; implies --slew-gate for each")
    ap.add_argument("--slew-gate-cost", type=float, default=flt.SLEW_GATE_COST)
    ap.add_argument("--slew-gate-saccade-p", type=float, default=flt.SLEW_GATE_SACCADE_P)
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
    ap.add_argument("--motion-prior-grid", default=None,
                    help="comma-separated sigma:tau_ms entries; implies --motion-prior")
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild candidate AQ caches")
    ap.add_argument("--rebuild-inputs", action="store_true",
                    help="rebuild chain/line caches before PF runs")
    ap.add_argument("--out-prefix", default=RESULT_PREFIX)
    ap.add_argument("--dry-run", action="store_true",
                    help="write a synthetic CSV/MD without running people-data PF")
    ap.add_argument("--write-policy-from", default=None,
                    help="comma-separated calibration CSVs; write a per-subject policy and exit")
    ap.add_argument("--policy-out-prefix", default=POLICY_PREFIX)
    ap.add_argument("--policy-from", default=None,
                    help="run a saved per-subject policy JSON instead of one shared variant grid")
    args = ap.parse_args(argv)

    if args.dur is None:
        dur_s = 20.0 if args.stage == "prune" else None
    else:
        dur_s = None if args.dur <= 0 else args.dur

    if args.write_policy_from:
        paths = [p.strip() for p in args.write_policy_from.split(",") if p.strip()]
        rows = []
        for path in paths:
            rows.extend(_read_csv_rows(path))
        out = write_subject_policy(rows, args.policy_out_prefix, source_paths=paths)
        print(f"Wrote {out[0]} and {out[1]}")
        return 0

    policy_variants = load_subject_policy(args.policy_from) if args.policy_from else None
    if policy_variants is not None:
        variants = []
    elif args.stage == "full" and args.configs is None and not args.dry_run:
        variants = select_finalist_variants(args.finalists_from, args.top_k)
    else:
        configs = _parse_configs(args.configs, args.max_configs)
        variants = _make_variants(
            configs, _parse_slew_grid(args.slew_gate, args.slew_max_deg_s,
                                      args.slew_max_grid),
            lag_specs=_parse_float_grid(args.lag_ms, args.lag_ms_grid),
            transition_sigma_specs=_parse_float_grid(
                args.hypothesis_transition_sigma_rows,
                args.hypothesis_transition_sigma_grid),
            obs_weight_specs=_parse_float_grid(
                args.hypothesis_obs_weight, args.hypothesis_obs_weight_grid),
            blend_immediate=args.hypothesis_blend_immediate,
            blend_delta_rows=args.hypothesis_blend_delta_rows,
            blend_alpha=args.hypothesis_blend_alpha,
            blend_saccade_p=args.hypothesis_blend_saccade_p,
            velocity_cost=args.hypothesis_velocity_cost,
            velocity_sigma_deg_s=args.hypothesis_velocity_sigma_deg_s,
            acceleration_cost=args.hypothesis_acceleration_cost,
            acceleration_sigma_deg_s2=args.hypothesis_acceleration_sigma_deg_s2,
            motion_specs=_parse_motion_prior_grid(
                args.motion_prior, args.motion_prior_sigma_rows,
                args.motion_prior_tau_ms, args.motion_prior_grid),
            motion_prior_ncc_thr=args.motion_prior_ncc_thr)
    if args.dry_run:
        names = ([x.strip() for x in args.subjects.split(",") if x.strip()]
                 if args.subjects else ["Ashton3", "Chong"])
        if policy_variants is not None:
            rows = []
            for name in names:
                if name not in policy_variants:
                    raise SystemExit(f"policy has no variant for subject {name!r}")
                rows.extend(_dry_rows([name], [policy_variants[name]],
                                      args.stage, dur_s))
        else:
            rows = _dry_rows(names, variants, args.stage, dur_s)
        paths = write_reports(rows, args.out_prefix)
        print(f"Wrote {paths[0]} and {paths[1]}")
        return 0

    subjects = _parse_subjects(args.subjects)
    rows = []
    for sub in subjects:
        sub_variants = ([policy_variants[sub.name]]
                        if policy_variants is not None and sub.name in policy_variants
                        else variants)
        if policy_variants is not None and sub.name not in policy_variants:
            raise SystemExit(f"policy has no variant for subject {sub.name!r}")
        print(f"\n=== {sub.name}: {len(sub_variants)} AQ variants, stage={args.stage} ===")
        for variant in sub_variants:
            cfg = variant.cfg
            print(f"  [{sub.name}] {variant.variant_id}")
            row = run_subject_config(
                sub, cfg, dur_s=dur_s, stage=args.stage,
                rebuild=args.rebuild, rebuild_inputs=args.rebuild_inputs,
                top_k=args.top_k, n_particles=args.n_particles,
                lag_ms=variant.lag_ms,
                hypothesis_transition_sigma_rows=variant.transition_sigma_rows,
                hypothesis_obs_weight=variant.obs_weight,
                hypothesis_blend_immediate=variant.blend_immediate,
                hypothesis_blend_delta_rows=variant.blend_delta_rows,
                hypothesis_blend_alpha=variant.blend_alpha,
                hypothesis_blend_saccade_p=variant.blend_saccade_p,
                slew_gate=variant.slew_gate,
                slew_max_deg_s=variant.slew_max_deg_s,
                slew_gate_cost=args.slew_gate_cost,
                slew_gate_saccade_p=args.slew_gate_saccade_p,
                hypothesis_velocity_cost=variant.velocity_cost,
                hypothesis_velocity_sigma_deg_s=variant.velocity_sigma_deg_s,
                hypothesis_acceleration_cost=variant.acceleration_cost,
                hypothesis_acceleration_sigma_deg_s2=variant.acceleration_sigma_deg_s2,
                motion_prior=variant.motion_prior,
                motion_prior_sigma_rows=variant.motion_prior_sigma_rows,
                motion_prior_tau_s=variant.motion_prior_tau_s,
                motion_prior_ncc_thr=variant.motion_prior_ncc_thr)
            rows.append(row)
            print(
                f"    gates={row['pass_hard_gates']} "
                f"jump_red={_fmt(row['raw_jump_reduction'])} "
                f"speed_red={_fmt(row['raw_speed_p999_reduction'])} "
                f"d_prec={_fmt(row['delta_prec_x'])} d_j30={_fmt(row['delta_j30'])}"
            )
    paths = write_reports(rows, args.out_prefix)
    print(f"\nWrote {paths[0]} and {paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
