"""Experiment 1 stimulus/logging for epistemic-decoder readiness.

This module builds a Gabor-orientation change-point sequence with computable
ideal-observer labels. It is deliberately independent of the SLO tracker path:
the output is the task/event substrate needed before running a real
epistemic-decoder experiment.

CLI example:

    python -m epistemic_task --out results/experiment1_demo --trials 4 --frames 240

Outputs:
  - experiment1_events.csv: one row per displayed frame/event
  - experiment1_manifest.json: config, column contract, and summary
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

SCHEMA_VERSION = "epistemic-task-v1"
TASK_NAME = "gabor_orientation_changepoint"


@dataclass(frozen=True)
class Experiment1Config:
    """Parameters for the first ground-truth uncertainty task.

    Orientations are axial Gabor orientations, so 0 and 180 degrees are
    equivalent. The ideal observer uses a discrete grid over this half-circle.
    """

    n_trials: int = 4
    frames_per_trial: int = 240
    frame_rate_hz: float = 60.0
    hazard: float = 0.08
    orientation_bins: int = 180
    observation_sd_deg: float = 6.0
    process_sd_deg: float = 0.0
    mean_luminance_cd_m2: float = 60.0
    gabor_contrast: float = 0.25
    seed: int = 0

    def validate(self) -> None:
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if self.frames_per_trial <= 0:
            raise ValueError("frames_per_trial must be positive")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")
        if not (0.0 <= self.hazard <= 1.0):
            raise ValueError("hazard must be in [0, 1]")
        if self.orientation_bins < 8:
            raise ValueError("orientation_bins must be >= 8")
        if self.observation_sd_deg <= 0:
            raise ValueError("observation_sd_deg must be positive")
        if self.process_sd_deg < 0:
            raise ValueError("process_sd_deg must be non-negative")
        if self.mean_luminance_cd_m2 <= 0:
            raise ValueError("mean_luminance_cd_m2 must be positive")
        if not (0.0 <= self.gabor_contrast <= 1.0):
            raise ValueError("gabor_contrast must be in [0, 1]")


@dataclass(frozen=True)
class FrameEvent:
    schema_version: str
    task_name: str
    trial_id: str
    trial_index: int
    frame_in_trial: int
    global_frame_id: int
    display_frame_id: int
    time_s: float
    hazard: float
    change: int
    latent_orientation_deg: float
    orientation_deg: float
    orientation_bin: int
    surprise_nats: float
    posterior_entropy_nats: float
    expected_information_gain_nats: float
    information_gain_nats: float
    predictive_probability: float
    mean_luminance_cd_m2: float
    luminance_cd_m2: float
    gabor_contrast: float


@dataclass(frozen=True)
class ExperimentLog:
    config: Experiment1Config
    orientation_grid_deg: np.ndarray
    events: list[FrameEvent]


def orientation_grid(n_bins: int) -> np.ndarray:
    """Return axial orientation bin centers in [0, 180)."""
    width = 180.0 / int(n_bins)
    return (np.arange(int(n_bins), dtype=float) + 0.5) * width


def orientation_delta_deg(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Smallest signed axial orientation difference a-b in degrees."""
    return (np.asarray(a) - np.asarray(b) + 90.0) % 180.0 - 90.0


def _kernel_matrix(grid: np.ndarray, sd_deg: float) -> np.ndarray:
    """Row-stochastic wrapped Gaussian kernel over the axial orientation grid."""
    if sd_deg == 0:
        return np.eye(len(grid), dtype=float)
    d = orientation_delta_deg(grid[None, :], grid[:, None])
    k = np.exp(-0.5 * (d / float(sd_deg)) ** 2)
    k /= k.sum(axis=1, keepdims=True)
    return k


def transition_matrix(grid: np.ndarray, process_sd_deg: float) -> np.ndarray:
    """p(theta_t | theta_{t-1}) for the no-change branch."""
    return _kernel_matrix(grid, process_sd_deg)


def emission_matrix(grid: np.ndarray, observation_sd_deg: float) -> np.ndarray:
    """p(displayed orientation bin | latent orientation bin)."""
    return _kernel_matrix(grid, observation_sd_deg)


def entropy_nats(p: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * np.log(p), 0.0)
    h = -np.sum(terms, axis=axis)
    if axis is None:
        return float(h)
    return h


def kl_nats(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0.0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.maximum(q[mask], 1e-300)))))


def advance_prior(
    previous_posterior: np.ndarray,
    transition: np.ndarray,
    hazard: float,
) -> np.ndarray:
    """Advance the ideal observer from t-1 to the predictive prior at t."""
    n = len(previous_posterior)
    uniform = np.full(n, 1.0 / n)
    stay = np.einsum("i,ij->j", previous_posterior, transition)
    prior = (1.0 - hazard) * stay + hazard * uniform
    return prior / prior.sum()


def expected_information_gain_nats(prior: np.ndarray, emission: np.ndarray) -> float:
    """Expected reduction in entropy before seeing the next orientation sample."""
    predictive = np.einsum("i,ij->j", prior, emission)
    weighted = prior[:, None] * emission
    posterior_by_obs = weighted / np.maximum(predictive[None, :], 1e-300)
    posterior_entropy = entropy_nats(posterior_by_obs, axis=0)
    eig = entropy_nats(prior) - float(np.sum(predictive * posterior_entropy))
    return max(0.0, eig)


def ideal_observer_update(
    prior: np.ndarray,
    observation_bin: int,
    emission: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float]:
    """Update on one displayed orientation.

    Returns posterior, predictive probability, surprise, posterior entropy, and
    actual Bayesian information gain KL(posterior || prior).
    """
    predictive = np.einsum("i,ij->j", prior, emission)
    p_obs = float(max(predictive[int(observation_bin)], 1e-300))
    posterior = prior * emission[:, int(observation_bin)]
    posterior /= posterior.sum()
    surprise = -math.log(p_obs)
    post_entropy = entropy_nats(posterior)
    information_gain = kl_nats(posterior, prior)
    return posterior, p_obs, surprise, post_entropy, information_gain


def _sample_next_state(
    rng: np.random.Generator,
    state_idx: int | None,
    hazard: float,
    transition: np.ndarray,
) -> tuple[int, bool]:
    n = transition.shape[0]
    if state_idx is None or rng.random() < hazard:
        return int(rng.integers(0, n)), True
    return int(rng.choice(n, p=transition[int(state_idx)])), False


def generate_experiment(config: Experiment1Config) -> ExperimentLog:
    """Generate frame-level task events with ideal-observer labels."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    grid = orientation_grid(config.orientation_bins)
    trans = transition_matrix(grid, config.process_sd_deg)
    emit = emission_matrix(grid, config.observation_sd_deg)
    uniform = np.full(config.orientation_bins, 1.0 / config.orientation_bins)
    events: list[FrameEvent] = []
    global_frame = 0

    for trial in range(config.n_trials):
        posterior = uniform.copy()
        state_idx: int | None = None
        trial_id = f"trial_{trial:04d}"
        for frame in range(config.frames_per_trial):
            prior = uniform.copy() if frame == 0 else advance_prior(
                posterior, trans, config.hazard)
            eig = expected_information_gain_nats(prior, emit)
            state_idx, changed = _sample_next_state(
                rng, state_idx, config.hazard if frame > 0 else 1.0, trans)
            obs_idx = int(rng.choice(config.orientation_bins, p=emit[state_idx]))
            posterior, p_obs, surprise, post_entropy, information_gain = (
                ideal_observer_update(prior, obs_idx, emit)
            )
            event = FrameEvent(
                schema_version=SCHEMA_VERSION,
                task_name=TASK_NAME,
                trial_id=trial_id,
                trial_index=trial,
                frame_in_trial=frame,
                global_frame_id=global_frame,
                display_frame_id=global_frame,
                time_s=global_frame / config.frame_rate_hz,
                hazard=config.hazard,
                change=int(changed),
                latent_orientation_deg=float(grid[state_idx]),
                orientation_deg=float(grid[obs_idx]),
                orientation_bin=obs_idx,
                surprise_nats=float(surprise),
                posterior_entropy_nats=float(post_entropy),
                expected_information_gain_nats=float(eig),
                information_gain_nats=float(information_gain),
                predictive_probability=float(p_obs),
                mean_luminance_cd_m2=config.mean_luminance_cd_m2,
                luminance_cd_m2=config.mean_luminance_cd_m2,
                gabor_contrast=config.gabor_contrast,
            )
            events.append(event)
            global_frame += 1

    return ExperimentLog(config=config, orientation_grid_deg=grid, events=events)


def event_columns() -> list[str]:
    return list(FrameEvent.__dataclass_fields__.keys())


def events_as_dicts(events: Iterable[FrameEvent]) -> list[dict]:
    rows = []
    for event in events:
        row = asdict(event)
        rows.append(row)
    return rows


def summary(log: ExperimentLog) -> dict:
    events = log.events
    surprises = np.array([e.surprise_nats for e in events], dtype=float)
    entropies = np.array([e.posterior_entropy_nats for e in events], dtype=float)
    eigs = np.array([e.expected_information_gain_nats for e in events], dtype=float)
    changes = np.array([e.change for e in events], dtype=int)
    return {
        "n_events": len(events),
        "n_trials": log.config.n_trials,
        "frames_per_trial": log.config.frames_per_trial,
        "duration_s": len(events) / log.config.frame_rate_hz,
        "change_count": int(changes.sum()),
        "change_fraction": float(changes.mean()) if len(changes) else 0.0,
        "surprise_nats_mean": float(surprises.mean()),
        "surprise_nats_max": float(surprises.max()),
        "posterior_entropy_nats_mean": float(entropies.mean()),
        "expected_information_gain_nats_mean": float(eigs.mean()),
    }


def write_logs(log: ExperimentLog, out_dir: str) -> tuple[str, str]:
    """Write CSV event log and JSON manifest. Returns (csv_path, manifest_path)."""
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "experiment1_events.csv")
    manifest_path = os.path.join(out_dir, "experiment1_manifest.json")

    columns = event_columns()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(events_as_dicts(log.events))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_name": TASK_NAME,
        "config": asdict(log.config),
        "orientation_grid_deg": log.orientation_grid_deg.tolist(),
        "event_log": os.path.basename(csv_path),
        "event_columns": columns,
        "summary": summary(log),
        "readiness_note": (
            "Use this for offline decoder development only until display timing, "
            "photodiode/TTL sync, calibrated luminance, and tracker precision "
            "certification are complete."
        ),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return csv_path, manifest_path


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/experiment1_demo",
                    help="directory for experiment1_events.csv and manifest")
    ap.add_argument("--trials", type=int, default=Experiment1Config.n_trials)
    ap.add_argument("--frames", type=int, default=Experiment1Config.frames_per_trial)
    ap.add_argument("--frame-rate", type=float, default=Experiment1Config.frame_rate_hz)
    ap.add_argument("--hazard", type=float, default=Experiment1Config.hazard)
    ap.add_argument("--bins", type=int, default=Experiment1Config.orientation_bins)
    ap.add_argument("--obs-sd", type=float, default=Experiment1Config.observation_sd_deg)
    ap.add_argument("--process-sd", type=float, default=Experiment1Config.process_sd_deg)
    ap.add_argument("--mean-luminance", type=float,
                    default=Experiment1Config.mean_luminance_cd_m2)
    ap.add_argument("--contrast", type=float, default=Experiment1Config.gabor_contrast)
    ap.add_argument("--seed", type=int, default=Experiment1Config.seed)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = Experiment1Config(
        n_trials=args.trials,
        frames_per_trial=args.frames,
        frame_rate_hz=args.frame_rate,
        hazard=args.hazard,
        orientation_bins=args.bins,
        observation_sd_deg=args.obs_sd,
        process_sd_deg=args.process_sd,
        mean_luminance_cd_m2=args.mean_luminance,
        gabor_contrast=args.contrast,
        seed=args.seed,
    )
    log = generate_experiment(cfg)
    csv_path, manifest_path = write_logs(log, args.out)
    s = summary(log)
    print(f"wrote {csv_path}")
    print(f"wrote {manifest_path}")
    print(
        f"{s['n_events']} events, {s['change_count']} changes, "
        f"mean surprise {s['surprise_nats_mean']:.3f} nats, "
        f"mean entropy {s['posterior_entropy_nats_mean']:.3f} nats"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
