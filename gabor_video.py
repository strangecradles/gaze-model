"""Render Gabor change-point stimulus videos with epistemic labels.

This is the concrete Experiment 1A stimulus builder. It uses
``epistemic_task.generate_experiment`` for the Bayesian ideal-observer labels,
then renders a luminance-stable Gabor movie plus sync-friendly logs.

Default output is three pilot videos:

    python3 -m gabor_video --preset-set pilot

Each condition directory contains:
  - stimulus.mp4
  - experiment1_events.csv        one row per orientation event
  - display_frames.csv            one row per video/display frame
  - stimulus_manifest.json
  - preview_contact_sheet.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass

import imageio.v2 as imageio
import numpy as np
from PIL import Image

import epistemic_task as et


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1920
    height: int = 1080
    display_fps: float = 60.0
    event_hz: float = 4.0
    gabor_size_px: int = 420
    spatial_cycles: float = 6.0
    gaussian_sigma_frac: float = 0.22
    background_gray: int = 128
    gabor_contrast: float = 0.30
    fixation_radius_px: int = 5
    fixation_gray: int = 10
    photodiode_size_px: int = 72
    photodiode_on_gray: int = 255
    photodiode_off_gray: int = 0
    photodiode_pulse_frames: int = 2
    codec: str = "libx264"
    quality: int = 8

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.display_fps <= 0 or self.event_hz <= 0:
            raise ValueError("display_fps and event_hz must be positive")
        hold = self.hold_frames
        if abs(hold - round(hold)) > 1e-9:
            raise ValueError(
                "display_fps / event_hz must be an integer so event boundaries "
                "land on display frames")
        if self.gabor_size_px <= 0 or self.gabor_size_px > min(self.width, self.height):
            raise ValueError("gabor_size_px must fit within the video frame")
        if not (0.0 <= self.gabor_contrast <= 1.0):
            raise ValueError("gabor_contrast must be in [0, 1]")
        if self.photodiode_size_px < 0:
            raise ValueError("photodiode_size_px must be non-negative")
        if self.photodiode_pulse_frames < 0:
            raise ValueError("photodiode_pulse_frames must be non-negative")

    @property
    def hold_frames(self) -> int:
        return int(round(self.display_fps / self.event_hz))


PRESETS = {
    "low": dict(hazard=0.02, seed=101),
    "medium": dict(hazard=0.08, seed=202),
    "high": dict(hazard=0.20, seed=303),
}


def _gabor_patch(
    orientation_deg: float,
    cfg: RenderConfig,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Return a uint8 square Gabor patch with matched mean gray."""
    n = int(cfg.gabor_size_px)
    coords = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    theta = math.radians(float(orientation_deg))

    # Orientation denotes the grating bar orientation. The sinusoid varies along
    # the perpendicular axis.
    carrier_axis = xx * math.cos(theta + math.pi / 2.0) + yy * math.sin(theta + math.pi / 2.0)
    carrier = np.cos(2.0 * math.pi * cfg.spatial_cycles * carrier_axis + phase_rad)
    sigma = float(cfg.gaussian_sigma_frac)
    envelope = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma))
    signal = envelope * carrier
    signal = signal - signal.mean()
    patch = cfg.background_gray + cfg.background_gray * cfg.gabor_contrast * signal
    return np.clip(np.rint(patch), 0, 255).astype(np.uint8)


def _base_frame(cfg: RenderConfig) -> np.ndarray:
    frame = np.full((cfg.height, cfg.width, 3), cfg.background_gray, dtype=np.uint8)
    return frame


def _draw_fixation(frame: np.ndarray, cfg: RenderConfig) -> None:
    cy = cfg.height // 2
    cx = cfg.width // 2
    r = int(cfg.fixation_radius_px)
    y0, y1 = max(0, cy - r), min(cfg.height, cy + r + 1)
    x0, x1 = max(0, cx - r), min(cfg.width, cx + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    frame[y0:y1, x0:x1][mask] = cfg.fixation_gray


def render_frame(event: et.FrameEvent, cfg: RenderConfig, pulse_on: bool) -> np.ndarray:
    """Render one video frame for one event."""
    frame = _base_frame(cfg)
    patch = _gabor_patch(event.orientation_deg, cfg)
    y0 = (cfg.height - cfg.gabor_size_px) // 2
    x0 = (cfg.width - cfg.gabor_size_px) // 2
    frame[y0:y0 + cfg.gabor_size_px, x0:x0 + cfg.gabor_size_px, :] = patch[:, :, None]
    _draw_fixation(frame, cfg)

    pd = int(cfg.photodiode_size_px)
    if pd > 0:
        level = cfg.photodiode_on_gray if pulse_on else cfg.photodiode_off_gray
        frame[:pd, :pd, :] = level
    return frame


def _display_frame_rows(log: et.ExperimentLog, cfg: RenderConfig) -> list[dict]:
    rows = []
    hold = cfg.hold_frames
    for event in log.events:
        for k in range(hold):
            display_frame_id = event.global_frame_id * hold + k
            pulse_on = k < cfg.photodiode_pulse_frames
            rows.append({
                "display_frame_id": display_frame_id,
                "display_time_s": display_frame_id / cfg.display_fps,
                "event_global_id": event.global_frame_id,
                "trial_id": event.trial_id,
                "trial_index": event.trial_index,
                "event_in_trial": event.frame_in_trial,
                "display_frame_in_event": k,
                "orientation_deg": event.orientation_deg,
                "orientation_bin": event.orientation_bin,
                "latent_orientation_deg": event.latent_orientation_deg,
                "hazard": event.hazard,
                "change": event.change if k == 0 else 0,
                "surprise_nats": event.surprise_nats,
                "posterior_entropy_nats": event.posterior_entropy_nats,
                "expected_information_gain_nats": event.expected_information_gain_nats,
                "information_gain_nats": event.information_gain_nats,
                "predictive_probability": event.predictive_probability,
                "mean_luminance_cd_m2": event.mean_luminance_cd_m2,
                "luminance_cd_m2": event.luminance_cd_m2,
                "gabor_contrast": event.gabor_contrast,
                "photodiode_level": (
                    cfg.photodiode_on_gray if pulse_on and cfg.photodiode_size_px > 0
                    else cfg.photodiode_off_gray
                ),
                "photodiode_pulse": int(pulse_on and cfg.photodiode_size_px > 0),
            })
    return rows


def _write_display_frames(rows: list[dict], path: str) -> None:
    if not rows:
        raise ValueError("no display frame rows")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_preview(log: et.ExperimentLog, cfg: RenderConfig, out_path: str) -> None:
    n = min(12, len(log.events))
    if n == 0:
        return
    thumbs = []
    for idx in np.linspace(0, len(log.events) - 1, n, dtype=int):
        im = Image.fromarray(render_frame(log.events[int(idx)], cfg, pulse_on=True))
        thumbs.append(im.resize((320, 180), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (320 * 4, 180 * math.ceil(n / 4)), (cfg.background_gray,) * 3)
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % 4) * 320, (i // 4) * 180))
    sheet.save(out_path)


def render_video(
    task_cfg: et.Experiment1Config,
    render_cfg: RenderConfig,
    out_dir: str,
    condition_name: str,
) -> dict:
    """Generate task labels, render MP4, and write all sidecar logs."""
    task_cfg.validate()
    render_cfg.validate()
    os.makedirs(out_dir, exist_ok=True)

    log = et.generate_experiment(task_cfg)
    events_csv, task_manifest_path = et.write_logs(log, out_dir)
    video_path = os.path.join(out_dir, "stimulus.mp4")
    display_csv = os.path.join(out_dir, "display_frames.csv")
    preview_path = os.path.join(out_dir, "preview_contact_sheet.png")
    stimulus_manifest_path = os.path.join(out_dir, "stimulus_manifest.json")

    frame_rows = _display_frame_rows(log, render_cfg)
    _write_display_frames(frame_rows, display_csv)

    with imageio.get_writer(
        video_path,
        fps=render_cfg.display_fps,
        codec=render_cfg.codec,
        quality=render_cfg.quality,
        pixelformat="yuv420p",
        macro_block_size=1,
    ) as writer:
        for event in log.events:
            for k in range(render_cfg.hold_frames):
                frame = render_frame(
                    event, render_cfg, pulse_on=k < render_cfg.photodiode_pulse_frames)
                writer.append_data(frame)

    _write_preview(log, render_cfg, preview_path)

    manifest = {
        "schema_version": "gabor-video-v1",
        "condition_name": condition_name,
        "task_name": et.TASK_NAME,
        "task_config": asdict(task_cfg),
        "render_config": asdict(render_cfg),
        "hold_frames_per_event": render_cfg.hold_frames,
        "n_events": len(log.events),
        "n_display_frames": len(frame_rows),
        "duration_s": len(frame_rows) / render_cfg.display_fps,
        "files": {
            "video": os.path.basename(video_path),
            "events_csv": os.path.basename(events_csv),
            "display_frames_csv": os.path.basename(display_csv),
            "task_manifest": os.path.basename(task_manifest_path),
            "preview_contact_sheet": os.path.basename(preview_path),
        },
        "sync_notes": (
            "Top-left photodiode square pulses high for the first "
            f"{render_cfg.photodiode_pulse_frames} display frames of every "
            "orientation event. Use display_frames.csv to align pulse edges to "
            "event_global_id and ideal-observer labels."
        ),
    }
    with open(stimulus_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="stimuli/epistemic_gabor",
                    help="output directory")
    ap.add_argument("--preset-set", choices=["pilot"], default=None,
                    help="render low/medium/high pilot videos")
    ap.add_argument("--condition", default="medium",
                    help="condition name for single-video mode")
    ap.add_argument("--hazard", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--events-per-trial", type=int, default=80)
    ap.add_argument("--event-hz", type=float, default=4.0)
    ap.add_argument("--display-fps", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--size", type=int, default=420, help="Gabor size in pixels")
    ap.add_argument("--contrast", type=float, default=0.30)
    ap.add_argument("--obs-sd", type=float, default=6.0)
    ap.add_argument("--no-photodiode", action="store_true",
                    help="hide the top-left photodiode square for headset presentation")
    return ap


def _task_cfg(args: argparse.Namespace, hazard: float, seed: int) -> et.Experiment1Config:
    return et.Experiment1Config(
        n_trials=args.trials,
        frames_per_trial=args.events_per_trial,
        frame_rate_hz=args.event_hz,
        hazard=hazard,
        observation_sd_deg=args.obs_sd,
        gabor_contrast=args.contrast,
        seed=seed,
    )


def _render_cfg(args: argparse.Namespace) -> RenderConfig:
    return RenderConfig(
        width=args.width,
        height=args.height,
        display_fps=args.display_fps,
        event_hz=args.event_hz,
        gabor_size_px=args.size,
        gabor_contrast=args.contrast,
        photodiode_size_px=0 if args.no_photodiode else RenderConfig.photodiode_size_px,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rcfg = _render_cfg(args)
    if args.preset_set == "pilot":
        root = os.path.join(args.out, "pilot")
        manifests = []
        for name, params in PRESETS.items():
            out_dir = os.path.join(root, f"{name}_hazard_{params['hazard']:0.2f}")
            manifest = render_video(
                _task_cfg(args, params["hazard"], params["seed"]),
                rcfg,
                out_dir,
                condition_name=name,
            )
            manifests.append({
                "condition_name": name,
                "hazard": params["hazard"],
                "directory": os.path.relpath(out_dir, root),
                "duration_s": manifest["duration_s"],
                "n_events": manifest["n_events"],
            })
            print(f"wrote {out_dir}/stimulus.mp4")
        index_path = os.path.join(root, "pilot_manifest.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": "gabor-pilot-set-v1",
                "conditions": manifests,
                "render_config": asdict(rcfg),
            }, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote {index_path}")
        return 0

    out_dir = os.path.join(args.out, args.condition)
    manifest = render_video(
        _task_cfg(args, args.hazard, args.seed),
        rcfg,
        out_dir,
        condition_name=args.condition,
    )
    print(f"wrote {out_dir}/stimulus.mp4")
    print(f"duration {manifest['duration_s']:.1f}s, events {manifest['n_events']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
