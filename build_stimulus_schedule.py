"""Build collection-ready schedules for rendered Gabor stimulus sets.

The renderer writes per-condition sidecars. This script combines them into:

  - collection_schedule_frames.csv: one row per 60 Hz video frame
  - collection_schedule_events.csv: one row per Gabor orientation event

Both outputs include video file, condition, hazard, frame/event IDs, and the
ideal-observer labels needed for analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


FRAME_COLUMNS = [
    "video_file",
    "video_path",
    "condition",
    "hazard",
    "display_frame_id",
    "display_time_s",
    "event_global_id",
    "trial_id",
    "trial_index",
    "event_in_trial",
    "display_frame_in_event",
    "orientation_deg",
    "change",
    "surprise_nats",
    "posterior_entropy_nats",
    "expected_information_gain_nats",
]

EVENT_COLUMNS = [
    "video_file",
    "video_path",
    "condition",
    "hazard",
    "event_global_id",
    "display_frame_id",
    "display_time_s",
    "trial_id",
    "trial_index",
    "event_in_trial",
    "orientation_deg",
    "change",
    "surprise_nats",
    "posterior_entropy_nats",
    "expected_information_gain_nats",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_schedule(root: Path) -> tuple[Path, Path]:
    manifest_path = root / "pilot_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    frame_rows: list[dict[str, str]] = []
    event_rows: list[dict[str, str]] = []
    for condition in manifest["conditions"]:
        condition_name = condition["condition_name"]
        condition_dir = root / condition["directory"]
        video_path = condition_dir / "stimulus.mp4"
        video_abs = str(video_path.resolve())
        video_file = f"{condition['directory']}/{os.path.basename(video_path)}"

        rows = _read_csv(condition_dir / "display_frames.csv")
        for row in rows:
            out = {
                "video_file": video_file,
                "video_path": video_abs,
                "condition": condition_name,
                "hazard": row["hazard"],
            }
            for key in FRAME_COLUMNS:
                if key not in out:
                    out[key] = row[key]
            frame_rows.append(out)

            if row["display_frame_in_event"] == "0":
                event_rows.append({key: out[key] for key in EVENT_COLUMNS})

    frames_path = root / "collection_schedule_frames.csv"
    events_path = root / "collection_schedule_events.csv"
    _write_csv(frames_path, FRAME_COLUMNS, frame_rows)
    _write_csv(events_path, EVENT_COLUMNS, event_rows)
    return frames_path, events_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default="stimuli/epistemic_gabor_4hz_headset/pilot",
        help="stimulus set directory containing pilot_manifest.json",
    )
    args = parser.parse_args(argv)
    frames_path, events_path = build_schedule(Path(args.root))
    print(f"wrote {frames_path}")
    print(f"wrote {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
