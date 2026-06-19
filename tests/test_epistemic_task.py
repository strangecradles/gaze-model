import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import epistemic_task as et  # noqa: E402


def test_generate_experiment_is_deterministic_and_frame_complete():
    cfg = et.Experiment1Config(n_trials=2, frames_per_trial=20, seed=42)
    a = et.generate_experiment(cfg)
    b = et.generate_experiment(cfg)

    assert len(a.events) == 40
    assert [e.orientation_bin for e in a.events] == [e.orientation_bin for e in b.events]
    assert [e.change for e in a.events] == [e.change for e in b.events]
    assert [e.display_frame_id for e in a.events] == list(range(40))
    assert {e.trial_id for e in a.events} == {"trial_0000", "trial_0001"}


def test_ideal_observer_labels_are_finite_probabilistic_and_consistent():
    cfg = et.Experiment1Config(
        n_trials=1,
        frames_per_trial=60,
        orientation_bins=90,
        observation_sd_deg=5.0,
        seed=7,
    )
    log = et.generate_experiment(cfg)
    events = log.events

    for event in events:
        assert 0.0 < event.predictive_probability <= 1.0
        assert math.isfinite(event.surprise_nats)
        assert math.isfinite(event.posterior_entropy_nats)
        assert math.isfinite(event.expected_information_gain_nats)
        assert event.expected_information_gain_nats >= 0.0
        assert abs(event.surprise_nats + math.log(event.predictive_probability)) < 1e-12

    assert events[0].change == 1
    assert max(e.posterior_entropy_nats for e in events) <= math.log(cfg.orientation_bins)


def test_expected_information_gain_matches_direct_enumeration():
    grid = et.orientation_grid(12)
    emission = et.emission_matrix(grid, observation_sd_deg=10.0)
    prior = np.linspace(1.0, 2.0, len(grid))
    prior /= prior.sum()

    predictive = np.einsum("i,ij->j", prior, emission)
    direct = et.entropy_nats(prior)
    for obs in range(len(grid)):
        posterior = prior * emission[:, obs] / predictive[obs]
        direct -= predictive[obs] * et.entropy_nats(posterior)

    assert abs(et.expected_information_gain_nats(prior, emission) - direct) < 1e-12


def test_writer_outputs_csv_and_manifest(tmp_path):
    cfg = et.Experiment1Config(n_trials=1, frames_per_trial=8, seed=3)
    log = et.generate_experiment(cfg)
    csv_path, manifest_path = et.write_logs(log, str(tmp_path))

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(rows) == 8
    assert rows[0]["schema_version"] == et.SCHEMA_VERSION
    assert rows[0]["task_name"] == et.TASK_NAME
    assert "surprise_nats" in rows[0]
    assert "posterior_entropy_nats" in rows[0]
    assert "expected_information_gain_nats" in rows[0]
    assert manifest["event_log"] == "experiment1_events.csv"
    assert manifest["summary"]["n_events"] == 8
