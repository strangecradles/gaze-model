"""tests/test_rate_sweep.py — G13 decisive rate-sweep gate (HARNESS test).

This validates the HARNESS and that the decisive STRUCTURE is measured + the
verdict is written — NOT a fudged positive science outcome. Per the G13 spec we
must NOT assert a saccade result that isn't real.

What is asserted:
- The sweep RUNS across multiple rates and produces FINITE metrics.
- The decisive quantities are measured: fixation accuracy improves/holds toward
  sub-0.1 deg at high rate; persistence (gross-run-length in ms) is reported at
  every rate; lock-rate vs rate is reported.
- The 3-panel figure is produced.
- The verdict file is written and contains a clear (a)/(b)/overall conclusion.

Runtime is kept reasonable via short durations, a modest seed count, and the
on-disk cache (cache/ratesweep_*). A small subset of rates is used that still
brackets the ~820 Hz gate (frame-rate, the gate, and a high rate).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rate_sweep as rs

# A subset that still brackets the gate (~826 Hz): low / gate / high.
TEST_RATES = [344.0, 820.0, 2000.0]
TEST_SEEDS = 3


@pytest.fixture(scope="module")
def sweep():
    """Run the sweep once (cached) for the test rate subset."""
    summaries, all_metrics = rs.run_sweep(
        rates=TEST_RATES, n_seeds=TEST_SEEDS, use_cache=True, verbose=False)
    return summaries, all_metrics


# ---------------------------------------------------------------------------
# Harness runs and produces finite metrics across all rates
# ---------------------------------------------------------------------------

def test_sweep_runs_all_rates(sweep):
    summaries, _ = sweep
    assert len(summaries) == len(TEST_RATES)
    for s, rate in zip(summaries, TEST_RATES):
        assert s.rate == rate
        assert s.n_seeds == TEST_SEEDS


def test_metrics_are_finite(sweep):
    summaries, _ = sweep
    for s in summaries:
        # fixation must always be measurable and finite
        assert np.isfinite(s.rms_fix_arcmin), f"fix RMS nan @ {s.rate}"
        assert np.isfinite(s.rms_overall_arcmin)
        assert np.isfinite(s.gross_fix)
        assert np.isfinite(s.lock_fix)
        # persistence is always defined (>= 0)
        assert np.isfinite(s.persist_max_ms) and s.persist_max_ms >= 0.0
        assert np.isfinite(s.persist_p90_ms) and s.persist_p90_ms >= 0.0
        # lock-rates are valid fractions
        assert 0.0 <= s.lock_fix <= 1.0
        assert 0.0 <= s.lock_overall <= 1.0


def test_saccade_metrics_present_somewhere(sweep):
    """At least one rate has saccade steps so the velocity split is exercised."""
    summaries, _ = sweep
    assert any(s.n_sac_total > 0 for s in summaries), \
        "no saccade steps in any stream — velocity split not exercised"
    for s in summaries:
        if s.n_sac_total > 0:
            assert np.isfinite(s.rms_sac_arcmin)


# ---------------------------------------------------------------------------
# Decisive STRUCTURE is measured (not a specific science pass/fail)
# ---------------------------------------------------------------------------

def test_fixation_accuracy_improves_or_holds_at_high_rate(sweep):
    """Fixation accuracy at the high rate is no worse than at the low rate
    (the physics: more samples + less blur at high rate). Decisive structure."""
    summaries, _ = sweep
    by_rate = {s.rate: s for s in summaries}
    lo = by_rate[min(TEST_RATES)]
    hi = by_rate[max(TEST_RATES)]
    # high-rate fixation RMS should improve (be <=) the low-rate one.
    assert hi.rms_fix_arcmin <= lo.rms_fix_arcmin + 1e-6, (
        f"fixation accuracy did not improve/hold with rate: "
        f"{lo.rate}Hz={lo.rms_fix_arcmin:.2f}' vs {hi.rate}Hz={hi.rms_fix_arcmin:.2f}'")


def test_high_rate_fixation_sub_0p1deg(sweep):
    """At the high rate, fixation/pursuit reaches sub-0.1 deg (the (b) finding
    for the fixation velocity class — this IS real and expected)."""
    summaries, _ = sweep
    hi = max(summaries, key=lambda s: s.rate)
    assert hi.rms_fix_arcmin < rs.DOD_ARCMIN, (
        f"high-rate ({hi.rate}Hz) fixation RMS {hi.rms_fix_arcmin:.2f}' "
        f"not sub-0.1 deg ({rs.DOD_ARCMIN}')")


def test_persistence_reported_in_ms(sweep):
    """Persistence-vs-rate is reported in TIME (ms) at every rate (decisive)."""
    summaries, _ = sweep
    persist = [s.persist_max_ms for s in summaries]
    assert all(np.isfinite(p) and p >= 0.0 for p in persist)
    # the key decisive trend exists as a measured series (not asserting collapse
    # sign here — that is reported in the verdict, not gated by the test).
    assert len(persist) == len(TEST_RATES)


def test_lock_rate_vs_velocity_reported(sweep):
    """Lock-rate-vs-velocity table has finite entries for populated bins."""
    summaries, _ = sweep
    for s in summaries:
        lk = s.lock_rate_by_vel
        finite = lk[np.isfinite(lk)]
        assert finite.size > 0, f"no lock-rate-vs-velocity data @ {s.rate}"
        assert np.all((finite >= 0.0) & (finite <= 1.0))


# ---------------------------------------------------------------------------
# Figure + verdict artifacts
# ---------------------------------------------------------------------------

def test_figure_is_produced(tmp_path, sweep):
    summaries, _ = sweep
    figp = rs.make_figure(summaries, path=str(tmp_path / "rate_sweep.png"))
    assert os.path.exists(figp)
    assert os.path.getsize(figp) > 0


def test_verdict_written_with_clear_conclusion(tmp_path, sweep):
    summaries, _ = sweep
    path = str(tmp_path / "rate_sweep_verdict.md")
    info = rs.save_verdict(summaries, path=path)
    assert os.path.exists(path)
    text = open(path).read()
    # the decisive (a)/(b)/overall structure must be present
    assert "(a)" in text and "persistence" in text.lower()
    assert "(b)" in text
    assert "Overall verdict" in text
    # a clear case must be selected
    assert info["verdict"] in ("i", "ii", "iii")
    # one of the explicit conclusion strings must appear
    assert any(tok in text for tok in
               ("VIABLE AS-IS", "VIABLE FOR FIXATION/PURSUIT", "DEAD"))
    # the figure reference is in the verdict
    assert "rate_sweep.png" in text


def test_determinism(sweep):
    """Re-running the sweep (from cache) yields identical fixation RMS."""
    summaries, _ = sweep
    summaries2, _ = rs.run_sweep(rates=TEST_RATES, n_seeds=TEST_SEEDS,
                                 use_cache=True, verbose=False)
    for a, b in zip(summaries, summaries2):
        assert a.rms_fix_arcmin == pytest.approx(b.rms_fix_arcmin, abs=1e-9)
        assert a.persist_max_ms == pytest.approx(b.persist_max_ms, abs=1e-9)
