"""tests/test_eval_real.py — G15 real-data deployment + validation (GATE 2, FINAL).

This is a HARNESS / honesty test: it asserts the three real-data checks RUN on the
real test2 stream and produce FINITE, well-formed numbers, and that the report
states the three checks + the honest verdict + the necessary-not-sufficient note.

It does NOT assert a specific positive SCIENCE result. Per the G15 spec, real
performance may DIVERGE from the synthetic prediction (the cross-scale regime is
the documented pessimistic case) and that divergence is a VALID, valuable outcome
that must be reported honestly — not asserted away.

Structural facts asserted (must hold regardless of the science outcome):
  - the self-supervised filter produces a FINITE trajectory on real data;
  - (a) the slow-vs-tracker correlation is computed and finite in [-1, 1];
  - (b) the oculomotor main-sequence slope (and log-corr) and drift-spectrum slope
        and microsaccade rate are finite;
  - (c) the per-rate alias / persistence numbers are finite and reported, and the
        collapse-with-rate trend is decided (a boolean) — not its truth value.
  - results/real_validation.md is written and contains the three sections + the
    necessary-not-sufficient note.

Runtime is kept reasonable: a small capped window, short durations, only two
effective rates for the alias scan, and the on-disk cache (cache/eval_real_*).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_real as er

# Small, fast configuration — still real data, still all three checks.
TEST_WINDOW = 60_000          # ~5 s @ ~12 kHz
SLOW_RATE = 820.0
SLOW_DUR = 4.0
ALIAS_RATES = (344.0, 2000.0)   # one below, one above the ~826 Hz gate
ALIAS_DUR = 3.0

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def validation():
    """Run the three real-data checks once (cached) on a small real window."""
    er.WINDOW_SWEEPS = TEST_WINDOW
    er._REAL = None  # force reload at the test window size
    v = er.run_evaluation(slow_rate=SLOW_RATE, slow_dur_s=SLOW_DUR,
                          alias_rates=ALIAS_RATES, alias_dur_s=ALIAS_DUR,
                          use_cache=True)
    return v


# ---------------------------------------------------------------------------
# The filter produces a finite trajectory on real data (self-supervised, no labels)
# ---------------------------------------------------------------------------

def test_filter_runs_finite_on_real_stream():
    run = er.run_real_filter(SLOW_RATE, dur_cap_s=SLOW_DUR, window_sweeps=TEST_WINDOW)
    assert run.est_perp.ndim == 1 and run.est_perp.size > 10
    assert np.isfinite(run.est_perp).all(), "real perp estimate has non-finite values"
    assert np.isfinite(run.est_along).all(), "real along estimate has non-finite values"
    assert np.isfinite(run.max_ncc).all()
    assert run.rate > 0
    # the estimate must move (a frozen/degenerate output would be a bug)
    assert np.ptp(run.est_perp) > 0


# ---------------------------------------------------------------------------
# (a) slow-vs-tracker correlation computed, finite, in [-1, 1]
# ---------------------------------------------------------------------------

def test_slow_vs_tracker_correlation_finite(validation):
    s = validation.slow
    for r in (s.r_perp, s.r_along, s.r_perp_fixed_off, s.r_along_fixed_off):
        assert np.isfinite(r), "slow-vs-tracker correlation is not finite"
        assert -1.0 <= r <= 1.0, f"correlation {r} out of [-1, 1]"
    assert np.isfinite(s.off_perp) and np.isfinite(s.off_along)


# ---------------------------------------------------------------------------
# (b) oculomotor stats finite
# ---------------------------------------------------------------------------

def test_oculomotor_stats_finite(validation):
    o = validation.oculo
    assert np.isfinite(o.main_seq_slope), "main-sequence slope is not finite"
    assert np.isfinite(o.main_seq_logcorr)
    assert -1.0 <= o.main_seq_logcorr <= 1.0
    assert np.isfinite(o.drift_psd_slope), "drift-spectrum slope is not finite"
    assert np.isfinite(o.microsaccade_rate_per_s)
    assert o.microsaccade_rate_per_s >= 0.0
    assert o.n_saccades >= 0


# ---------------------------------------------------------------------------
# (c) per-rate alias / persistence numbers finite and reported
# ---------------------------------------------------------------------------

def test_alias_collapse_numbers_finite(validation):
    pts = validation.alias
    assert len(pts) == len(ALIAS_RATES)
    for p in pts:
        assert np.isfinite(p.psr_median)
        assert np.isfinite(p.n_modes_median)
        assert np.isfinite(p.persist_max_ms) and p.persist_max_ms >= 0.0
        assert np.isfinite(p.persist_p90_ms) and p.persist_p90_ms >= 0.0
        assert np.isfinite(p.max_ncc_median)
        assert p.n_steps > 0
    # the collapse trend is DECIDED (a boolean); its truth value is NOT asserted
    assert isinstance(validation.alias_collapses, bool)


# ---------------------------------------------------------------------------
# determinism (cached + seeded -> identical estimate)
# ---------------------------------------------------------------------------

def test_real_run_deterministic():
    a = er.run_real_filter(SLOW_RATE, dur_cap_s=SLOW_DUR, window_sweeps=TEST_WINDOW,
                           use_cache=False)
    b = er.run_real_filter(SLOW_RATE, dur_cap_s=SLOW_DUR, window_sweeps=TEST_WINDOW,
                           use_cache=False)
    assert np.allclose(a.est_perp, b.est_perp)
    assert np.allclose(a.est_along, b.est_along)


# ---------------------------------------------------------------------------
# report writes the three sections + the necessary-not-sufficient note
# ---------------------------------------------------------------------------

def test_report_contains_three_checks_and_honesty_note(validation, tmp_path):
    path = tmp_path / "real_validation.md"
    er.save_report(validation, path=str(path))
    assert path.exists()
    text = path.read_text()
    # the three checks
    assert "(a) SLOW component vs frame-rate truth" in text
    assert "(b) Oculomotor statistics" in text
    assert "(c) Alias-structure collapse with rate" in text
    # numbers present
    assert "r = " in text          # correlation reported
    assert "PersistMax" in text    # per-rate persistence table
    assert "main-sequence" in text.lower() or "MAIN SEQUENCE" in text
    # the explicit honesty notes
    assert "NECESSARY, NOT SUFFICIENT" in text
    assert "self-consistent != correct" in text
    assert "cross-scale" in text.lower()
    # a clear MATCH-or-DIVERGE verdict
    assert "VERDICT" in text
    assert ("MATCH" in text) or ("DIVERGE" in text)


def test_report_written_to_results_path(validation):
    """The canonical results/real_validation.md is produced by the report path."""
    path = er.save_report(validation,
                          path=os.path.join(HERE, "results", "real_validation.md"))
    assert os.path.exists(path)
    text = open(path).read()
    assert "G15" in text and "NECESSARY, NOT SUFFICIENT" in text
