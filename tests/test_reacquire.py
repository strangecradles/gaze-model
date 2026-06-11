"""G11 acceptance tests: reacquisition / reseed.

Tests that the particle filter detects lock loss and reseeds from the coarse
anchor, recovers within a bounded number of steps, does NOT coast indefinitely,
and leaves the G10 fixation accuracy intact (no regression).

The lock-loss trigger is a sustained low observation-NCC, distinct from the
normal ESS-driven resample that fires on ~97% of healthy fixation steps.

DoD: tests pass; filter reseeds and recovers after injected lock loss;
clean fixation has zero / negligible reseeds; G10 accuracy preserved.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calib          # noqa: E402
import data           # noqa: E402
import filter as flt  # noqa: E402
import synth_stream as ss  # noqa: E402
import traj_gen       # noqa: E402

ATLAS = data.load_atlas()
RATE = 4000.0
LINE_LEN = 200
DEG_THRESH_ROWS = 0.1 * calib.ROWS_PER_DEG     # ~12.5 rows = 0.1 deg
# Recovery bound: how many steps after garbage ends until the filter is back
# within 0.1 deg and STAYS there for 20 consecutive steps.
RECOVERY_CAP_STEPS = 300  # 75 ms at 4 kHz — a generous but finite bound


def _make_stream(seed: int = 7, duration: float = 0.4):
    """Pure fixation/drift stream at RATE (seed 7 gives a fully mode==0 run)."""
    traj = traj_gen.sample_trajectory(duration, RATE, seed)
    assert np.all(traj.mode == 0), f"seed {seed} produced saccades; pick a pure-fixation seed"
    return ss.render_stream(traj, ATLAS, rate=RATE, line_len=LINE_LEN, seed=seed + 100)


def _run(stream, along_meas, coarse_anchor, seed=0, **kwargs):
    """Thin wrapper around flt.run with the stream's true along + coarse anchor."""
    tp = stream.trajectory.perp_rows
    ta = stream.trajectory.along_cols
    return flt.run(
        stream.lines, along_meas, RATE, ATLAS,
        init_perp=float(tp[0]), init_along=float(ta[0]),
        n_particles=500, perp_spread=calib.ALIAS_SPACING_ROWS,
        along_spread=2.0, line_len=LINE_LEN,
        coarse_anchor=coarse_anchor, seed=seed, **kwargs,
    )


def _make_along_and_coarse(stream, seed=3):
    """Generate noisy along measurement and synthetic coarse anchor."""
    tp = stream.trajectory.perp_rows
    ta = stream.trajectory.along_cols
    rng = np.random.default_rng(seed)
    along_meas = ta + rng.normal(0.0, 1.0, len(ta))
    coarse = tp + rng.normal(0.0, flt.COARSE_SIGMA_ROWS, len(tp))
    return along_meas, coarse


# ---------------------------------------------------------------------------
# Test 1: clean fixation — zero reseeds, G10 accuracy preserved
# ---------------------------------------------------------------------------

def test_clean_fixation_no_reseeds_and_accuracy_preserved():
    """On a clean fixation stream the reseed trigger must NOT fire (zero reseeds)
    and the G10 sub-0.1-deg perp accuracy must be preserved."""
    stream = _make_stream(seed=7)
    tp = stream.trajectory.perp_rows
    along_meas, coarse = _make_along_and_coarse(stream)

    res = _run(stream, along_meas, coarse, seed=0)

    assert res.n_reseeds == 0, (
        f"reseed fired {res.n_reseeds} times on a clean fixation stream "
        f"(trigger should ONLY fire on lock loss, not healthy fixation)")
    assert not res.reseeded.any(), "reseeded array must be all-False on clean stream"

    perp_rms = float(np.sqrt(np.mean((res.est_perp - tp) ** 2)))
    print(f"\n[G11] clean fixation: n_reseeds={res.n_reseeds}, "
          f"perp RMS={perp_rms:.3f} rows ({perp_rms*calib.ARCMIN_PER_ROW:.3f}')")
    assert perp_rms < DEG_THRESH_ROWS, (
        f"G10 accuracy regression: perp RMS {perp_rms:.3f} >= {DEG_THRESH_ROWS:.3f}")


# ---------------------------------------------------------------------------
# Test 2: injected garbage lines → reseed fires, recovery within bound
# ---------------------------------------------------------------------------

def test_injected_lock_loss_reseeds_and_recovers():
    """Replace a contiguous block of lines with pure noise (simulating a lock
    loss / blink / occlusion).  Assert:
    - At least one reseed fires during or shortly after the corrupted block.
    - The filter recovers to sub-0.1 deg within RECOVERY_CAP_STEPS steps after
      real lines resume.
    - The maximum consecutive gross-error run (post-recovery) is bounded.
    """
    stream = _make_stream(seed=7, duration=0.5)
    tp = stream.trajectory.perp_rows
    ta = stream.trajectory.along_cols
    T = len(tp)
    along_meas, coarse = _make_along_and_coarse(stream)

    # Inject garbage: replace t=400..700 with broadband noise
    lines_corrupt = stream.lines.copy()
    rng_g = np.random.default_rng(99)
    noise_std = float(stream.lines.std()) * 3.0
    corrupt_start, corrupt_end = 400, min(700, T)
    lines_corrupt[corrupt_start:corrupt_end] = (
        rng_g.standard_normal((corrupt_end - corrupt_start, LINE_LEN)) * noise_std
    )

    res = flt.run(
        lines_corrupt, along_meas, RATE, ATLAS,
        init_perp=float(tp[0]), init_along=float(ta[0]),
        n_particles=500, perp_spread=calib.ALIAS_SPACING_ROWS, along_spread=2.0,
        line_len=LINE_LEN, coarse_anchor=coarse, seed=0,
    )

    # (a) At least one reseed within the corrupted+recovery window
    reseed_window = res.reseeded[corrupt_start: corrupt_end + RECOVERY_CAP_STEPS]
    assert reseed_window.any(), (
        f"no reseed fired during/after the corrupted block "
        f"(n_reseeds total={res.n_reseeds})")
    first_reseed = int(np.where(res.reseeded)[0][0])
    print(f"\n[G11] first reseed at step {first_reseed} "
          f"(corrupt block {corrupt_start}:{corrupt_end}), "
          f"total reseeds={res.n_reseeds}")

    # (b) Recovery: after corrupt_end, filter reaches sub-0.1 deg and stays there
    #     for 20 consecutive steps within RECOVERY_CAP_STEPS.
    post_err = np.abs(res.est_perp[corrupt_end:] - tp[corrupt_end:])
    recovery_step = None
    for i in range(min(RECOVERY_CAP_STEPS, len(post_err) - 20)):
        if np.all(post_err[i: i + 20] < DEG_THRESH_ROWS):
            recovery_step = i
            break
    print(f"[G11] recovery step (post corrupt_end): {recovery_step} "
          f"(cap {RECOVERY_CAP_STEPS})")
    assert recovery_step is not None, (
        f"filter did not recover to sub-0.1 deg within {RECOVERY_CAP_STEPS} steps "
        f"after lock resumed (post-resume errors: min={post_err.min():.2f} "
        f"p50={np.median(post_err):.2f})")
    assert recovery_step <= RECOVERY_CAP_STEPS, (
        f"recovery took {recovery_step} steps > cap {RECOVERY_CAP_STEPS}")

    # (c) No indefinite coast: longest gross-error run after corrupt_end is bounded
    gross = post_err >= DEG_THRESH_ROWS
    max_run = 0
    run_len = 0
    for g in gross:
        run_len = run_len + 1 if g else 0
        max_run = max(max_run, run_len)
    print(f"[G11] longest gross-error run post-corrupt: {max_run} steps "
          f"({max_run/RATE*1000:.1f} ms)")
    assert max_run <= RECOVERY_CAP_STEPS, (
        f"indefinite coast detected: longest gross run {max_run} > {RECOVERY_CAP_STEPS}")


# ---------------------------------------------------------------------------
# Test 3: forced alias mislock → reseed fires quickly, recovery is immediate
# ---------------------------------------------------------------------------

def test_alias_mislock_triggers_reseed():
    """Teleport the particle cloud to a wrong alias (+160 rows from truth) at
    t=300, then let the real lines continue.  The filter should detect the
    mislock quickly (within NCC_LOSS_WINDOW steps) and reseed."""
    stream = _make_stream(seed=7, duration=0.4)
    tp = stream.trajectory.perp_rows
    ta = stream.trajectory.along_cols
    along_meas, coarse = _make_along_and_coarse(stream)

    # We need direct access to the ParticleFilter to teleport the cloud.
    rng = np.random.default_rng(0)
    dt = 1.0 / RATE
    T = len(tp)
    state = flt.init_filter(500, float(tp[0]), float(ta[0]),
                             calib.ALIAS_SPACING_ROWS, 2.0, rng=rng)
    pf = flt.ParticleFilter(state, ATLAS, LINE_LEN,
                             ncc_loss_thr=flt.NCC_LOCK_LOSS_THR,
                             ncc_loss_window=flt.NCC_LOSS_WINDOW,
                             coast_cap=flt.COAST_CAP)
    MISLOCK_T = 300
    MISLOCK_OFFSET = 160.0   # one alias spacing away

    reseeded_steps = []
    for t in range(T):
        if t == MISLOCK_T:
            # Teleport the cloud to a wrong alias
            pf.state.pos_perp[:] = tp[t] + MISLOCK_OFFSET
            pf.state.pos_along[:] = ta[t]
            pf.state.weight[:] = 1.0 / 500

        ca_t = float(coarse[t])
        post = pf.step(stream.lines[t], float(along_meas[t]), dt, rng,
                       coarse_anchor=ca_t)
        if post.reseeded:
            reseeded_steps.append(t)

    assert len(reseeded_steps) >= 1, (
        "no reseed fired after alias mislock — filter did not detect the lock loss")
    first_reseed = reseeded_steps[0]
    steps_to_reseed = first_reseed - MISLOCK_T
    print(f"\n[G11] alias mislock at t={MISLOCK_T}, "
          f"first reseed at t={first_reseed} "
          f"({steps_to_reseed} steps = {steps_to_reseed/RATE*1000:.2f} ms)")
    # Should detect within a small multiple of NCC_LOSS_WINDOW
    assert steps_to_reseed <= 3 * flt.NCC_LOSS_WINDOW, (
        f"too slow to detect mislock: {steps_to_reseed} steps "
        f"(expected <= {3 * flt.NCC_LOSS_WINDOW})")


# ---------------------------------------------------------------------------
# Test 4: coast cap — no indefinite coast even with permanent garbage
# ---------------------------------------------------------------------------

def test_coast_cap_prevents_indefinite_coast():
    """With permanent garbage lines the reseed must fire at least every
    COAST_CAP steps — no indefinite coast."""
    stream = _make_stream(seed=7, duration=0.5)
    tp = stream.trajectory.perp_rows
    ta = stream.trajectory.along_cols
    T = len(tp)
    along_meas, coarse = _make_along_and_coarse(stream)

    # All garbage lines
    lines_all_garbage = np.random.default_rng(55).standard_normal(
        (T, LINE_LEN)) * float(stream.lines.std()) * 5

    res = flt.run(
        lines_all_garbage, along_meas, RATE, ATLAS,
        init_perp=float(tp[0]), init_along=float(ta[0]),
        n_particles=500, perp_spread=calib.ALIAS_SPACING_ROWS, along_spread=2.0,
        line_len=LINE_LEN, coarse_anchor=coarse, seed=0,
    )

    assert res.n_reseeds >= 1, "no reseed fired during permanent garbage"

    # Max gap between consecutive reseeds must be <= coast_cap
    reseed_steps = np.where(res.reseeded)[0]
    if len(reseed_steps) >= 2:
        gaps = np.diff(reseed_steps)
        max_gap = int(gaps.max())
        print(f"\n[G11] permanent garbage: {res.n_reseeds} reseeds, "
              f"max coast gap={max_gap} (cap={flt.COAST_CAP})")
        assert max_gap <= flt.COAST_CAP, (
            f"coast gap {max_gap} > coast_cap {flt.COAST_CAP}")
    else:
        print(f"\n[G11] permanent garbage: {res.n_reseeds} reseed(s)")


# ---------------------------------------------------------------------------
# Test 5: pre-loss accuracy is preserved (no regression from adding G11)
# ---------------------------------------------------------------------------

def test_pre_loss_accuracy_matches_g10():
    """The portion of the run BEFORE the injected loss must still meet the
    G10 sub-0.1-deg threshold — adding reacquisition must not regress tracking."""
    stream = _make_stream(seed=7, duration=0.5)
    tp = stream.trajectory.perp_rows
    ta = stream.trajectory.along_cols
    T = len(tp)
    along_meas, coarse = _make_along_and_coarse(stream)

    lines_c = stream.lines.copy()
    rng_g = np.random.default_rng(99)
    corrupt_start = min(400, T)
    lines_c[corrupt_start:] = (
        rng_g.standard_normal((T - corrupt_start, LINE_LEN))
        * float(stream.lines.std()) * 3
    )

    res = flt.run(
        lines_c, along_meas, RATE, ATLAS,
        init_perp=float(tp[0]), init_along=float(ta[0]),
        n_particles=500, perp_spread=calib.ALIAS_SPACING_ROWS, along_spread=2.0,
        line_len=LINE_LEN, coarse_anchor=coarse, seed=0,
    )

    pre_rms = float(np.sqrt(np.mean((res.est_perp[:corrupt_start]
                                     - tp[:corrupt_start]) ** 2)))
    print(f"\n[G11] pre-loss perp RMS = {pre_rms:.3f} rows "
          f"({pre_rms * calib.ARCMIN_PER_ROW:.3f}')")
    assert pre_rms < DEG_THRESH_ROWS, (
        f"pre-loss accuracy regression: {pre_rms:.3f} rows >= {DEG_THRESH_ROWS:.3f}")
