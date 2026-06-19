# PF de-jittering report

## Root cause (confirmed)

Fixation jitter on `people_data_fov` is a **filter-gain** problem, not an IMM prior problem.

- OU prior predicts ~4×10⁻⁴ rows/step during pursuit lock.
- Per-line NCC localization noise is ~4–6 rows (~2–3′).
- **`BETA=20`** over-sharpens weights → ESS→1 every healthy step → resample wipes temporal memory.
- **`roughen_perp=0.5`** injects ~0.5 rows/step of artificial bandwidth.

## Step 0 — Noisy-static ablation (`tests/test_jitter_ablation.py`)

Per-line observation noise (same true gaze, independent speckle each step):

| Run | Config | std (rows) | med \|Δ²\| |
|-----|--------|------------|------------|
| A | default BETA=20, rp=0.5 | 0.061 | 0.076 |
| B | + resample off | 0.039 | 0.000 |
| C | + roughen=0 | 0.015 | 0.001 |
| D | lock-gated gain | 0.009 | 0.001 |
| E | BETA=5, rp=0.08 | 0.014 | 0.003 |

Lock-gated and low-BETA configs reduce HF step noise by ~10× vs default.

## Step 2 — Igor BETA × roughen sweep (15 s first pass)

Grid: BETA ∈ {3,5,8,20} × roughen ∈ {0.05,0.15,0.5} + lock-gated row.  
Data: `cache/people_fov/Igor/`, pursuit_fov, ~218k lines / 15 s.

**Best guardrailed cell:** `b3_rp0.15` — prec_x **3.29′**, j30 **1.20′**, r_dot_x **0.89** (baseline Igor ~0.80).

| Trend | Observation |
|-------|-------------|
| BETA ↓ | prec_x improves monotonically at fixed roughen (e.g. rp=0.05: 3.30→3.74′ for BETA 3→20) |
| roughen ↓ | modest HF gain at low BETA; at BETA=20 effect smaller |
| lock-gated | prec_x=3.50′ — better than BETA=20 defaults, not best vs fixed BETA=3 |

Artifacts: `results/pf_dejitter_sweep_Igor_d15.csv`, `results/pf_dejitter_sweep_Igor_d15.png`

**Recommended operating point (15 s Igor):** `beta=3`, `roughen_perp=0.15` or lock-gated for reseed/saccade safety.

## Step 4 — Ashton3 M4 vs M5 (`people_fov_m5.py`)

Batch MAP smoother on full capture (~1M lines):

| Metric | M4 (default) | M5 MAP | M5/M4 |
|--------|--------------|--------|-------|
| HF prec_x (>25 ms) | 2.69′ | 3.70′ | 1.37 |
| Frame jitter @30 fps | 0.93′ | 0.85′ | 0.91 |
| r vs dot (x) | 0.943 | 0.941 | 1.00 |
| PSD HF power | 0.089 | 0.209 | 2.34 |
| valid fraction | 0.932 | 0.944 | 1.01 |

M5 slightly reduces display-frame jitter but **does not** beat M4 on HF precision on this capture; MAP data term at β=4 still leaves HF in the horizontal trace. Pursuit tracking (r vs dot) preserved.

## Implementation summary

| File | Change |
|------|--------|
| `filter.py` | lock-gated BETA/roughen, `resample_enabled` |
| `khz2d_methods.py::m4_dpf` | kwargs + `ess`/`p_saccade` cache export |
| `people_fov_pf.py` | shared people M4 harness + metrics |
| `pf_gain_sweep.py` | BETA×roughen grid runner |
| `people_fov_m5.py` | people_fov M5 adapter + M4 compare |
| `people_data_fov_run.py` | restored batch runner (`--lock-gated`, `--cache-tag`) |
| `people_data_fov_anim.py` | `--cache-tag` for alternate caches |

## Next steps

1. Full-length Igor rerun on top 3 cells: `python pf_gain_sweep.py --person Igor --dur 15 --full`
2. Re-run Ashton3 with lock-gated M4: `python people_data_fov_run.py --person Ashton3 --lock-gated --cache-tag m4_dpf_physics_lg`
3. Regenerate animation: `python people_data_fov_anim.py --person Ashton3 --cache-tag m4_dpf_physics_lg --style raw`

## Deferred

- `ess_frac=0.7` / `nw3` (test1 reacquisition, not de-jitter)
- Tremor IMM (would inject HF)
- Along-informed saccade direction (through-saccade blur, separate issue)
- Observation blocking / Rao-Blackwellized hybrid
