# Scalar Visual-Attention Metric a(t) — Teleop Imitation-Learning Feature

## Definition

`a(t) = clip(smooth(g_intake * g_lock, TAU_ATT), 0, 1)` where

- **g_intake** = `0.5*(1 - tanh((speed - V_HALF)/(0.5*V_HALF)))`, V_HALF = 15.0 deg/s,

- **g_lock** = `clip((max_ncc - 0.12)/( 0.6 - 0.12 ), 0, 1)`,

- smoothed over an attentional-dwell window TAU_ATT = 50 ms.

Intake half-speed V_HALF was set adaptively to 15.0 deg/s (the 95th pct of valid gaze speed, floored at 15); the reconstruction attenuates true saccade peak velocity so this sits at the actual pursuit<->saccade boundary.

## Intake gate falls monotonically with gaze speed

| gaze speed band (deg/s) | mean g_intake | n |
|---|---|---|
| 0-5 | 0.975 | 478383 |
| 5-15 | 0.829 | 77564 |
| 15-30 | 0.250 | 17261 |
| 30-60 | 0.006 | 3342 |
| 60-inf | 0.000 | 64 |

## Behaviour (validation — necessary, not sufficient)

| quantity | a(t) |
|---|---|
| fixation / slow pursuit (speed < 5 deg/s, n=478383) | 0.706 |
| in-flight saccade (speed > 30 deg/s, n=3406) | 0.508 |
| blink / out-of-FOV (invalid) | 0.000 |
| valid | 0.666 |
| right/temporal gaze | 0.403 |
| left/centre gaze | 0.736 |

a(t): N=827392 @ 11823 Hz, mean=0.464, frac>0.5=52%; resampled to 100 Hz -> 6998 steps.
